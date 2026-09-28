"""分块节点：根据 OCR 文本完成发票文本分块，并在需要时引入 RAG few-shot 参考。"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .ocr_node import Config, InvoiceState, LLMService
from .rag_store import RAGStore, normalize_rag_doc_id


CHUNKING_PROMPT = """你是一名发票文本分块专家。
请完成以下任务：
1. 阅读原始 OCR 文本。
2. 按语义和版面结构将内容划分为若干个 BLOCK。
3. 仅输出 JSON，不要输出额外解释。

输出要求：
1. 返回合法 JSON 对象。
2. 键名使用 BLOCK_1、BLOCK_2 等连续编号。
3. 每个 BLOCK 必须包含两个字段：
   - REASON: 说明为什么这样分块。
   - TEXT: 该块对应的原始文本内容。
{fewshot_section}待分块 OCR 文本如下：
{ocr}
"""


def chunk_node(
    state: InvoiceState,
    config: Config,
    rag_store: Optional[RAGStore] = None,
    fewshot_dir: Optional[Path] = None,
) -> InvoiceState:
    """根据 OCR 文本生成分块结果。"""
    source_text = str(state.get("ocr_raw_text") or "")
    if not source_text.strip():
        state["errors"].append("无OCR文本可供分块")
        return state

    fewshot_examples: List[Dict[str, Any]] = []
    mode = "base"
    current_doc_key = normalize_rag_doc_id(state.get("doc_id"))

    if fewshot_dir is not None:
        try:
            cached_examples = _load_precomputed_doc_examples(fewshot_dir, state["doc_id"])
            cached_examples = _filter_doc_examples(cached_examples, current_doc_key)
            if cached_examples:
                fewshot_examples = cached_examples
                mode = "rag_precomputed"
        except Exception as exc:
            state["errors"].append(f"预计算few-shot加载失败: {exc}")

    if not fewshot_examples and rag_store is not None:
        try:
            live_examples = rag_store.retrieve_doc_examples(
                query_text=source_text,
                k=3,
                exclude_doc_id=state.get("doc_id"),
            )
            fewshot_examples = _filter_doc_examples(live_examples, current_doc_key)
            mode = "rag_live" if fewshot_examples else "rag_fallback"
        except Exception as exc:
            state["errors"].append(f"RAG chunk retrieval failed: {exc}")
            mode = "rag_fallback"

    fewshot_text = _build_doc_fewshot(fewshot_examples)
    prompt = _render_chunk_prompt(source_text, fewshot_text)
    prompt_trace = {
        "prompt": prompt,
        "mode": mode,
        "fewshot_count": len(fewshot_examples),
    }
    state.setdefault("prompts", {})["chunk"] = prompt_trace
    state["rag_doc_examples"] = [
        {"doc_id": item.get("doc_id"), "distance": item.get("distance")}
        for item in fewshot_examples
    ] or None

    try:
        llm_result = LLMService(config.llm).invoke(prompt, parse_json=True)
        prompt_trace["response"] = _serialize_for_log(llm_result)
        parsed_candidate = _extract_chunk_response(llm_result)
        parsed_blocks = _coerce_blocks(parsed_candidate)
        if parsed_blocks is None and mode != "base":
            fallback_prompt = _render_chunk_prompt(source_text, "")
            prompt_trace["fallback_prompt"] = fallback_prompt
            llm_result = LLMService(config.llm).invoke(fallback_prompt, parse_json=True)
            prompt_trace["fallback_response"] = _serialize_for_log(llm_result)
            parsed_candidate = _extract_chunk_response(llm_result)
            parsed_blocks = _coerce_blocks(parsed_candidate)
            if parsed_blocks is not None:
                mode = f"{mode}_fallback_base"
        if parsed_blocks is None:
            state["errors"].append("LLM的回复无法解析为有效的块结构")
            state["blocks"] = None
            return state
        if not parsed_blocks:
            state["errors"].append("分块未产生有效的块")
            state["blocks"] = None
            return state

        state["blocks"] = parsed_blocks
        state.setdefault("metadata", {})["block_count"] = len(parsed_blocks)
        state["metadata"]["chunk_mode"] = mode
    except Exception as exc:
        if mode != "base" and _should_retry_without_fewshot(exc):
            try:
                fallback_prompt = _render_chunk_prompt(source_text, "")
                prompt_trace["fallback_prompt"] = fallback_prompt
                llm_result = LLMService(config.llm).invoke(fallback_prompt, parse_json=True)
                prompt_trace["fallback_response"] = _serialize_for_log(llm_result)
                parsed_candidate = _extract_chunk_response(llm_result)
                parsed_blocks = _coerce_blocks(parsed_candidate)
                if parsed_blocks:
                    state["blocks"] = parsed_blocks
                    state.setdefault("metadata", {})["block_count"] = len(parsed_blocks)
                    state["metadata"]["chunk_mode"] = f"{mode}_fallback_base"
                    return state
            except Exception as fallback_exc:
                prompt_trace["fallback_error"] = str(fallback_exc)
        state["blocks"] = None
        state["errors"].append(f"分块失败: {exc}")
        prompt_trace["error"] = str(exc)

    return state


def _render_chunk_prompt(ocr_text: str, fewshot_text: str) -> str:
    """构造分块提示词。"""
    fewshot_section = ""
    if fewshot_text:
        fewshot_section = "以下是检索到的相似发票分块示例，可作为参考：\n" + fewshot_text + "\n\n"
    return CHUNKING_PROMPT.format(ocr=ocr_text, fewshot_section=fewshot_section)


def _build_doc_fewshot(retrieved: List[Dict[str, Any]]) -> str:
    """将检索到的相似整单样本拼接为 few-shot 文本。"""
    fragments: List[str] = []
    for index, item in enumerate(retrieved[:2], start=1):
        blocks_json = item.get("blocks_json") or {}
        doc_ocr = _trim_fewshot_text(str(item.get("doc_ocr") or "").strip(), 900)
        if not blocks_json:
            continue
        snippet_parts = [
            f"示例{index}：",
            f"doc_id: {item.get('doc_id', '')}",
            f"distance: {item.get('distance', '')}",
        ]
        if doc_ocr:
            snippet_parts.extend(["<doc_ocr>", doc_ocr, "</doc_ocr>"])
        snippet_parts.extend([
            "blocks_json:",
            json.dumps(blocks_json, ensure_ascii=False, indent=2),
        ])
        fragments.append("\n".join(snippet_parts))
    return "\n\n".join(fragments)


def _load_precomputed_doc_examples(fewshot_dir: Optional[Path], doc_id: str) -> List[Dict[str, Any]]:
    """从 fewshot_precomputed 中读取当前文档对应的预计算示例。"""
    if fewshot_dir is None or not fewshot_dir.exists():
        return []

    candidates = [
        fewshot_dir / f"{doc_id}.fewshot.json",
        fewshot_dir / f"{doc_id}.deepseek_ocr_raw.fewshot.json",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        retrieved = payload.get("retrieve", [])
        if isinstance(retrieved, dict):
            retrieved = retrieved.get("rerank") or retrieved.get("embedding_candidates") or []
        if not isinstance(retrieved, list):
            return []
        result: List[Dict[str, Any]] = []
        for item in retrieved:
            if not isinstance(item, dict):
                continue
            result.append(
                {
                    "doc_id": item.get("doc_id", ""),
                    "distance": item.get("distance", 0.0),
                    "doc_ocr": item.get("doc_ocr", ""),
                    "blocks_json": item.get("blocks_json", {}),
                }
            )
        return result
    return []


def _filter_doc_examples(retrieved: List[Dict[str, Any]], current_doc_key: str) -> List[Dict[str, Any]]:
    """过滤掉当前文档自身及重复样本。"""
    filtered: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in retrieved:
        if not isinstance(item, dict):
            continue
        example_key = normalize_rag_doc_id(str(item.get("doc_id") or ""))
        if not example_key or example_key == current_doc_key or example_key in seen:
            continue
        if not item.get("blocks_json"):
            continue
        seen.add(example_key)
        filtered.append(item)
    return filtered


def _should_retry_without_fewshot(exc: Exception) -> bool:
    """判断是否应该在遇到错误时重试一次，去掉 few-shot 示例以规避可能的内容过滤问题。"""
    message = str(exc)
    return "contentFilter" in message or "1301" in message or "400" in message


def _extract_chunk_response(value: Any) -> Any:
    """提取分块响应内容"""
    if isinstance(value, dict):
        return value

    raw_text = str(value or "").strip()
    parsed = LLMService.extract_json(raw_text)
    if isinstance(parsed, dict):
        return parsed

    cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    candidates = [cleaned, _slice_json_object(cleaned), _slice_json_object(raw_text)]
    for candidate in candidates:
        if not candidate:
            continue
        sanitized = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', candidate)
        try:
            parsed = json.loads(sanitized)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return value


def _slice_json_object(text: str) -> str:
    """从文本中提取 JSON，必须以"{"开头，以"}"结尾。"""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return ""
    return text[start : end + 1]


def _trim_fewshot_text(text: str, limit: int) -> str:
    """修改few-shot文本，避免OCR文本过长。"""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _coerce_blocks(candidate: Any) -> Optional[Dict[str, Dict[str, str]]]:
    """将模型返回规整为 BLOCK 字典。"""
    if not isinstance(candidate, dict):
        return None

    normalized: Dict[str, Dict[str, str]] = {}
    for position, pair in enumerate(candidate.items(), start=1):
        block_name, block_payload = pair
        block_data = _coerce_single_block(block_payload)
        if block_data is None:
            continue
        normalized[_canonical_block_name(str(block_name), position)] = block_data
    return normalized


def _coerce_single_block(value: Any) -> Optional[Dict[str, str]]:
    """校验单个分块是否同时包含 REASON 和 TEXT。"""
    if not isinstance(value, dict):
        return None

    upper_map = {str(key).strip().upper(): value[key] for key in value}
    if "REASON" not in upper_map or "TEXT" not in upper_map:
        return None

    return {
        "REASON": str(upper_map.get("REASON", "") or ""),
        "TEXT": str(upper_map.get("TEXT", "") or ""),
    }


def _canonical_block_name(raw_name: str, fallback_index: int) -> str:
    """将块名统一整理为 BLOCK_n 形式。"""
    name = raw_name.strip().upper()
    if name.startswith("BLOCK_"):
        return name

    matched = re.search(r"(\d+)", name)
    if matched:
        return f"BLOCK_{matched.group(1)}"
    return f"BLOCK_{fallback_index}"


def _serialize_for_log(value: Any) -> str:
    """将返回值序列化为便于落盘的字符串。"""
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)
