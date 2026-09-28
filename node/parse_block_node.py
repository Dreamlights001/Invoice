"""分块解析节点：逐块抽取字段，并在需要时引入 RAG few-shot 参考。"""
import json
from typing import Any, Dict, List, Optional, Tuple

from .ocr_node import Config, InvoiceState, LLMService
from .rag_store import RAGStore, normalize_rag_doc_id


BLOCK_SCHEMA = """
{
  "buyerName": "购方名称",
  "sellerName": "销方名称",
  "invoiceNumber": "发票号码",
  "invoiceDate": "发票日期，格式 YYYY-MM-DD",
  "totalAmount": "总金额",
  "currency": "币种，例如 USD、CNY、EUR",
  "details": [
    {
      "itemNo": "物料编号或项目编号",
      "deliveryNote": "送货单号",
      "quantity": "数量",
      "price": "单价",
      "amount": "金额",
      "orderNo": "订单号或采购单号"
    }
  ]
}
"""

BLOCK_PARSE_PROMPT = """你是一名发票字段抽取专家。

请依据给定 schema，从当前文本块中提取尽可能准确的字段信息。

要求：
1. 仅输出 JSON 对象。
2. 字段值必须来自当前文本块，不能凭空猜测。
3. 如果某字段无法确认，请省略该字段或填空，不要臆造。
4. 日期尽量规范为 YYYY-MM-DD。
5. 明细字段仅在当前块确实包含明细内容时提取。

Schema:
{schema}

{fewshot_section}当前块的分块理由：
{query_reason}

当前块文本：
{query_block}
"""


def parse_block_node(state: InvoiceState, config: Config, rag_store: Optional[RAGStore] = None) -> InvoiceState:
    """逐个解析分块，得到块级结构化结果。"""
    block_map = state.get("blocks")
    if not block_map:
        state["errors"].append("没有找到分块数据，无法进行块级解析")
        return state

    prompt_bucket = state.setdefault("prompts", {}).setdefault("block_parses", {})
    parser = LLMService(config.llm)
    results: Dict[str, Dict[str, Any]] = {}
    current_doc_key = normalize_rag_doc_id(state.get("doc_id"))

    for block_id, block_data in block_map.items():
        reason, text = _read_block_inputs(block_data)
        fewshot_examples: List[Dict[str, Any]] = []
        fewshot_text = ""
        mode = "base"
        if rag_store is not None:
            try:
                retrieved = rag_store.retrieve_block_examples(
                    query_text=text,
                    k=5,
                    exclude_doc_id=state.get("doc_id"),
                )
                fewshot_examples = _filter_block_examples(retrieved, current_doc_key)
                fewshot_text = _build_block_fewshot(fewshot_examples)
                mode = "rag" if fewshot_examples else "rag_fallback"
            except Exception as exc:
                state["errors"].append(f"RAG解析 {block_id} 失败: {exc}")
                fewshot_examples = []
                fewshot_text = ""
                mode = "rag_fallback"

        prompt = _render_block_prompt(reason, text, fewshot_text)
        trace = {
            "prompt": prompt,
            "mode": mode,
            "retrieved_blocks_count": len(fewshot_examples),
        }
        prompt_bucket[block_id] = trace

        try:
            raw_result = parser.invoke(prompt, parse_json=True)
            trace["response"] = _serialize(raw_result)
            results[block_id] = _coerce_parse_result(raw_result, parser)
            if results[block_id] == {} and not isinstance(raw_result, dict):
                state["errors"].append(f"无法解析块 {block_id}")
        except Exception as exc:
            trace["error"] = str(exc)
            state["errors"].append(f"解析块 {block_id} 失败: {exc}")
            results[block_id] = {}

    state["block_parses"] = results
    state.setdefault("metadata", {})["parse_mode"] = "rag" if rag_store is not None else "base"
    return state


def _read_block_inputs(block_data: Dict[str, Any]) -> Tuple[str, str]:
    """读取分块中的 reason 和 text。"""
    return (
        str(block_data.get("REASON", "") or ""),
        str(block_data.get("TEXT", "") or ""),
    )


def _render_block_prompt(reason: str, block_text: str, fewshot_text: str) -> str:
    """构造分块解析提示词。"""
    fewshot_section = ""
    if fewshot_text:
        fewshot_section = "以下是检索到的相似块解析示例，可作为参考：\n" + fewshot_text + "\n\n"
    return BLOCK_PARSE_PROMPT.format(
        schema=BLOCK_SCHEMA,
        fewshot_section=fewshot_section,
        query_reason=reason,
        query_block=block_text,
    )


def _build_block_fewshot(retrieved_blocks: List[Dict[str, Any]]) -> str:
    """将检索到的相似文本块拼接为 few-shot 示例。"""
    fragments: List[str] = []
    for index, block in enumerate(retrieved_blocks[:5], start=1):
        fragments.append(
            "\n".join(
                [
                    f"示例块{index}：doc_id: {block.get('doc_id', '')}",
                    f"block_id: {block.get('block_id', '')}",
                    f"distance: {block.get('distance', '')}",
                    f"reason: {block.get('reason', '')}",
                    f"text: {block.get('text', '')}",
                    f"parsed: {block.get('parsed', '')}",
                ]
            )
        )
    return "\n\n".join(fragments)


def _filter_block_examples(retrieved_blocks: List[Dict[str, Any]], current_doc_key: str) -> List[Dict[str, Any]]:
    """过滤掉当前文档自身及重复块。"""
    filtered: List[Dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for block in retrieved_blocks:
        if not isinstance(block, dict):
            continue
        example_key = normalize_rag_doc_id(str(block.get("doc_id") or ""))
        block_id = str(block.get("block_id") or "")
        pair = (example_key, block_id)
        if not example_key or example_key == current_doc_key or pair in seen_pairs:
            continue
        if not str(block.get("text") or "").strip():
            continue
        seen_pairs.add(pair)
        filtered.append(block)
    return filtered


def _coerce_parse_result(raw_result: Any, service: LLMService) -> Dict[str, Any]:
    """将模型返回规整为字典。"""
    if isinstance(raw_result, dict):
        return raw_result
    extracted = service.extract_json(str(raw_result))
    return extracted if isinstance(extracted, dict) else {}


def _serialize(value: Any) -> str:
    """将解析结果序列化为日志字符串。"""
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)
