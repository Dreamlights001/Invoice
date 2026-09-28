"""合并节点，以及流程执行和结果落盘工具。"""
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

from .chunk_node import chunk_node
from .evaluate_node import evaluate_node
from .ocr_node import Config, InvoiceState, LLMService, build_initial_state, ocr_node
from .parse_block_node import parse_block_node
from .rag_store import RAGStore
from .reflect_node import reflect_node


MERGE_SCHEMA = """
{
  "buyerName": "购买方名称",
  "sellerName": "销售方名称",
  "invoiceNumber": "发票号码",
  "invoiceDate": "发票日期，统一输出为 YYYY-MM-DD",
  "totalAmount": "发票总金额",
  "currency": "币种代码",
  "details": [
    {
      "itemNo": "明细行号",
      "deliveryNote": "送货单号",
      "quantity": "数量",
      "price": "单价",
      "amount": "金额",
      "orderNo": "订单号"
    }
  ]
}
"""

MERGE_PROMPT = """你是发票整单归并专家。

你将收到整份文档的 OCR 文本、部分定位框信息、文本分块结果以及每个分块的结构化解析。
请综合所有信息，输出最终发票 JSON。

要求：
1. 只输出 JSON 对象。
2. 顶层必须包含 REASON 和 INVOICE 两个字段。
3. REASON 用中文说明最终取值依据。
4. INVOICE 必须符合 schema。
5. 优先使用全局上下文纠正 OCR 带来的数字、日期和币种噪声。

Schema:
{schema}

全文 OCR：
{text}

定位框摘要：
{bboxes}

分块与解析结果：
{blocks_and_parses}
"""

_METHOD_OPTIONS = {"base", "rag_only", "reflect_only", "rag_reflect"}


def merge_node(state: InvoiceState, config: Config) -> InvoiceState:
    """整合全文、定位框和块级解析结果，生成最终发票。"""
    blocks = state.get("blocks")
    block_parses = state.get("block_parses")
    if not blocks or not block_parses:
        state["errors"].append("缺少分块或块解析结果，无法进行合并")
        return state

    prompt = _build_merge_prompt(
        raw_text=str(state.get("ocr_raw_text") or ""),
        atoms=state.get("ocr_atoms"),
        blocks=blocks,
        block_parses=block_parses,
    )
    merge_trace = {"prompt": prompt, "blocks_count": len(blocks)}
    state.setdefault("prompts", {})["merge"] = merge_trace

    try:
        llm_output = LLMService(config.llm).invoke(prompt, parse_json=True)
        merge_trace["response"] = _stringify(llm_output)
        if not isinstance(llm_output, dict):
            state["errors"].append("从LLM回复中解析合并后的发票失败")
            state["merged_invoice"] = None
            return state
        state["merged_invoice"] = llm_output
        state["final_invoice"] = llm_output
        state.setdefault("metadata", {})["final_invoice_source"] = "merge"
    except Exception as exc:
        merge_trace["error"] = str(exc)
        state["errors"].append(f"合并失败: {exc}")
        state["merged_invoice"] = None
        state["final_invoice"] = None

    return state


def save_outputs(state: InvoiceState, output_dir: Path) -> None:
    """将流程中的关键产物写入输出目录。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_id = state["doc_id"]
    final_invoice = state.get("final_invoice") or state.get("merged_invoice")

    payload_files = {
        f"{doc_id}.blocks.json": state.get("blocks"),
        f"{doc_id}.block_parses.json": state.get("block_parses"),
        f"{doc_id}.merged.json": state.get("merged_invoice"),
        f"{doc_id}.reflected.json": state.get("reflected_invoice"),
        f"{doc_id}.invoice.json": final_invoice,
        f"{doc_id}.evaluation.json": state.get("evaluation_result"),
    }
    for file_name, payload in payload_files.items():
        if payload is None:
            continue
        _write_json(output_dir / file_name, payload)

    _write_json(
        output_dir / f"{doc_id}.metadata.json",
        {
            "doc_id": doc_id,
            "method": state.get("method", "base"),
            "errors": state.get("errors", []),
            "metadata": state.get("metadata", {}),
        },
    )

    _dump_prompt_traces(output_dir / "prompts", doc_id, state.get("prompts", {}))


def process_invoice(
    input: Optional[Path] = None,
    config: Optional[Config] = None,
    truth: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
    intervar: Optional[Path] = None,
    method: str = "base",
    **aliases: Any,
) -> InvoiceState:
    """按固定顺序执行节点流程，兼容新旧入口参数命名。"""
    resolved_input = input or aliases.get("input_file")
    resolved_output = output_dir or aliases.get("output")
    resolved_intervar = intervar or aliases.get("intervar_dir") or aliases.get("intervar")
    resolved_method = str(aliases.get("strategy") or aliases.get("method") or method)
    if resolved_input is None:
        raise ValueError("input is required")
    if config is None:
        raise ValueError("config is required")
    if resolved_method not in _METHOD_OPTIONS:
        raise ValueError(f"unsupported method: {resolved_method}")

    state = build_initial_state(input=Path(resolved_input), truth=truth, method=resolved_method)
    rag_store = _build_rag_store_if_needed(config, resolved_method, state)
    use_rag = resolved_method in {"rag_only", "rag_reflect"}
    use_reflect = resolved_method in {"reflect_only", "rag_reflect"}
    pipeline = _pipeline_steps(
        config=config,
        intervar=resolved_intervar,
        use_rag=use_rag,
        use_reflect=use_reflect,
        rag_store=rag_store,
        fewshot_dir=config.fewshot_dir,
    )
    for step in pipeline:
        state = step(state)

    if resolved_output is not None:
        save_outputs(state, Path(resolved_output))
    return state


def _pipeline_steps(
    config: Config,
    intervar: Optional[Path],
    use_rag: bool,
    use_reflect: bool,
    rag_store: Optional[RAGStore],
    fewshot_dir: Optional[Path],
) -> Iterable[Callable[[InvoiceState], InvoiceState]]:
    """返回完整的执行序列。"""
    steps: list[Callable[[InvoiceState], InvoiceState]] = [
        lambda current: ocr_node(current, config, intervar=intervar),
        lambda current: chunk_node(current, config, rag_store=rag_store if use_rag else None, fewshot_dir=fewshot_dir if use_rag else None),
        lambda current: parse_block_node(current, config, rag_store=rag_store if use_rag else None),
        lambda current: merge_node(current, config),
    ]
    if use_reflect:
        steps.append(lambda current: reflect_node(current, config))
    steps.append(evaluate_node)
    return tuple(steps)


def _build_rag_store_if_needed(config: Config, method: str, state: InvoiceState) -> Optional[RAGStore]:
    """仅在启用 RAG 的方法下初始化检索对象。"""
    if method not in {"rag_only", "rag_reflect"}:
        return None
    if config.rag_db_path is None:
        state["errors"].append("已启用 RAG 方法，但未找到知识库文件")
        return None
    try:
        return RAGStore(config.rag_db_path, config.embedding)
    except Exception as exc:
        state["errors"].append(f"RAG 初始化失败: {exc}")
        return None


def _build_merge_prompt(
    raw_text: str,
    atoms: Optional[list[Dict[str, Any]]],
    blocks: Dict[str, Dict[str, str]],
    block_parses: Dict[str, Dict[str, Any]],
) -> str:
    """构造整单归并提示词。"""
    return MERGE_PROMPT.format(
        schema=MERGE_SCHEMA,
        text=raw_text,
        bboxes=_summarize_bboxes(atoms),
        blocks_and_parses=_combine_blocks_and_results(blocks, block_parses),
    )


def _combine_blocks_and_results(
    blocks: Dict[str, Dict[str, str]],
    block_parses: Dict[str, Dict[str, Any]],
) -> str:
    """把分块与对应解析结果拼成提示词上下文。"""
    fragments = []
    for block_id, block in blocks.items():
        fragments.append(
            "\n".join(
                [
                    f"{block_id}:",
                    f"REASON: {block.get('REASON', '')}",
                    f"TEXT: {block.get('TEXT', '')}",
                    f"PARSED: {json.dumps(block_parses.get(block_id, {}), ensure_ascii=False)}",
                ]
            )
        )
    return "\n".join(fragments)


def _summarize_bboxes(atoms: Optional[list[Dict[str, Any]]]) -> str:
    """压缩 OCR 坐标信息，避免提示词过长。"""
    if not atoms:
        return "无边界框信息"

    lines = []
    for atom in atoms[:50]:
        atom_type = atom.get("type", "unknown")
        atom_box = atom.get("box_2d", [])
        atom_text = str(atom.get("text", ""))[:50]
        lines.append(f"Type: {atom_type}, Box: {atom_box}, Text: {atom_text}")
    return "\n".join(lines)


def _dump_prompt_traces(prompts_dir: Path, doc_id: str, prompts: Dict[str, Any]) -> None:
    """将各阶段 prompt 与 response 保存到 prompts 子目录。"""
    prompts_dir.mkdir(exist_ok=True)

    chunk_trace = prompts.get("chunk")
    if isinstance(chunk_trace, dict):
        _write_text(prompts_dir / f"{doc_id}.chunk.prompt.txt", chunk_trace.get("prompt"))
        _write_text(prompts_dir / f"{doc_id}.chunk.response.txt", chunk_trace.get("response"))

    block_trace = prompts.get("block_parses")
    if isinstance(block_trace, dict):
        for block_id, trace in block_trace.items():
            if not isinstance(trace, dict):
                continue
            _write_text(prompts_dir / f"{doc_id}__{block_id}.prompt.txt", trace.get("prompt"))
            _write_text(prompts_dir / f"{doc_id}__{block_id}.response.txt", trace.get("response"))

    merge_trace = prompts.get("merge")
    if isinstance(merge_trace, dict):
        _write_text(prompts_dir / f"{doc_id}.merge.prompt.txt", merge_trace.get("prompt"))
        _write_text(prompts_dir / f"{doc_id}.merge.response.txt", merge_trace.get("response"))

    reflect_trace = prompts.get("reflect")
    if isinstance(reflect_trace, dict):
        _write_text(prompts_dir / f"{doc_id}.reflect.prompt.txt", reflect_trace.get("prompt"))
        _write_text(prompts_dir / f"{doc_id}.reflect.response.txt", reflect_trace.get("raw_response"))
        _write_text(prompts_dir / f"{doc_id}.reflect.response.json", reflect_trace.get("response"))

    _write_json(prompts_dir / f"{doc_id}.prompts_meta.json", prompts)


def _write_json(path: Path, payload: Any) -> None:
    """写出 JSON 文件。"""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, content: Any) -> None:
    """仅在内容非空时写出文本文件。"""
    if content is None:
        return
    path.write_text(str(content), encoding="utf-8")


def _stringify(value: Any) -> str:
    """将对象转换为调试日志文本。"""
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)
