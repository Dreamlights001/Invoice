"""视觉反思节点：使用原始图像或 PDF 对整单结果做二次核验。"""
import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from .ocr_node import Config, InvoiceState, LLMConfig, LLMService


_IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

REFLECT_SCHEMA = """
{
  "buyerName": "购买方名称",
  "sellerName": "销售方名称",
  "invoiceNumber": "发票号码",
  "invoiceDate": "发票日期，统一输出为 YYYY-MM-DD",
  "totalAmount": "发票总金额",
  "currency": "币种代码，例如 USD、CNY、EUR",
  "details": [
    {
      "itemNo": "明细行号，可为空",
      "deliveryNote": "送货单号，可为空",
      "quantity": "数量",
      "price": "单价",
      "amount": "金额",
      "orderNo": "订单号，可为空"
    }
  ]
}
"""

REFLECT_PROMPT = """你是一名发票核验专家。

你会收到原始发票图像以及当前自动抽取得到的发票 JSON。请逐字段核验并修正结果。

要求：
1. 只输出 JSON 对象。
2. 顶层必须包含 REASON、CORRECTIONS、INVOICE 三个字段。
3. 如果某字段在图像中无法确认，不得臆造。
4. 重点检查买卖双方名称、发票号、日期、总金额、币种和明细表。
5. 若发现 orderNo、deliveryNote、数量、单价、金额存在 OCR 或解析错误，应据图像纠正。

当前自动抽取结果：
{current_json}

Schema:
{schema}
"""


def _resolve_reflect_config(config: Config) -> LLMConfig:
    """优先使用反思节点配置，缺失时回退到通用 LLM 配置。"""
    if config.reflect_llm.api_url and config.reflect_llm.api_key and config.reflect_llm.model:
        return config.reflect_llm
    return config.llm


def reflect_node(state: InvoiceState, config: Config) -> InvoiceState:
    """对整单结果做视觉复核，并生成反思后的最终发票。"""
    merged_payload = state.get("merged_invoice")
    if not merged_payload:
        state["errors"].append("Reflect node skipped: no merged invoice")
        return state

    reflect_config = _resolve_reflect_config(config)
    if not reflect_config.api_url or not reflect_config.model or not reflect_config.api_key:
        state["errors"].append("Reflect node skipped: reflect_LLM_API_URL、reflect_LLM_MODEL、reflect_LLM_API_KEY 未完整配置")
        state["final_invoice"] = merged_payload
        return state

    try:
        prompt_text, raw_response, parsed = _call_reflect_model(state["input"], merged_payload, reflect_config)
        state.setdefault("prompts", {})["reflect"] = {
            "model": reflect_config.model,
            "prompt": prompt_text,
            "raw_response": raw_response,
            "response": json.dumps(parsed, ensure_ascii=False, indent=2) if isinstance(parsed, dict) else raw_response,
        }

        if isinstance(parsed, dict):
            state["reflected_invoice"] = parsed
            state["final_invoice"] = parsed
            state.setdefault("metadata", {})["final_invoice_source"] = "reflect"
        else:
            state["errors"].append("Reflect node: failed to parse reflect response")
            state["reflected_invoice"] = None
            state["final_invoice"] = merged_payload
            state.setdefault("metadata", {})["final_invoice_source"] = "merge"
    except Exception as exc:
        state["errors"].append(f"Reflect node failed: {exc}")
        state["reflected_invoice"] = None
        state["final_invoice"] = merged_payload
        state.setdefault("metadata", {})["final_invoice_source"] = "merge"

    return state


def _call_reflect_model(file_path: Path, current_invoice: Dict[str, Any], config: LLMConfig) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    """调用多模态模型完成反思核验。"""
    data_urls = _build_image_data_urls(file_path)
    invoice_payload = current_invoice.get("INVOICE") if isinstance(current_invoice, dict) and "INVOICE" in current_invoice else current_invoice
    prompt_text = REFLECT_PROMPT.format(
        current_json=json.dumps(invoice_payload, ensure_ascii=False, indent=2),
        schema=REFLECT_SCHEMA,
    )

    client = OpenAI(api_key=config.api_key, base_url=config.api_url)
    content: List[Dict[str, Any]] = [{"type": "image_url", "image_url": {"url": url}} for url in data_urls]
    content.append({"type": "text", "text": prompt_text})
    response = client.chat.completions.create(
        model=config.model,
        temperature=0.0,
        messages=[{"role": "user", "content": content}],
        max_tokens=8192,
    )
    raw_text = response.choices[0].message.content or ""
    return prompt_text, raw_text, LLMService.extract_json(raw_text)


def _build_image_data_urls(file_path: Path) -> List[str]:
    """将 PDF 或图片转成模型可直接读取的 data url。"""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("缺少 PyMuPDF，请先安装 pymupdf") from exc

        document = fitz.open(str(file_path))
        data_urls: List[str] = []
        for page in document:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            image_bytes = pixmap.tobytes("png")
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            data_urls.append(f"data:image/png;base64,{encoded}")
        document.close()
        return data_urls

    mime = _IMAGE_MIME.get(suffix)
    if mime is None:
        raise ValueError(f"反思节点暂不支持该文件类型: {suffix}")

    encoded = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    return [f"data:{mime};base64,{encoded}"]
