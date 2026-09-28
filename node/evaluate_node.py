"""评估节点：对预测结果与 truth 做字段级比较。"""
import re
from typing import Any, Dict, List, Optional, Tuple

from .ocr_node import InvoiceState


_COMPARE_TOLERANCE = 1e-6


def compare_fields(predicted: Dict[str, Any], truth: Dict[str, Any]) -> Dict[str, Any]:
    """对预测值和 truth 执行逐叶子字段比较。"""
    predicted_leaf_map = flatten_for_compare(predicted)
    truth_leaf_map = flatten_for_compare(truth)

    matched = 0
    mismatches: List[Dict[str, Any]] = []
    for field_path, expected_value in truth_leaf_map.items():
        actual_value = predicted_leaf_map.get(field_path)
        if values_match(field_path, actual_value, expected_value):
            matched += 1
            continue
        mismatches.append(
            {
                "field": field_path,
                "predicted": actual_value,
                "truth": expected_value,
            }
        )

    total = len(truth_leaf_map)
    return {
        "score": (matched / total) if total else 0.0,
        "correct_fields": matched,
        "total_fields": total,
        "wrong_fields": mismatches,
    }


def evaluate_node(state: InvoiceState) -> InvoiceState:
    """在合并完成后，将预测结果与 truth 做对齐比较。"""
    final_payload = state.get("final_invoice") or state.get("reflected_invoice") or state.get("merged_invoice")
    if not final_payload:
        state["errors"].append("没有可用的发票数据进行评估")
        return state

    invoice_payload = _pick_invoice_payload(final_payload)
    truth_payload = state.get("truth")
    if not truth_payload:
        state["evaluation_result"] = {
            "score": None,
            "correct_fields": None,
            "total_fields": None,
            "wrong_fields": [],
            "predicted": invoice_payload,
            "truth": None,
            "note": "没有提供truth数据，无法进行评估",
        }
        return state

    try:
        result = compare_fields(invoice_payload, truth_payload)
        result["predicted"] = invoice_payload
        result["truth"] = truth_payload
        state["evaluation_result"] = result
    except Exception as exc:
        state["errors"].append(f"评估失败: {exc}")
        state["evaluation_result"] = {
            "score": None,
            "error": str(exc),
            "predicted": invoice_payload,
            "truth": truth_payload,
        }
    return state


def flatten_for_compare(source: Any) -> Dict[str, Any]:
    """将嵌套对象压平成路径到值的映射。"""
    flattened: Dict[str, Any] = {}
    pending: List[Tuple[str, Any]] = [("", source)]

    while pending:
        prefix, current = pending.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                pending.append((next_prefix, value))
            continue

        if isinstance(current, list):
            for index, value in enumerate(current):
                next_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
                pending.append((next_prefix, value))
            continue

        if prefix:
            flattened[prefix] = current

    return flattened


def values_match(field_path: str, actual: Any, expected: Any) -> bool:
    """判断两个值在字段规则下是否可视为一致。"""
    if _is_emptyish(actual) and _is_emptyish(expected):
        return True

    if _requires_strict_match(field_path):
        return _normalize_strict_text(actual) == _normalize_strict_text(expected)

    actual_number = _parse_number_like(actual)
    expected_number = _parse_number_like(expected)
    if actual_number is not None and expected_number is not None:
        return abs(actual_number - expected_number) <= _COMPARE_TOLERANCE

    strict_actual = _normalize_text(actual)
    strict_expected = _normalize_text(expected)
    if strict_actual == strict_expected:
        return True

    return _normalize_loose_text(actual) == _normalize_loose_text(expected)


def _requires_strict_match(field_path: str) -> bool:
    """对发票号、订单号字段保持严格匹配。"""
    normalized = field_path.replace("[", ".").replace("]", "")
    return (
        normalized == "invoiceNumber"
        or normalized == "orderNo"
        or normalized.endswith(".invoiceNumber")
        or normalized.endswith(".orderNo")
    )


def _pick_invoice_payload(merged_payload: Any) -> Any:
    """兼容顶层直接为发票对象或包裹在 INVOICE 字段中的情况。"""
    if isinstance(merged_payload, dict) and "INVOICE" in merged_payload:
        return merged_payload["INVOICE"]
    return merged_payload


def _is_emptyish(value: Any) -> bool:
    """判断值是否为空。"""
    if value is None:
        return True
    if isinstance(value, str):
        return len(value.strip()) == 0
    return False


def _normalize_strict_text(value: Any) -> str:
    """严格匹配时仅去除首尾空白。"""
    if value is None:
        return ""
    return str(value).strip()


def _normalize_text(value: Any) -> str:
    """做基础文本归一化。"""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().casefold())


def _normalize_loose_text(value: Any) -> str:
    """做宽松文本归一化，用于忽略大小写、尾部标点和多余符号。"""
    text = _normalize_text(value)
    text = re.sub(r"[\.,;:!?'\"()\[\]{}]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_number_like(value: Any) -> Optional[float]:
    """将常见金额、数量、比率文本解析为数值。"""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    compact = re.sub(r"\s+", "", text)
    ratio_match = re.fullmatch(r"([-+]?[0-9][0-9,\.]*)/100", compact)
    if ratio_match:
        base_number = _parse_plain_number(ratio_match.group(1))
        return None if base_number is None else base_number / 100.0

    percent_match = re.fullmatch(r"([-+]?[0-9][0-9,\.]*)%", compact)
    if percent_match:
        base_number = _parse_plain_number(percent_match.group(1))
        return None if base_number is None else base_number / 100.0

    return _parse_plain_number(text)


def _parse_plain_number(text: str) -> Optional[float]:
    """解析普通数字字符串，兼容千分位与小数点格式。"""
    cleaned = re.sub(r"[^0-9,\.\-+]", "", str(text))
    if not cleaned or cleaned in {"-", "+", ".", ","}:
        return None

    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        parts = cleaned.split(",")
        if len(parts) == 2 and len(parts[1]) <= 4:
            cleaned = f"{parts[0]}.{parts[1]}"
        else:
            cleaned = "".join(parts)
    elif cleaned.count(".") > 1:
        head, tail = cleaned.rsplit(".", 1)
        cleaned = head.replace(".", "") + "." + tail

    try:
        return float(cleaned)
    except ValueError:
        return None
