"""本文件用于对比预测结果与 truth 结果，并导出 Excel 与 CSV。"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from node.evaluate_node import flatten_for_compare, values_match
from node.ocr_node import normalize_name


_METHODS = {"base", "rag_only", "reflect_only", "rag_reflect"}
_METHOD_PREFIX = {
    "base": "",
    "rag_only": "rag_only_",
    "reflect_only": "reflect_only_",
    "rag_reflect": "rag_reflect_",
}


def load_truth_map(truth: Path) -> Dict[str, Dict[str, Any]]:
    """从 truth 目录中读取所有 truth JSON。"""
    if not truth.exists():
        return {}

    truth_map: Dict[str, Dict[str, Any]] = {}
    for path in truth.rglob("*.json"):
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(parsed, dict):
            truth_map[normalize_name(path.name)] = parsed
    return truth_map


def find_prediction_path(output: Path, doc_id: str, method: str) -> Path:
    """在输出目录中查找与 truth 对应的预测文件。"""
    direct_dir_name = f"{_METHOD_PREFIX[method]}{doc_id}"
    direct_path = output / direct_dir_name / f"{doc_id}.invoice.json"
    if direct_path.exists():
        return direct_path

    if output.exists():
        for path in output.iterdir():
            if not path.is_dir():
                continue
            is_target_method_dir = path.name.startswith(_METHOD_PREFIX[method]) if _METHOD_PREFIX[method] else not any(
                path.name.startswith(prefix) for name, prefix in _METHOD_PREFIX.items() if name != "base"
            )
            if not is_target_method_dir:
                continue
            if normalize_name(path.name) != normalize_name(direct_dir_name):
                continue
            candidate = path / f"{doc_id}.invoice.json"
            if candidate.exists():
                return candidate
    return direct_path


def load_predicted_invoice(pred_path: Path) -> Optional[Dict[str, Any]]:
    """读取预测发票 JSON。"""
    if not pred_path.exists():
        return None
    try:
        obj = json.loads(pred_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(obj, dict) and "INVOICE" in obj:
        return obj["INVOICE"]
    if isinstance(obj, dict):
        return obj
    return None


def compare_invoice(truth: Dict[str, Any], predicted: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """对比 truth 和预测结果。"""
    truth_flat = flatten_for_compare(truth)
    predicted_flat = flatten_for_compare(predicted or {})

    wrong_fields: List[Dict[str, Any]] = []
    correct = 0
    total = 0
    for path, truth_value in truth_flat.items():
        total += 1
        predicted_value = predicted_flat.get(path)
        if values_match(path, predicted_value, truth_value):
            correct += 1
            continue
        wrong_fields.append({"path": path, "truth": truth_value, "predicted": predicted_value})

    return {
        "correct": correct,
        "total": total,
        "score": (correct / total) if total else 0.0,
        "wrong_fields": wrong_fields,
    }


def main() -> int:
    """生成 truth 对比报表。"""
    parser = argparse.ArgumentParser(description="生成 truth 对比报表")
    parser.add_argument("--truth", type=str, default="input/truth", help="truth 目录")
    parser.add_argument("--output", type=str, default="output/pending", help="预测结果根目录")
    parser.add_argument("--method", type=str, default="base", choices=sorted(_METHODS), help="处理方法")
    parser.add_argument("--out_file", type=str, default=None, help="输出 Excel 路径")
    args = parser.parse_args()

    truth = Path(args.truth)
    output = Path(args.output)
    out_file = Path(args.out_file) if args.out_file else output / ("comparison.xlsx" if args.method == "base" else f"comparison.{args.method}.xlsx")

    truth_map = load_truth_map(truth)
    if not truth_map:
        print(f"错误: 在 {truth} 中未找到 truth JSON")
        return 1

    rows_out = []
    diffs_rows = []
    for row_index, (doc_id, truth_payload) in enumerate(sorted(truth_map.items()), start=1):
        pred_path = find_prediction_path(output, doc_id, args.method)
        predicted_invoice = load_predicted_invoice(pred_path)
        compare_result = compare_invoice(truth_payload, predicted_invoice)

        score_reason = f"正确字段 {compare_result['correct']}/{compare_result['total']}"
        if compare_result["wrong_fields"]:
            wrong_paths = [item["path"] for item in compare_result["wrong_fields"][:12]]
            score_reason += f"；错误字段: {', '.join(wrong_paths)}"

        rows_out.append(
            {
                "doc_id": doc_id,
                "method": args.method,
                "truth_found": True,
                "pred_found": predicted_invoice is not None,
                "pred_json": json.dumps(predicted_invoice, ensure_ascii=False, indent=2) if predicted_invoice else "",
                "score": float(round(compare_result["score"], 6)),
                "correct_fields": int(compare_result["correct"]),
                "total_fields": int(compare_result["total"]),
                "wrong_fields": json.dumps(compare_result["wrong_fields"], ensure_ascii=False),
                "score_reason": score_reason,
                "pred_path": str(pred_path),
            }
        )

        for wrong in compare_result["wrong_fields"]:
            diffs_rows.append(
                {
                    "row_idx": row_index,
                    "doc_id": doc_id,
                    "method": args.method,
                    "path": wrong.get("path"),
                    "truth": wrong.get("truth"),
                    "predicted": wrong.get("predicted"),
                }
            )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(rows_out)
    with pd.ExcelWriter(str(out_file), engine="openpyxl") as writer:
        result_df.to_excel(writer, index=False, sheet_name="comparison")
        if diffs_rows:
            pd.DataFrame(diffs_rows).to_excel(writer, index=False, sheet_name="field_diffs")

    if diffs_rows:
        csv_name = "field_diffs.csv" if args.method == "base" else f"field_diffs.{args.method}.csv"
        pd.DataFrame(diffs_rows).to_csv(out_file.parent / csv_name, index=False, encoding="utf-8-sig")

    print("\n=== 对比完成 ===")
    print(f"处理方法: {args.method}")
    print(f"输出文件: {out_file}")
    print(f"文档总数: {len(rows_out)}")
    print(f"找到预测结果: {sum(1 for row in rows_out if row['pred_found'])}")
    found_count = max(1, sum(1 for row in rows_out if row['pred_found']))
    print(f"平均得分: {sum(row['score'] for row in rows_out if row['pred_found']) / found_count:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
