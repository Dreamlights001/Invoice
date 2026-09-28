"""对比基准方法与 RAG/reflect 各实验方法的评估结果。"""
import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from node.ocr_node import normalize_name


_METHOD_PREFIX = {
    "base": "",
    "rag_only": "rag_only_",
    "reflect_only": "reflect_only_",
    "rag_reflect": "rag_reflect_",
}


def main() -> int:
    """汇总不同方法的 evaluation 结果并导出 Excel。"""
    parser = argparse.ArgumentParser(description="汇总不同方法的实验结果")
    parser.add_argument("--base_output", type=str, default="output/ac", help="基准实验输出目录")
    parser.add_argument("--experiment_output", type=str, default="output/pending", help="新增实验输出目录")
    parser.add_argument("--out_file", type=str, default="output/pending/method_comparison.xlsx", help="输出 Excel 文件")
    args = parser.parse_args()

    source_map = {
        "base": Path(args.base_output),
        "rag_only": Path(args.experiment_output),
        "reflect_only": Path(args.experiment_output),
        "rag_reflect": Path(args.experiment_output),
    }

    frames: Dict[str, pd.DataFrame] = {}
    for method, root in source_map.items():
        frame = load_method_frame(root, method)
        if frame is None or frame.empty:
            print(f"未找到 {method} 的有效结果: {root}")
            return 1
        frames[method] = frame

    merged: Optional[pd.DataFrame] = None
    summary_rows = []
    for method, frame in frames.items():
        current = frame[["doc_id", "score", "correct_fields", "total_fields"]].rename(
            columns={
                "score": f"{method}_score",
                "correct_fields": f"{method}_correct_fields",
                "total_fields": f"{method}_total_fields",
            }
        )
        merged = current if merged is None else merged.merge(current, on="doc_id", how="outer")
        summary_rows.append(
            {
                "method": method,
                "source_root": str(source_map[method]),
                "avg_score": float(frame["score"].mean()),
                "documents": int(len(frame)),
            }
        )

    assert merged is not None
    merged["rag_only_delta"] = merged["rag_only_score"] - merged["base_score"]
    merged["reflect_only_delta"] = merged["reflect_only_score"] - merged["base_score"]
    merged["rag_reflect_delta"] = merged["rag_reflect_score"] - merged["base_score"]
    merged = merged.sort_values(by=["rag_reflect_delta", "doc_id"], ascending=[False, True])

    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        merged.to_excel(writer, index=False, sheet_name="method_comparison")
        pd.DataFrame(summary_rows).to_excel(writer, index=False, sheet_name="summary")

    print(f"方法对比结果已输出: {out_file}")
    return 0


def load_method_frame(root: Path, method: str) -> Optional[pd.DataFrame]:
    """优先读取 comparison.xlsx，不存在时回退扫描 evaluation.json。"""
    comparison_name = "comparison.xlsx" if method == "base" else f"comparison.{method}.xlsx"
    comparison_path = root / comparison_name
    if comparison_path.exists():
        return pd.read_excel(comparison_path, sheet_name="comparison")
    return build_frame_from_evaluations(root, method)


def build_frame_from_evaluations(root: Path, method: str) -> Optional[pd.DataFrame]:
    """从各文档目录中的 evaluation.json 重建对比表。"""
    if not root.exists():
        return None

    rows = []
    prefix = _METHOD_PREFIX[method]
    for subdir in sorted(root.iterdir(), key=lambda p: p.name.casefold()):
        if not subdir.is_dir():
            continue
        name = subdir.name
        if method == "base":
            if any(name.startswith(other_prefix) for key, other_prefix in _METHOD_PREFIX.items() if key != "base"):
                continue
        elif not name.startswith(prefix):
            continue

        evaluation_file = next(subdir.glob("*.evaluation.json"), None)
        if evaluation_file is None:
            continue
        try:
            payload = json.loads(evaluation_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue

        doc_id = normalize_name(name[len(prefix):] if prefix else name)
        score = payload.get("score")
        correct_fields = payload.get("correct_fields")
        total_fields = payload.get("total_fields")
        wrong_fields = payload.get("wrong_fields", [])
        if not isinstance(wrong_fields, list):
            wrong_fields = []

        score_reason = f"正确字段 {correct_fields}/{total_fields}" if score is not None else "未提供 truth，无法评分"
        rows.append(
            {
                "doc_id": doc_id,
                "method": method,
                "truth_found": payload.get("truth") is not None,
                "pred_found": payload.get("predicted") is not None,
                "pred_json": json.dumps(payload.get("predicted"), ensure_ascii=False, indent=2) if payload.get("predicted") is not None else "",
                "score": float(score) if score is not None else None,
                "correct_fields": int(correct_fields) if correct_fields is not None else None,
                "total_fields": int(total_fields) if total_fields is not None else None,
                "wrong_fields": json.dumps(wrong_fields, ensure_ascii=False),
                "score_reason": score_reason,
                "pred_path": str(next(subdir.glob("*.invoice.json"), subdir)),
            }
        )

    if not rows:
        return None
    return pd.DataFrame(rows)


if __name__ == "__main__":
    raise SystemExit(main())
