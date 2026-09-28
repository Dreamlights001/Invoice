"""本文件是单文件处理入口，负责读取配置、加载 truth 并执行发票处理流程。"""
import argparse
import json
from pathlib import Path
from typing import Dict

from node.merge_node import process_invoice
from node.ocr_node import load_config, normalize_name


_METHODS = {"base", "rag_only", "reflect_only", "rag_reflect"}
_METHOD_PREFIX = {
    "base": "",
    "rag_only": "rag_only_",
    "reflect_only": "reflect_only_",
    "rag_reflect": "rag_reflect_",
}


def load_truth_map(truth: Path) -> Dict[str, Dict]:
    """从 truth 目录中按文件名加载标注结果。"""
    if not truth.exists():
        return {}

    truth_map: Dict[str, Dict] = {}
    for path in truth.rglob("*.json"):
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(parsed, dict):
            truth_map[normalize_name(path.name)] = parsed
    return truth_map


def build_output_subdir(output: Path, input_file: Path, method: str) -> Path:
    """为不同方法生成互不覆盖的输出目录。"""
    return output / f"{_METHOD_PREFIX[method]}{input_file.stem}"


def main() -> None:
    """解析参数并执行单文件抽取。"""
    parser = argparse.ArgumentParser(description="发票处理流程")
    parser.add_argument("--input", type=str, required=True, help="输入图片或 PDF 路径")
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径")
    parser.add_argument("--truth", type=str, default="input/truth", help="truth 目录，按文件名匹配")
    parser.add_argument("--intervar", type=str, default=None, help="OCR 缓存目录；不传则不读取也不保存缓存")
    parser.add_argument("--output", type=str, default="output/pending", help="输出根目录")
    parser.add_argument("--method", type=str, default="base", choices=sorted(_METHODS), help="处理方法")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    input_file = Path(args.input)
    truth_map = load_truth_map(Path(args.truth))
    truth = truth_map.get(normalize_name(input_file.name))
    output_subdir = build_output_subdir(Path(args.output), input_file, args.method)
    intervar = Path(args.intervar) if args.intervar else None

    state = process_invoice(
        input=input_file,
        config=config,
        truth=truth,
        output_dir=output_subdir,
        intervar=intervar,
        method=args.method,
    )

    print("\n=== 处理完成 ===")
    print(f"文档编号: {state['doc_id']}")
    print(f"处理方法: {args.method}")
    print(f"输出目录: {output_subdir}")
    print(f"错误数量: {len(state['errors'])}")
    for error in state["errors"]:
        print(f"- {error}")

    print(f"OCR 来源: {state['metadata'].get('ocr_source', 'unknown')}")
    cache_path = state["metadata"].get("ocr_cache_path", "")
    if cache_path:
        print(f"OCR 缓存文件: {cache_path}")
    else:
        print("OCR 缓存文件: 未启用或未生成")

    final_invoice = state.get("final_invoice") or state.get("merged_invoice")
    if final_invoice:
        print("\n=== 最终发票结果 ===")
        print(json.dumps(final_invoice, ensure_ascii=False, indent=2))

    result = state.get("evaluation_result")
    if result and result.get("score") is not None:
        print("\n=== 对比结果 ===")
        print(f"得分: {result['score']:.2%}")
        print(f"正确字段: {result['correct_fields']}/{result['total_fields']}")


if __name__ == "__main__":
    main()
