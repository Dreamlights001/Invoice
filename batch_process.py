"""本文件是批处理入口，负责执行发票流程并汇总结果。"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from node.merge_node import process_invoice
from node.ocr_node import load_config, normalize_doc_id, normalize_name


_SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
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


def scan_input_files(input_root: Path) -> List[Path]:
    """扫描输入目录下支持的文件，并去除大小写重复。"""
    found: Dict[str, Path] = {}
    for path in input_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            continue
        found[str(path).lower()] = path
    return sorted(found.values(), key=lambda item: str(item).lower())


def find_output_subdir(input_file: Path, output: Path, method: str) -> Path:
    """按方法生成输出子目录。"""
    return output / f"{_METHOD_PREFIX[method]}{input_file.stem}"


def find_existing_output_subdir(doc_id: str, output: Path, method: str) -> Optional[Path]:
    """在输出目录中查找与文档编号对应的已有子目录。"""
    if not output.exists():
        return None
    prefix = _METHOD_PREFIX[method]
    for path in output.iterdir():
        if not path.is_dir():
            continue
        if prefix and not path.name.startswith(prefix):
            continue
        if not prefix and any(path.name.startswith(item) for key, item in _METHOD_PREFIX.items() if key != "base"):
            continue
        if normalize_name(path.name) == normalize_name(prefix + doc_id):
            return path
    return None


def is_already_processed(input_file: Path, output: Path, method: str) -> bool:
    """判断某个文档是否已产出关键结果。"""
    doc_id = normalize_doc_id(input_file)
    output_subdir = find_output_subdir(input_file, output, method)
    if not output_subdir.exists():
        output_subdir = find_existing_output_subdir(doc_id, output, method) or output_subdir
        if not output_subdir.exists():
            return False

    invoice_file = output_subdir / f"{doc_id}.invoice.json"
    metadata_file = output_subdir / f"{doc_id}.metadata.json"
    if not invoice_file.exists() or not metadata_file.exists():
        return False

    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except Exception:
        return invoice_file.exists()
    return metadata.get("errors", []) == [] or invoice_file.exists()


def process_one(
    input_file: Path,
    config,
    truth_map: Dict[str, Dict],
    output: Path,
    intervar: Optional[Path],
    method: str,
) -> Dict:
    """处理单个文档并返回摘要结果。"""
    doc_id = normalize_doc_id(input_file)
    truth = truth_map.get(doc_id)
    output_subdir = find_output_subdir(input_file, output, method)

    try:
        state = process_invoice(
            input=input_file,
            config=config,
            truth=truth,
            output_dir=output_subdir,
            intervar=intervar,
            method=method,
        )
        return {
            "file": str(input_file),
            "doc_id": doc_id,
            "method": method,
            "output": str(output_subdir),
            "success": len(state["errors"]) == 0,
            "errors": state["errors"],
            "has_invoice": (state.get("final_invoice") or state.get("merged_invoice")) is not None,
            "score": state.get("evaluation_result", {}).get("score") if state.get("evaluation_result") else None,
            "ocr_source": state.get("metadata", {}).get("ocr_source"),
            "final_source": state.get("metadata", {}).get("final_invoice_source"),
        }
    except Exception as exc:
        return {
            "file": str(input_file),
            "doc_id": doc_id,
            "method": method,
            "output": str(output_subdir),
            "success": False,
            "errors": [str(exc)],
            "has_invoice": False,
            "score": None,
            "ocr_source": None,
            "final_source": None,
        }


def main() -> None:
    """执行目录批处理。"""
    parser = argparse.ArgumentParser(description="发票批量处理")
    parser.add_argument("--input", type=str, required=True, help="待处理目录")
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径")
    parser.add_argument("--truth", type=str, default="input/truth", help="truth 目录，按文件名匹配")
    parser.add_argument("--intervar", type=str, default=None, help="OCR 缓存目录；不传则不读取也不保存缓存")
    parser.add_argument("--output", type=str, default="output/pending", help="输出根目录")
    parser.add_argument("--method", type=str, default="base", choices=sorted(_METHODS), help="处理方法")
    parser.add_argument("--workers", type=int, default=4, help="并行线程数")
    parser.add_argument("--skip_existing", action="store_true", help="跳过已处理文件")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    truth_map = load_truth_map(Path(args.truth))
    intervar = Path(args.intervar) if args.intervar else None
    output = Path(args.output)
    input_files = scan_input_files(Path(args.input))

    if args.skip_existing:
        input_files = [item for item in input_files if not is_already_processed(item, output, args.method)]

    if not input_files:
        print("没有需要处理的文件")
        return

    output.mkdir(parents=True, exist_ok=True)
    print(f"待处理文件数: {len(input_files)}")
    print(f"处理方法: {args.method}")
    print(f"输出根目录: {output}")
    print(f"OCR 缓存: {'启用' if intervar else '未启用'}")
    if intervar:
        print(f"OCR 缓存目录: {intervar}")

    results: List[Dict] = []
    if args.workers <= 1:
        for index, input_file in enumerate(input_files, start=1):
            print(f"[{index}/{len(input_files)}] 正在处理 {input_file.name}")
            results.append(process_one(input_file, config, truth_map, output, intervar, args.method))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_one, input_file, config, truth_map, output, intervar, args.method): input_file
                for input_file in input_files
            }
            for index, future in enumerate(as_completed(futures), start=1):
                input_file = futures[future]
                print(f"[{index}/{len(input_files)}] 已完成 {input_file.name}")
                results.append(future.result())

    summary = {
        "method": args.method,
        "total": len(results),
        "successful": sum(1 for item in results if item["success"]),
        "failed": sum(1 for item in results if not item["success"]),
        "with_invoice": sum(1 for item in results if item["has_invoice"]),
        "ocr_cache_hits": sum(1 for item in results if item.get("ocr_source") == "interVar"),
        "ocr_api_calls": sum(1 for item in results if item.get("ocr_source") == "api"),
        "reflect_outputs": sum(1 for item in results if item.get("final_source") == "reflect"),
        "results": results,
    }
    summary_name = "summary.json" if args.method == "base" else f"summary.{args.method}.json"
    (output / summary_name).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 汇总 ===")
    print(f"总数: {summary['total']}")
    print(f"成功: {summary['successful']}")
    print(f"失败: {summary['failed']}")
    print(f"产出发票结果: {summary['with_invoice']}")
    print(f"OCR 缓存命中: {summary['ocr_cache_hits']}")
    print(f"OCR 接口调用: {summary['ocr_api_calls']}")
    if args.method in {"reflect_only", "rag_reflect"}:
        print(f"反思结果生效: {summary['reflect_outputs']}")

    scores = [item["score"] for item in results if item["score"] is not None]
    if scores:
        print(f"平均得分: {sum(scores) / len(scores):.2%}")


if __name__ == "__main__":
    main()
