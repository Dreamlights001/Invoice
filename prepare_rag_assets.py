"""准备 RAG 实验资产：构建 36 张知识库，并为 64 张测试集生成 few-shot。"""
import argparse
import json
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set

from node.ocr_node import normalize_name
from node.rag_store import normalize_rag_doc_id


DEFAULT_SOURCE_DB = Path(r"D:\Documents\Python\invoice_rag_v2\invoice_rag_v2\step3_knowledge\knowledge_v3.sqlite")
DEFAULT_VALIDATION_DIRS = [
    Path(r"D:\Documents\Python\Invoice-hjc-新原始\input\validation"),
    Path(r"D:\Documents\Python\invoice_rag_v2\invoice_rag_v2\input\validation"),
]


def canonical_doc_key(value: str) -> str:
    """将不同来源的文件名统一映射到同一文档键。"""
    normalized = normalize_name(str(value))
    normalized = normalize_rag_doc_id(normalized)
    return re.sub(r"[^0-9a-z]+", "", normalized.casefold())


def resolve_validation_dir(candidate: Path) -> Path:
    """解析原始 100 张发票所在目录。"""
    if candidate.exists():
        return candidate
    for path in DEFAULT_VALIDATION_DIRS:
        if path.exists():
            return path
    raise FileNotFoundError(f"未找到 validation 目录: {candidate}")


def load_test_doc_ids(base_output: Path) -> Set[str]:
    """从基准实验输出中识别 64 个测试文档。"""
    doc_ids: Set[str] = set()
    for path in base_output.iterdir():
        if not path.is_dir():
            continue
        if not any(path.glob("*.metadata.json")):
            continue
        doc_ids.add(canonical_doc_key(path.name))
    return doc_ids


def scan_validation_files(validation_dir: Path) -> Dict[str, Path]:
    """扫描原始 100 张发票文件，并建立文档键到路径的映射。"""
    mapping: Dict[str, Path] = {}
    for path in validation_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".pdf", ".png", ".jpg", ".jpeg", ".webp"}:
            continue
        key = canonical_doc_key(path.name)
        mapping.setdefault(key, path)
    return mapping


def load_source_docs(source_db: Path) -> List[str]:
    """读取原始知识库中的全部文档编号。"""
    conn = sqlite3.connect(str(source_db))
    rows = conn.execute("SELECT doc_id FROM docs ORDER BY doc_id").fetchall()
    conn.close()
    return [str(row[0]) for row in rows]


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    """加载 sqlite-vec，以便访问 vec_items 虚表。"""
    try:
        import sqlite_vec
    except ImportError as exc:
        raise RuntimeError("缺少 sqlite-vec，请先执行 pip install sqlite-vec") from exc
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def build_filtered_knowledge_db(source_db: Path, target_db: Path, test_doc_keys: Set[str]) -> Dict[str, object]:
    """从 100 张知识库中过滤测试集，得到只含 36 张文档的新知识库。"""
    target_db.parent.mkdir(parents=True, exist_ok=True)
    if target_db.exists():
        target_db.unlink()
    shutil.copy2(source_db, target_db)

    conn = sqlite3.connect(str(target_db))
    conn.row_factory = sqlite3.Row
    _load_sqlite_vec(conn)

    docs_rows = conn.execute("SELECT doc_id FROM docs").fetchall()
    remove_doc_ids = [str(row["doc_id"]) for row in docs_rows if canonical_doc_key(str(row["doc_id"])) in test_doc_keys]
    keep_doc_ids = [str(row["doc_id"]) for row in docs_rows if canonical_doc_key(str(row["doc_id"])) not in test_doc_keys]

    item_rows = conn.execute("SELECT id, doc_id FROM items").fetchall()
    remove_item_ids = [int(row["id"]) for row in item_rows if str(row["doc_id"]) in remove_doc_ids]

    with conn:
        for item_id in remove_item_ids:
            conn.execute("DELETE FROM vec_items WHERE rowid = ?", (item_id,))
        if remove_item_ids:
            conn.executemany("DELETE FROM items WHERE id = ?", [(item_id,) for item_id in remove_item_ids])
        if remove_doc_ids:
            conn.executemany("DELETE FROM blocks WHERE doc_id = ?", [(doc_id,) for doc_id in remove_doc_ids])
            conn.executemany("DELETE FROM docs WHERE doc_id = ?", [(doc_id,) for doc_id in remove_doc_ids])

    remaining_docs = conn.execute("SELECT count(*) FROM docs").fetchone()[0]
    remaining_blocks = conn.execute("SELECT count(*) FROM blocks").fetchone()[0]
    remaining_items = conn.execute("SELECT count(*) FROM items").fetchone()[0]
    conn.close()

    return {
        "removed_docs": len(remove_doc_ids),
        "kept_docs": len(keep_doc_ids),
        "remaining_docs": int(remaining_docs),
        "remaining_blocks": int(remaining_blocks),
        "remaining_items": int(remaining_items),
        "removed_doc_ids": sorted(remove_doc_ids),
        "kept_doc_ids": sorted(keep_doc_ids),
    }


def write_split_manifest(
    manifest_path: Path,
    *,
    test_doc_keys: Set[str],
    validation_map: Dict[str, Path],
    source_docs: List[str],
    db_stats: Dict[str, object],
) -> None:
    """输出知识库划分清单，便于后续论文和实验核查。"""
    test_doc_keys_sorted = sorted(test_doc_keys)
    kept_doc_ids = [str(item) for item in db_stats.get("kept_doc_ids", [])]
    payload = {
        "test_doc_count": len(test_doc_keys_sorted),
        "knowledge_doc_count": len(kept_doc_ids),
        "source_doc_count": len(source_docs),
        "test_docs": [
            {
                "normalized_doc_id": key,
                "validation_file": str(validation_map.get(key, "")),
            }
            for key in test_doc_keys_sorted
        ],
        "knowledge_docs": [
            {
                "normalized_doc_id": canonical_doc_key(doc_id),
                "knowledge_doc_id": doc_id,
                "validation_file": str(validation_map.get(canonical_doc_key(doc_id), "")),
            }
            for doc_id in kept_doc_ids
        ],
        "db_stats": db_stats,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_precompute(config: Path, intervar: Path, output_dir: Path) -> int:
    """调用 precompute_fewshot 为 64 个测试样本生成 few-shot。"""
    if output_dir.exists():
        for path in output_dir.glob("*.fewshot.json"):
            path.unlink()
        manifest = output_dir / "manifest.fewshot.json"
        if manifest.exists():
            manifest.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "precompute_fewshot.py",
        "--input",
        str(intervar),
        "--config",
        str(config),
        "--output",
        str(output_dir),
        "--enable_rerank",
        "--workers",
        "1",
    ]
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="用 36 张知识库样本为 64 张测试样本准备 RAG 资产")
    parser.add_argument("--base_output", type=str, default="output/ac", help="64 张测试样本的基准实验输出目录")
    parser.add_argument("--source_db", type=str, default=str(DEFAULT_SOURCE_DB), help="原始 100 张知识库")
    parser.add_argument("--validation", type=str, default="", help="100 张原始发票所在目录")
    parser.add_argument("--target_db", type=str, default="knowledge/knowledge.sqlite", help="输出的 36 张知识库路径")
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径")
    parser.add_argument("--intervar", type=str, default="interVar", help="64 张测试样本 OCR 文本目录")
    parser.add_argument("--fewshot_output", type=str, default="fewshot_precomputed", help="few-shot 输出目录")
    parser.add_argument("--skip_precompute", action="store_true", help="仅构建知识库，不生成 few-shot")
    args = parser.parse_args()

    base_output = Path(args.base_output)
    source_db = Path(args.source_db)
    validation_dir = resolve_validation_dir(Path(args.validation) if args.validation else Path("."))
    target_db = Path(args.target_db)
    config_path = Path(args.config)
    intervar = Path(args.intervar)
    fewshot_output = Path(args.fewshot_output)

    if not base_output.exists():
        raise SystemExit(f"基准实验输出目录不存在: {base_output}")
    if not source_db.exists():
        raise SystemExit(f"原始知识库不存在: {source_db}")
    if not validation_dir.exists():
        raise SystemExit(f"validation 目录不存在: {validation_dir}")

    test_doc_keys = load_test_doc_ids(base_output)
    if len(test_doc_keys) != 64:
        raise SystemExit(f"从 {base_output} 中识别到的测试样本不是 64 张，而是 {len(test_doc_keys)} 张")

    validation_map = scan_validation_files(validation_dir)
    source_docs = load_source_docs(source_db)
    db_stats = build_filtered_knowledge_db(source_db, target_db, test_doc_keys)

    manifest_path = target_db.parent / "knowledge_split_manifest.json"
    write_split_manifest(
        manifest_path,
        test_doc_keys=test_doc_keys,
        validation_map=validation_map,
        source_docs=source_docs,
        db_stats=db_stats,
    )

    print(f"[done] 新知识库已生成: {target_db}")
    print(f"[info] 划分清单已输出: {manifest_path}")
    print(f"[info] 知识库文档数: {db_stats['remaining_docs']}")

    if args.skip_precompute:
        return 0

    if not intervar.exists():
        raise SystemExit(f"OCR 文本目录不存在: {intervar}")
    return run_precompute(config_path, intervar, fewshot_output)


if __name__ == "__main__":
    raise SystemExit(main())
