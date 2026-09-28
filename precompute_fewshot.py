"""离线预计算 RAG few-shot 结果，供分块节点优先复用。"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from node.ocr_node import load_config, normalize_name
from node.rag_store import RAGStore, normalize_rag_doc_id


def _doc_id_from_txt_filename(path: Path) -> str:
    """从 OCR 文本文件名中恢复文档编号。"""
    return normalize_name(path.name)


def rerank_documents(
    query: str,
    documents: List[Dict[str, Any]],
    *,
    url: str,
    api_key: str,
    model: str,
    top_k: int,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> List[Dict[str, Any]]:
    """调用 rerank 接口对候选文档进行重排。"""
    if not documents:
        return []
    if not url or not api_key or not model:
        return documents[:top_k]

    doc_texts: List[str] = []
    for doc in documents:
        doc_ocr = str(doc.get("doc_ocr") or "").strip()
        if not doc_ocr:
            blocks_json = doc.get("blocks_json") or {}
            doc_ocr = " ".join(
                f"{item.get('reason', '')} {item.get('text', '')} {item.get('parsed', '')}"
                for item in blocks_json.values()
            )
        doc_texts.append(doc_ocr[:2000])

    payload = {
        "model": model,
        "query": query[:2000],
        "documents": doc_texts,
        "top_n": min(int(top_k), len(documents)),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code == 429 and attempt < max_retries - 1:
                import time
                time.sleep(retry_delay * (2 ** attempt))
                continue
            response.raise_for_status()
            data = response.json()
            reranked: List[Dict[str, Any]] = []
            for rank, item in enumerate(data.get("results", [])[:top_k], start=1):
                index = item.get("index")
                if index is None or not (0 <= index < len(documents)):
                    continue
                doc = documents[index].copy()
                doc["rank"] = rank
                doc["rerank_score"] = item.get("relevance_score", 0.0)
                reranked.append(doc)
            return reranked or documents[:top_k]
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries - 1:
                break
            import time
            time.sleep(retry_delay * (2 ** attempt))

    if last_error is not None:
        print(f"[warning] rerank failed, fallback to embedding candidates: {last_error}")
    return documents[:top_k]


def precompute_one(
    text_file: Path,
    rag_store: RAGStore,
    top_k: int,
    top_k_candidates: int,
    enable_rerank: bool,
    rerank_url: str,
    rerank_api_key: str,
    rerank_model: str,
) -> Dict[str, Any]:
    """为单个 OCR 文本生成 few-shot 缓存。"""
    ocr_text = text_file.read_text(encoding="utf-8", errors="replace")
    doc_id = _doc_id_from_txt_filename(text_file)
    exclude_doc_id = normalize_rag_doc_id(doc_id)

    candidates = rag_store.retrieve_doc_examples(
        query_text=ocr_text,
        k=int(top_k_candidates),
        exclude_doc_id=exclude_doc_id,
    )

    if enable_rerank:
        rerank_results = rerank_documents(
            query=ocr_text,
            documents=candidates,
            url=rerank_url,
            api_key=rerank_api_key,
            model=rerank_model,
            top_k=int(top_k),
        )
        retrieve_payload: Any = {
            "rerank": rerank_results,
            "embedding_candidates": candidates,
        }
    else:
        rerank_results = candidates[:int(top_k)]
        retrieve_payload = rerank_results

    return {
        "id": text_file.name,
        "doc_id": doc_id,
        "ocr_text": ocr_text,
        "query_used_chars": min(len(ocr_text), 800),
        "retrieve": retrieve_payload,
        "rerank_enabled": bool(enable_rerank),
        "top_k": int(top_k),
        "top_k_candidates": int(top_k_candidates),
        "retrieve_count": len(rerank_results),
        "candidates_count": len(candidates),
    }


def scan_text_files(input_dir: Path) -> List[Path]:
    """扫描 OCR 文本目录，返回全部 txt 文件。"""
    found: Dict[str, Path] = {}
    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() != ".txt":
            continue
        found[str(path).lower()] = path
    return sorted(found.values(), key=lambda item: str(item).lower())


def main() -> int:
    parser = argparse.ArgumentParser(description="预计算发票 few-shot 结果")
    parser.add_argument("--input", type=str, default="interVar", help="OCR 文本目录")
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径")
    parser.add_argument("--output", type=str, default="fewshot_precomputed", help="few-shot 输出目录")
    parser.add_argument("--top_k", type=int, default=5, help="最终保留的 few-shot 数量")
    parser.add_argument("--top_k_candidates", type=int, default=20, help="rerank 前候选数量")
    parser.add_argument("--enable_rerank", action="store_true", help="启用 rerank 重排")
    parser.add_argument("--workers", type=int, default=4, help="并发线程数")
    parser.add_argument("--skip_existing", action="store_true", help="跳过已存在结果")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    if config.rag_db_path is None or not config.rag_db_path.exists():
        raise SystemExit("未找到可用知识库，请先准备 knowledge/knowledge.sqlite")

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    if not input_dir.exists():
        raise SystemExit(f"输入目录不存在: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    text_files = scan_text_files(input_dir)
    if not text_files:
        raise SystemExit(f"在 {input_dir} 下未发现 OCR 文本")

    manifest: Dict[str, Any] = {
        "createdAt": datetime.now().isoformat(timespec="seconds"),
        "input": str(input_dir),
        "output": str(output_dir),
        "db": str(config.rag_db_path),
        "topK": int(args.top_k),
        "topKCandidates": int(args.top_k_candidates),
        "rerankEnabled": bool(args.enable_rerank),
        "outputs": [],
        "errors": [],
    }

    to_process: List[Path] = []
    for path in text_files:
        doc_id = _doc_id_from_txt_filename(path)
        out_path = output_dir / f"{doc_id}.fewshot.json"
        if args.skip_existing and out_path.exists():
            continue
        to_process.append(path)

    print(f"[info] text_files={len(text_files)} to_process={len(to_process)}")
    rag_store = RAGStore(config.rag_db_path, config.embedding)

    futures = {}
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        for path in to_process:
            future = executor.submit(
                precompute_one,
                path,
                rag_store,
                int(args.top_k),
                int(args.top_k_candidates),
                bool(args.enable_rerank),
                config.rerank.url,
                config.rerank.api_key,
                config.rerank.model,
            )
            futures[future] = path

        done = 0
        for future in as_completed(futures):
            path = futures[future]
            doc_id = _doc_id_from_txt_filename(path)
            out_path = output_dir / f"{doc_id}.fewshot.json"
            done += 1
            try:
                payload = future.result()
                out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                manifest["outputs"].append(
                    {
                        "doc_id": doc_id,
                        "path": str(out_path),
                        "retrieveCount": payload.get("retrieve_count", 0),
                        "candidatesCount": payload.get("candidates_count", 0),
                    }
                )
            except Exception as exc:
                manifest["errors"].append(
                    {
                        "doc_id": doc_id,
                        "file": path.name,
                        "error": str(exc),
                    }
                )
                print(f"[error] {doc_id}: {exc}")

            if done % 10 == 0 or done == len(to_process):
                print(f"[info] processed {done}/{len(to_process)}")

    (output_dir / "manifest.fewshot.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rag_store.close()
    print(f"[done] outputs={len(manifest['outputs'])} errors={len(manifest['errors'])} out_dir={output_dir}")
    return 0 if not manifest["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
