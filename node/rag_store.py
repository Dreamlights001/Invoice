"""RAG 存储层：提供整单级和块级的向量检索能力。"""
import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .ocr_node import EmbeddingConfig


class RAGStore:
    """基于 SQLite-vec 的轻量检索封装。"""

    def __init__(self, db_path: Path, embedding: EmbeddingConfig):
        if not db_path.exists():
            raise FileNotFoundError(f"RAG 知识库不存在: {db_path}")
        self.db_path = db_path
        self.embedding = embedding

    def retrieve_doc_examples(
        self,
        query_text: str,
        k: int = 3,
        exclude_doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """检索整单级 few-shot 示例。"""
        query_vector = self._embed_text(f"TYPE=DOC_JSON\n{(query_text or '').strip()}")
        conn = self._get_connection()
        limit = max(int(k), 1) * 4
        sql = """
        WITH candidates AS (
          SELECT id
          FROM items
          WHERE chunk_type = 'doc_json'
            AND (:exclude_doc_id = '' OR doc_id != :exclude_doc_id)
        ),
        knn AS (
          SELECT rowid, distance
          FROM vec_items
          WHERE embedding MATCH :query
            AND rowid IN (SELECT id FROM candidates)
          ORDER BY distance
          LIMIT :limit
        )
        SELECT i.doc_id, knn.distance
        FROM knn
        JOIN items i ON i.id = knn.rowid
        ORDER BY knn.distance
        """
        rows = conn.execute(
            sql,
            {
                "query": json.dumps(query_vector),
                "limit": limit,
                "exclude_doc_id": exclude_doc_id or "",
            },
        ).fetchall()

        results: List[Dict[str, Any]] = []
        excluded_key = normalize_rag_doc_id(exclude_doc_id)
        for row in rows:
            doc_id = str(row[0])
            if excluded_key and normalize_rag_doc_id(doc_id) == excluded_key:
                continue
            blocks_rows = conn.execute(
                """
                SELECT block_id, reason, text, parsed
                FROM blocks
                WHERE doc_id = ?
                ORDER BY block_id
                """,
                (doc_id,),
            ).fetchall()
            blocks_json: Dict[str, Dict[str, Any]] = {}
            for block_row in blocks_rows:
                blocks_json[str(block_row["block_id"])] = {
                    "reason": block_row["reason"] or "",
                    "text": block_row["text"] or "",
                    "parsed": block_row["parsed"] or "",
                }

            doc_ocr = ""
            try:
                ocr_row = conn.execute("SELECT ocr_text FROM docs WHERE doc_id = ?", (doc_id,)).fetchone()
                if ocr_row is not None:
                    doc_ocr = str(ocr_row[0])
            except sqlite3.OperationalError:
                doc_ocr = ""

            results.append(
                {
                    "doc_id": doc_id,
                    "distance": float(row[1]),
                    "doc_ocr": doc_ocr,
                    "blocks_json": blocks_json,
                }
            )
            if len(results) >= int(k):
                break
        return results

    def retrieve_block_examples(
        self,
        query_text: str,
        k: int = 5,
        exclude_doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """检索块级 few-shot 示例。"""
        query_vector = self._embed_text(f"TYPE=BLOCK\n{(query_text or '').strip()}")
        conn = self._get_connection()
        limit = max(int(k), 1) * 4
        sql = """
        WITH candidates AS (
          SELECT id
          FROM items
          WHERE chunk_type = 'block'
            AND (:exclude_doc_id = '' OR doc_id != :exclude_doc_id)
        ),
        knn AS (
          SELECT rowid, distance
          FROM vec_items
          WHERE embedding MATCH :query
            AND rowid IN (SELECT id FROM candidates)
          ORDER BY distance
          LIMIT :limit
        )
        SELECT i.doc_id, i.block_id, knn.distance, b.reason, b.text, b.parsed
        FROM knn
        JOIN items i ON i.id = knn.rowid
        LEFT JOIN blocks b ON b.doc_id = i.doc_id AND b.block_id = i.block_id
        ORDER BY knn.distance
        """
        rows = conn.execute(
            sql,
            {
                "query": json.dumps(query_vector),
                "limit": limit,
                "exclude_doc_id": exclude_doc_id or "",
            },
        ).fetchall()

        results: List[Dict[str, Any]] = []
        excluded_key = normalize_rag_doc_id(exclude_doc_id)
        for row in rows:
            doc_id = str(row[0])
            if excluded_key and normalize_rag_doc_id(doc_id) == excluded_key:
                continue
            results.append(
                {
                    "doc_id": doc_id,
                    "block_id": str(row[1] or ""),
                    "distance": float(row[2]),
                    "reason": row[3] or "",
                    "text": row[4] or "",
                    "parsed": row[5] or "",
                }
            )
            if len(results) >= int(k):
                break
        return results

    def _get_connection(self) -> sqlite3.Connection:
        """为当前线程提供独立数据库连接。"""
        local = threading.current_thread()
        if not hasattr(local, "_invoice_rag_conn"):
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._load_sqlite_vec(conn)
            setattr(local, "_invoice_rag_conn", conn)
        return getattr(local, "_invoice_rag_conn")

    def _load_sqlite_vec(self, conn: sqlite3.Connection) -> None:
        """加载 sqlite-vec 扩展。"""
        try:
            import sqlite_vec
        except ImportError as exc:
            raise RuntimeError("缺少 sqlite-vec，无法启用 RAG 检索") from exc

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

    def _embed_text(self, text: str, retries: int = 2, max_chars: int = 800, min_chars: int = 200) -> List[float]:
        """调用向量接口生成检索向量，并在 413 时自动缩短文本。"""
        if not self.embedding.url or not self.embedding.model or not self.embedding.api_key:
            raise RuntimeError("EMBEDDING_URL、EMBEDDING_MODEL、EMBEDDING_API_KEY 未完整配置")

        used_text = (text or "").strip()
        if max_chars > 0 and len(used_text) > max_chars:
            used_text = used_text[:max_chars]

        headers = {
            "Authorization": f"Bearer {self.embedding.api_key}",
            "Content-Type": "application/json",
        }

        while True:
            payload = {"model": self.embedding.model, "input": used_text}
            last_error: Optional[Exception] = None
            for attempt in range(retries + 1):
                try:
                    response = requests.post(self.embedding.url, json=payload, headers=headers, timeout=120)
                    response.raise_for_status()
                    data = response.json()
                    embedding = data["data"][0]["embedding"]
                    if not isinstance(embedding, list):
                        raise RuntimeError("向量接口返回结果缺少 embedding 列表")
                    return embedding
                except Exception as exc:
                    last_error = exc
                    if attempt >= retries:
                        break
                    time.sleep(2.0 * (2**attempt))

            message = str(last_error or "")
            if "413" in message and len(used_text) > min_chars:
                used_text = used_text[: max(min_chars, len(used_text) // 2)]
                continue
            raise RuntimeError(f"向量接口调用失败: {last_error}")

    def close(self) -> None:
        """关闭当前线程已打开的数据库连接。"""
        local = threading.current_thread()
        if hasattr(local, "_invoice_rag_conn"):
            try:
                getattr(local, "_invoice_rag_conn").close()
            except Exception:
                pass
            delattr(local, "_invoice_rag_conn")


def normalize_rag_doc_id(value: Optional[str]) -> str:
    """统一 RAG 检索中使用的文档编号，便于排除自身和去重。"""
    if not value:
        return ""
    text = Path(str(value)).stem
    text = re.sub(r"\.deepseek_ocr_raw$", "", text, flags=re.IGNORECASE)
    text = text.replace(".deepseek_ocr_raw", "")
    text = re.sub(r"[\s\.-]+", "_", text.strip())
    return text.casefold()
