from typing import List, Sequence, Protocol
from pathlib import Path
import sqlite3
import hashlib
import os
import threading
import numpy as np

__all__ = ["GranuscoreCache"]


def _md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


class _Scorer(Protocol):
    def score_wo_cache(self, answers: Sequence[str], encoding_batch_size: int | None = None) -> np.ndarray | list: ...


class GranuscoreCache:
    """Thin wrapper around the Granuscore with SQLite caching."""

    def __init__(
            self,
            scorer: "_Scorer",
            hierarchy_model: str,
            faiss_index: str,
            lgb_model: str,
            cache_dir: str | os.PathLike = None,
    ) -> None:
        self.scorer = scorer
        self.hierarchy_model = hierarchy_model
        self.faiss_index = faiss_index
        self.lgb_model = lgb_model

        if cache_dir is None:
            cache_dir = os.getenv("GRANUSCORE_CACHE_DIR", f"{Path(__file__).parent}/.cache") # default is .cache dir next to this file
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        self.db_path = cache_path / "granuscores.db"

        self._init_db()

        self._w_lock = threading.Lock()

    def get(self, text: str) -> float:
        """Return the granuscore for *text*, fetching from cache or scoring."""
        text_hash = _md5(text)

        # 1. try cache ------------------------------------------------------
        row = self._select(text_hash)
        if row is not None:
            return row

        score = self.scorer.score_wo_cache([text])[0]
        self._insert(text_hash, text, score)
        return score

    def batch(self, texts: Sequence[str], encoding_batch_size: int | None = None) -> List[float]:
        texts = list(texts)
        if not texts:
            return []

        # Precompute hashes and preserve order
        hashes = [_md5(t) for t in texts]

        # 1) Bulk fetch cache hits
        hit_map = self._select_many(hashes)  # {text_hash: granuscore}

        # 2) Identify misses (preserve order)
        miss_texts = [t for t, h in zip(texts, hashes) if h not in hit_map]
        miss_hashes = [h for h in hashes if h not in hit_map]

        # 3) Score misses in one model call
        if miss_texts:
            miss_scores = self.scorer.score_wo_cache(miss_texts, encoding_batch_size)
            miss_scores = np.asarray(miss_scores, dtype=float)

            # 4) Bulk insert misses
            self._insert_many(miss_hashes, miss_texts, miss_scores)

            # update hit_map so we can fill outputs
            hit_map.update({h: float(s) for h, s in zip(miss_hashes, miss_scores)})

        # 5) Build output aligned to input order
        return [float(hit_map[h]) for h in hashes]

    # ------------------------------------------------------------------
    # private db helpers
    # ------------------------------------------------------------------
    def _open_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            timeout=30,
            check_same_thread=False,
            isolation_level=None  # autocommit
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS granuscores (
                    text_hash       TEXT,
                    hierarchy_model TEXT,
                    faiss_index TEXT,
                    lgb_model  TEXT,
                    granuscore REAL NOT NULL,
                    text       TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(text_hash, hierarchy_model, faiss_index, lgb_model)
                )
                """
            )

            conn.execute("CREATE INDEX IF NOT EXISTS idx_model ON granuscores(hierarchy_model, faiss_index, lgb_model);")

    def _select(self, text_hash: str) -> float | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT granuscore FROM granuscores WHERE text_hash=? AND hierarchy_model=? AND faiss_index=? AND lgb_model=?",
                (text_hash, self.hierarchy_model, self.faiss_index, self.lgb_model),
            ).fetchone()
            if row:
                return row[0]
            return None

    def _select_many(self, text_hashes: Sequence[str], chunk_size: int = 900) -> dict[str, float]:
        """
        Fetch many hashes at once. Chunked because SQLite has a limit on bound variables
        (commonly 999).
        """
        out: dict[str, float] = {}
        if not text_hashes:
            return out

        with sqlite3.connect(self.db_path) as conn:
            for i in range(0, len(text_hashes), chunk_size):
                chunk = text_hashes[i:i + chunk_size]
                placeholders = ",".join(["?"] * len(chunk))

                rows = conn.execute(
                    f"""
                    SELECT text_hash, granuscore
                    FROM granuscores
                    WHERE hierarchy_model = ?
                      AND faiss_index = ?
                      AND lgb_model = ?
                      AND text_hash IN ({placeholders})
                    """,
                    (self.hierarchy_model, self.faiss_index, self.lgb_model, *chunk),
                ).fetchall()

                out.update({h: float(s) for h, s in rows})

        return out

    def _insert(self, text_hash: str, text: str, granusccore: float) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO granuscores (text_hash, hierarchy_model, faiss_index, lgb_model, granuscore, text) VALUES (?,?,?,?,?,?)",
                (text_hash, self.hierarchy_model, self.faiss_index, self.lgb_model, granusccore, text),
            )

    def _insert_many(self, hashes: Sequence[str], texts: Sequence[str], scores: Sequence[float]) -> None:
        """
        Bulk insert inside one transaction. Much faster than opening a connection per row.
        """
        if not hashes:
            return

        rows = [
            (h, self.hierarchy_model, self.faiss_index, self.lgb_model, float(s), t)
            for h, t, s in zip(hashes, texts, scores)
        ]

        with self._w_lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO granuscores
                    (text_hash, hierarchy_model, faiss_index, lgb_model, granuscore, text)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )