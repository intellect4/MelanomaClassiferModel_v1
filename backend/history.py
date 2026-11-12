"""Backend SQLite Database Storage for Analyzed Histopathology History.

Persists diagnostic results, staging metadata, and base64-encoded visual overlays
into a local SQLite database file stored directly in the backend codebase.
"""

import os
import shutil
import sqlite3
import tempfile
import uuid
import io
import base64
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image

# Database file stored inside the backend directory (with read-only serverless fallback)
_BUNDLED_DB_PATH = Path(__file__).resolve().parent / "history_storage.db"


def _resolve_db_path() -> Path:
    if os.access(_BUNDLED_DB_PATH.parent, os.W_OK):
        return _BUNDLED_DB_PATH
    tmp_db = Path(tempfile.gettempdir()) / "history_storage.db"
    if not tmp_db.exists() and _BUNDLED_DB_PATH.exists():
        shutil.copy2(_BUNDLED_DB_PATH, tmp_db)
    return tmp_db


HISTORY_DB_PATH = _resolve_db_path()


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(HISTORY_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id TEXT UNIQUE NOT NULL,
            specimen_id TEXT NOT NULL,
            image_hash TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            predicted_idx INTEGER NOT NULL,
            confidence_percent REAL NOT NULL,
            primary_probability REAL NOT NULL,
            metastatic_probability REAL NOT NULL,
            stage TEXT NOT NULL,
            tier TEXT NOT NULL,
            risk TEXT NOT NULL,
            action TEXT NOT NULL,
            color TEXT NOT NULL,
            alpha REAL NOT NULL,
            salient_threshold REAL NOT NULL,
            orig_image_base64 TEXT,
            heatmap_overlay_base64 TEXT,
            salient_foci_base64 TEXT,
            analyzed_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_specimen ON analysis_history (specimen_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_analyzed_at ON analysis_history (analyzed_at DESC)")
    conn.commit()


def init_history_db() -> None:
    """Initializes the SQLite analysis_history table, recovering cleanly if malformed."""
    conn = _get_connection()
    try:
        with conn:
            _create_schema(conn)
    except sqlite3.DatabaseError:
        conn.close()
        if HISTORY_DB_PATH.exists():
            HISTORY_DB_PATH.unlink(missing_ok=True)
        conn = _get_connection()
        with conn:
            _create_schema(conn)
    finally:
        conn.close()


def _pil_to_base64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_pil(b64_str: str) -> Image.Image:
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def save_analysis_record(
    diagnosis: Dict[str, Any],
    image_hash: str,
    alpha: float,
    salient_threshold: float,
) -> str:
    """Persists a diagnostic assessment into backend/history_storage.db.

    If an identical image_hash + specimen_id was just analyzed with updated
    explainability parameters (alpha/salient_threshold), updates the existing
    latest record instead of creating duplicate entries on slider drags.
    """
    init_history_db()

    orig_b64 = diagnosis.get("orig_image_base64")
    heat_b64 = diagnosis.get("heatmap_overlay_base64")
    foci_b64 = diagnosis.get("salient_foci_base64")

    if not orig_b64 and "pil_images" in diagnosis:
        orig_b64 = _pil_to_base64(diagnosis["pil_images"]["original"])
        heat_b64 = _pil_to_base64(diagnosis["pil_images"]["heatmap"])
        foci_b64 = _pil_to_base64(diagnosis["pil_images"]["salient_foci"])

    staging = diagnosis["staging"]
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with _get_connection() as conn:
        # Check if the most recent entry matches this exact specimen upload
        latest = conn.execute(
            "SELECT id, report_id, image_hash, specimen_id FROM analysis_history ORDER BY id DESC LIMIT 1"
        ).fetchone()

        if latest and latest["image_hash"] == image_hash and latest["specimen_id"] == diagnosis["specimen_id"]:
            conn.execute("""
                UPDATE analysis_history
                SET predicted_class = ?, predicted_idx = ?, confidence_percent = ?,
                    primary_probability = ?, metastatic_probability = ?,
                    stage = ?, tier = ?, risk = ?, action = ?, color = ?,
                    alpha = ?, salient_threshold = ?,
                    orig_image_base64 = ?, heatmap_overlay_base64 = ?, salient_foci_base64 = ?,
                    analyzed_at = ?
                WHERE id = ?
            """, (
                diagnosis["predicted_class"],
                diagnosis["predicted_idx"],
                diagnosis["confidence_percent"],
                diagnosis["primary_probability"],
                diagnosis["metastatic_probability"],
                staging["stage"],
                staging["tier"],
                staging["risk"],
                staging["action"],
                staging["color"],
                alpha,
                salient_threshold,
                orig_b64,
                heat_b64,
                foci_b64,
                now_str,
                latest["id"]
            ))
            conn.commit()
            return latest["report_id"]

        report_id = f"MEL-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        conn.execute("""
            INSERT INTO analysis_history (
                report_id, specimen_id, image_hash,
                predicted_class, predicted_idx, confidence_percent,
                primary_probability, metastatic_probability,
                stage, tier, risk, action, color,
                alpha, salient_threshold,
                orig_image_base64, heatmap_overlay_base64, salient_foci_base64,
                analyzed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            report_id,
            diagnosis["specimen_id"],
            image_hash,
            diagnosis["predicted_class"],
            diagnosis["predicted_idx"],
            diagnosis["confidence_percent"],
            diagnosis["primary_probability"],
            diagnosis["metastatic_probability"],
            staging["stage"],
            staging["tier"],
            staging["risk"],
            staging["action"],
            staging["color"],
            alpha,
            salient_threshold,
            orig_b64,
            heat_b64,
            foci_b64,
            now_str
        ))
        conn.commit()
        return report_id


def search_history(
    query: str = "",
    subtype_filter: str = "All",
    stage_filter: str = "All",
    limit: int = 100
) -> List[Dict[str, Any]]:
    """Searches stored diagnostic history records in backend/history_storage.db."""
    init_history_db()
    sql = "SELECT * FROM analysis_history WHERE 1=1"
    params: List[Any] = []

    if query and query.strip():
        q = f"%{query.strip()}%"
        sql += " AND (specimen_id LIKE ? OR report_id LIKE ? OR tier LIKE ? OR risk LIKE ?)"
        params.extend([q, q, q, q])

    if subtype_filter and subtype_filter != "All":
        sql += " AND predicted_class = ?"
        params.append(subtype_filter)

    if stage_filter and stage_filter != "All":
        sql += " AND stage = ?"
        params.append(stage_filter)

    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with _get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


def delete_history_record(record_id: int) -> bool:
    """Deletes a specific history record by its database ID."""
    init_history_db()
    with _get_connection() as conn:
        cur = conn.execute("DELETE FROM analysis_history WHERE id = ?", (record_id,))
        conn.commit()
        return cur.rowcount > 0


def clear_all_history() -> int:
    """Clears all records from the history database."""
    init_history_db()
    with _get_connection() as conn:
        cur = conn.execute("DELETE FROM analysis_history")
        conn.commit()
        return cur.rowcount


# Ensure database file and table exist upon module import
init_history_db()

