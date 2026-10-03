import sqlite3
import struct
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "vigil_ia.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    video TEXT NOT NULL,
    category TEXT NOT NULL,
    classification TEXT NOT NULL,
    risk REAL NOT NULL DEFAULT 0,
    frames INTEGER NOT NULL DEFAULT 0,
    suspicious_frames REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    analyzed_by TEXT NOT NULL DEFAULT 'admin'
);
"""

DEMO_ROWS = [
    (
        "demo-001",
        "Shoplifting (60).mp4",
        "suspeitos",
        "FURTO",
        78.4,
        1840,
        0.31,
        "Concluída",
        "2026-09-13 10:42",
    ),
    (
        "demo-002",
        "Normal (21).mp4",
        "normais",
        "NAO_FURTO",
        19.4,
        1610,
        0.02,
        "Concluída",
        "2026-09-13 10:31",
    ),
]


def connect():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize():
    with connect() as connection:
        connection.executescript(SCHEMA)
        columns = {row[1] for row in connection.execute("PRAGMA table_info(analyses)")}
        if "analyzed_by" not in columns:
            connection.execute(
                "ALTER TABLE analyses ADD COLUMN analyzed_by TEXT NOT NULL DEFAULT 'admin'"
            )
        connection.execute(
            "UPDATE analyses SET status = 'EM_ANALISE' "
            "WHERE status = 'Fila criada'"
        )


def list_analyses():
    with connect() as connection:
        rows = connection.execute(
            "SELECT * FROM analyses ORDER BY updated_at DESC"
        ).fetchall()
    analyses = []
    for row in rows:
        analysis = dict(row)
        if isinstance(analysis["risk"], bytes) and len(analysis["risk"]) == 4:
            analysis["risk"] = float(struct.unpack("<f", analysis["risk"])[0])
        analyses.append(analysis)
    return analyses


def create_analysis(analysis_id, video, category, analyzed_by="admin"):
    analysis = {
        "id": analysis_id,
        "video": video,
        "category": category,
        "classification": "EM_ANALISE",
        "risk": 0.0,
        "frames": 0,
        "suspicious_frames": 0.0,
        "status": "EM_ANALISE",
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "analyzed_by": analyzed_by,
    }
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO analyses
            (id, video, category, classification, risk, frames,
             suspicious_frames, status, updated_at, analyzed_by)
                VALUES (:id, :video, :category, :classification, :risk,
                    :frames, :suspicious_frames, :status, :updated_at,
                    :analyzed_by)
            """,
            analysis,
        )
    return analysis


def update_analysis(analysis_id, **fields):
    if not fields:
        return
    assignments = ", ".join(f"{field} = ?" for field in fields)
    values = list(fields.values()) + [analysis_id]
    with connect() as connection:
        connection.execute(
            f"UPDATE analyses SET {assignments} WHERE id = ?",
            values,
        )
