"""SQLite history for AI-Summarizer.
Stores every summary with the active login session_id so the UI can
filter by date and by session.
"""
import os
import sqlite3
from pathlib import Path

_render_data = Path("/var/data")
_DATA_DIR = _render_data if os.getenv("RENDER") and _render_data.exists() else Path(__file__).parent
DB_PATH = _DATA_DIR / "ais.db"


class SummarizerDatabase:
    def __init__(self):
        self.db_path = str(DB_PATH)
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS summary_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT,
                source_preview TEXT,
                summary TEXT NOT NULL,
                length TEXT,
                grade_level TEXT,
                language TEXT,
                word_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # backfill session_id for older DBs
        try:
            cols = [r[1] for r in c.execute("PRAGMA table_info(summary_history)").fetchall()]
            if 'session_id' not in cols:
                c.execute("ALTER TABLE summary_history ADD COLUMN session_id TEXT")
        except Exception as _e:
            print(f"[DB] session_id migration warning: {_e}")
        conn.commit()
        conn.close()
        print("[DB] AI-Summarizer database initialized")

    def save_summary(self, user_id: str, source_preview: str, summary: str,
                     length: str = "", grade_level: str = "", language: str = "",
                     word_count: int = 0, session_id: str = None) -> int:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO summary_history
            (user_id, session_id, source_preview, summary, length, grade_level, language, word_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, session_id, (source_preview or '')[:200], summary,
              length, grade_level, language, int(word_count or 0)))
        sid = c.lastrowid
        conn.commit()
        conn.close()
        self._cleanup(user_id, keep=100)
        return sid

    def _cleanup(self, user_id: str, keep: int = 100):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            SELECT id FROM summary_history WHERE user_id = ?
            ORDER BY created_at DESC LIMIT -1 OFFSET ?
        ''', (user_id, keep))
        rows = c.fetchall()
        if rows:
            ids = [r[0] for r in rows]
            ph = ','.join('?' * len(ids))
            c.execute(f'DELETE FROM summary_history WHERE id IN ({ph})', ids)
        conn.commit()
        conn.close()

    def get_history(self, user_id: str, date_from: str = None, date_to: str = None,
                    session_id: str = None, limit: int = 100) -> list:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        sql = '''SELECT id, session_id, source_preview, summary, length, grade_level,
                        language, word_count, created_at
                 FROM summary_history WHERE user_id = ?'''
        params = [user_id]
        if date_from:
            sql += ' AND date(created_at) >= date(?)'; params.append(date_from)
        if date_to:
            sql += ' AND date(created_at) <= date(?)'; params.append(date_to)
        if session_id:
            sql += ' AND session_id = ?'; params.append(session_id)
        sql += ' ORDER BY created_at DESC LIMIT ?'
        params.append(int(limit))
        c.execute(sql, params)
        rows = c.fetchall()
        conn.close()
        return [{
            'id': r[0], 'session_id': r[1],
            'source_preview': r[2] or '', 'summary': r[3] or '',
            'length': r[4], 'grade_level': r[5], 'language': r[6],
            'word_count': r[7], 'created_at': r[8],
            'preview': (r[3] or '')[:120],
        } for r in rows]

    def list_sessions(self, user_id: str) -> list:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            SELECT session_id, COUNT(*), MIN(created_at), MAX(created_at)
            FROM summary_history
            WHERE user_id = ? AND session_id IS NOT NULL AND session_id != ''
            GROUP BY session_id ORDER BY MAX(created_at) DESC LIMIT 50
        ''', (user_id,))
        rows = c.fetchall()
        conn.close()
        return [{'session_id': r[0], 'count': r[1], 'first_at': r[2], 'last_at': r[3]} for r in rows]


db = SummarizerDatabase()
