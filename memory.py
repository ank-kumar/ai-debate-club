"""SQLite memory for the AI Debate Club, in one file next to the app.

Session memory: LangGraph checkpoints, one thread per debate id.
Overall memory: debates, turns, passages, claims, every step's raw output, and a
Tavily cache so a repeated search costs no credits.
"""
import json
import sqlite3
import threading
import time
import uuid

DB_PATH = "/Users/ank/projects/ai-debate-club/debate_club.db"
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS debates (
    id TEXT PRIMARY KEY, started_at REAL, finished_at REAL, status TEXT,
    motion TEXT, rounds INTEGER, reading TEXT, proposition TEXT,
    for_position TEXT, against_position TEXT, winner TEXT, reasoning TEXT,
    audit_json TEXT, error TEXT);
CREATE TABLE IF NOT EXISTS turns (
    debate_id TEXT, round INTEGER, side TEXT, text TEXT, citations_json TEXT,
    tags_json TEXT, searches INTEGER, at REAL);
CREATE TABLE IF NOT EXISTS passages (
    debate_id TEXT, pid TEXT, side TEXT, title TEXT, url TEXT, text TEXT,
    PRIMARY KEY (debate_id, pid));
CREATE TABLE IF NOT EXISTS claims (
    debate_id TEXT, idx INTEGER, speaker TEXT, claim TEXT, citations_json TEXT);
CREATE TABLE IF NOT EXISTS events (
    debate_id TEXT, node TEXT, at REAL, data_json TEXT);
CREATE TABLE IF NOT EXISTS web_cache (
    query TEXT PRIMARY KEY, pages_json TEXT, fetched_at REAL);
"""

def _connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")   # the UI can read while a debate writes
    conn.executescript(SCHEMA)
    return conn

_conn = _connect()
# A debate still 'running' at server start was interrupted by a restart or refresh.
_conn.execute("UPDATE debates SET status='abandoned' WHERE status='running'")
_conn.commit()

def _exec(sql, args=()):
    with _lock:
        _conn.execute(sql, args)
        _conn.commit()

def _j(x):
    return json.dumps(x, default=str, ensure_ascii=False)

def checkpointer():
    """Session memory: LangGraph saves the full state after every node, per debate id."""
    from langgraph.checkpoint.sqlite import SqliteSaver
    return SqliteSaver(sqlite3.connect(DB_PATH, check_same_thread=False))

def new_debate(motion, rounds):
    debate_id = uuid.uuid4().hex[:12]
    _exec("INSERT INTO debates (id, started_at, status, motion, rounds) VALUES (?,?,?,?,?)",
          (debate_id, time.time(), "running", motion, rounds))
    return debate_id

def record(debate_id, node, data):
    """Overall memory: every step's raw output, plus readable rows for the key steps."""
    _exec("INSERT INTO events VALUES (?,?,?,?)", (debate_id, node, time.time(), _j(data)))
    if node == "frame":
        _exec("UPDATE debates SET reading=?, proposition=?, for_position=?, against_position=? "
              "WHERE id=?", (data.get("reading", ""), data.get("proposition", ""),
                             data.get("for_position", ""), data.get("against_position", ""),
                             debate_id))
    elif node in ("pro", "con") and data.get("transcript"):
        t = data["transcript"][-1]
        _exec("INSERT INTO turns VALUES (?,?,?,?,?,?,?,?)",
              (debate_id, t.get("round"), t.get("side"), t.get("text", ""),
               _j(t.get("citations")), _j(t.get("tags")), t.get("searches", 0), time.time()))
        for x in t.get("sources") or []:
            _exec("INSERT OR REPLACE INTO passages VALUES (?,?,?,?,?,?)",
                  (debate_id, x["id"], x.get("side"), x.get("title"), x.get("url"), x.get("text")))
    elif node == "extract":
        for i, c in enumerate(data.get("claims") or [], 1):
            _exec("INSERT INTO claims VALUES (?,?,?,?,?)",
                  (debate_id, i, c.get("speaker"), c.get("claim"), _j(c.get("citations"))))
    elif node == "reconcile":
        _exec("UPDATE debates SET audit_json=? WHERE id=?", (_j(data.get("audit")), debate_id))
    elif node == "decide":
        v = data.get("verdict") or {}
        _exec("UPDATE debates SET winner=?, reasoning=?, status='done', finished_at=? WHERE id=?",
              (v.get("winner"), v.get("reasoning"), time.time(), debate_id))

def fail(debate_id, error):
    _exec("UPDATE debates SET status='error', error=?, finished_at=? WHERE id=?",
          (error, time.time(), debate_id))

def cache_get(query):
    with _lock:
        row = _conn.execute("SELECT pages_json FROM web_cache WHERE query=?", (query,)).fetchone()
    return json.loads(row[0]) if row else None

def cache_put(query, pages):
    _exec("INSERT OR REPLACE INTO web_cache VALUES (?,?,?)", (query, _j(pages), time.time()))


COLS = ("id", "started_at", "status", "winner", "motion", "rounds")

def list_debates(limit=20):
    """Newest saved debates, for the History box."""
    with _lock:
        rows = _conn.execute("SELECT id, started_at, status, winner, motion, rounds "
                             "FROM debates ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(zip(COLS, r)) for r in rows]

def get_debate(debate_id):
    """(debate row, [(node, data), ...] in the order they ran), for replay."""
    with _lock:
        row = _conn.execute("SELECT id, started_at, status, winner, motion, rounds "
                            "FROM debates WHERE id=?", (debate_id,)).fetchone()
        events = _conn.execute("SELECT node, data_json FROM events WHERE debate_id=? "
                               "ORDER BY rowid", (debate_id,)).fetchall()
    if not row:
        return None, []
    return dict(zip(COLS, row)), [(n, json.loads(d)) for n, d in events]
