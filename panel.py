"""Live architecture diagram and telemetry panel for the AI Debate Club UI."""
from html import escape

from langchain_core.callbacks import BaseCallbackHandler

# Debates saved before per-debate side draws always had Haiku FOR and Nova AGAINST.
LEGACY_LINEUP = {"FOR": "Claude Haiku 4.5", "AGAINST": "Amazon Nova 2 Lite"}
UNDRAWN = {"FOR": "Drawn per debate", "AGAINST": "Drawn per debate"}

# (graph node, display name)
STAGES = [
    ("frame", "Motion framing"),
    ("ground", "Intent sources"),
    ("pro", "FOR agent"),
    ("con", "AGAINST agent"),
    ("extract", "Claim extractor"),
    ("audit_a", "Auditor A"),
    ("audit_b", "Auditor B"),
    ("reconcile", "Reconcile"),
    ("decide", "Judge"),
]

PANEL_CSS = """
<style>
@keyframes arch-pulse { 0%, 100% { stroke-opacity: 1; } 50% { stroke-opacity: 0.3; } }
@keyframes arch-blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.2; } }
.arch-active { animation: arch-pulse 1.2s ease-in-out infinite; }
.arch-blink { animation: arch-blink 1s ease-in-out infinite; }
.st-key-arch-panel { position: sticky; top: 0.8rem; }
.panel-card { background: #17123a; border: 1px solid #3b3478; border-radius: 14px;
              padding: 10px 12px; margin-bottom: 12px; }
.panel-title { font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase;
               color: #8a83bd; margin-bottom: 6px; }
.tele { width: 100%; border-collapse: collapse; font-size: 0.74rem; }
.tele tr { background: transparent !important; }
.tele th, .tele td { border: none !important; padding: 3px 5px; text-align: right;
                     white-space: nowrap; }
.tele th:first-child, .tele td:first-child { text-align: left; }
.tele th { color: #8a83bd; font-weight: 600; border-bottom: 1px solid #3b3478 !important; }
.tele td.detail { text-align: left; white-space: normal; color: #b9b0ec; }
.tele tr.total td { border-top: 1px solid #3b3478 !important; font-weight: 600; }
.dot-pending { color: #4b4386; } .dot-done { color: #86efac; } .dot-failed { color: #f87171; }
.dot-active { color: #c4b5fd; animation: arch-blink 1s ease-in-out infinite; }
.s-pending { opacity: 0.5; } .s-active { color: #c4b5fd; } .s-done { color: #86efac; }
.s-failed { color: #f87171; }
.tele-note { font-size: 0.68rem; color: #8a83bd; margin-top: 6px; }
</style>
"""

def next_nodes(node, data, max_rounds, panel):
    """Nodes that start after `node` finishes (mirrors the edges in debate.py)."""
    if node == "con":
        return ["pro"] if data["round"] <= max_rounds else ["extract"]
    if node == "extract":
        return ["audit_a", "audit_b"]                       # fan-out
    if node in ("audit_a", "audit_b"):
        other = "audit_b" if node == "audit_a" else "audit_a"
        return ["reconcile"] if panel.status[other] == "done" else []   # fan-in
    return {"frame": ["ground"], "ground": ["pro"], "pro": ["con"],
            "reconcile": ["decide"]}.get(node, [])

class TokenTracker(BaseCallbackHandler):
    """Counts Bedrock calls and tokens per LangGraph node, using the node name that
    LangGraph adds to each call's metadata ('langgraph_node')."""
    def __init__(self):
        self.by_node = {}
        self._run_node = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        self._run_node[run_id] = (metadata or {}).get("langgraph_node", "other")

    def on_llm_end(self, response, *, run_id, **kwargs):
        node = self._run_node.pop(run_id, "other")
        stats = self.by_node.setdefault(node, {"calls": 0, "in": 0, "out": 0})
        stats["calls"] += 1
        for generations in response.generations:
            for g in generations:
                usage = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                stats["in"] += usage.get("input_tokens", 0)
                stats["out"] += usage.get("output_tokens", 0)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._run_node.pop(run_id, None)

class PanelState:
    """Status, timing and details for each stage of one debate run."""
    def __init__(self, max_rounds, topic=""):
        self.max_rounds, self.topic = max_rounds, topic
        self.round = 0
        self.lineup = dict(UNDRAWN)
        self.status = {k: "pending" for k, _ in STAGES}
        self.seconds = {k: 0.0 for k, _ in STAGES}
        self.detail = {k: "" for k, _ in STAGES}
        self.turns = {"pro": 0, "con": 0}
        self.cites = {"pro": [0, 0], "con": [0, 0]}     # valid, invalid
        self.searches = {"pro": 0, "con": 0}
        self.tokens = {}
        self.winner = None
        self.run_start = None
        self.wall = 0.0
        self._t0 = {}

    def start(self, node, now):
        if self.run_start is None:
            self.run_start = now
        self.status[node] = "active"
        self._t0[node] = now
        if node == "pro":
            self.round += 1

    def finish(self, node, data, now):
        self.status[node] = "done"
        self.seconds[node] += now - self._t0.get(node, now)
        self.wall = now - (self.run_start or now)
        if node == "frame":
            self.lineup = data.get("lineup") or dict(LEGACY_LINEUP)
            self.detail[node] = f"{len(data.get('subjects') or [])} subjects"
        elif node == "ground":
            n = len([e for e in data.get("entities") or [] if e.get("url")])
            self.detail[node] = f"{n} intent sources"
        elif node in self.turns:
            t = data["transcript"][-1]
            self.turns[node] += 1
            self.cites[node][0] += len(t["citations"]["valid"])
            self.cites[node][1] += len(t["citations"]["invalid"])
            self.searches[node] += t.get("searches", 0)
            n, (v, inv), s = self.turns[node], self.cites[node], self.searches[node]
            self.detail[node] = (f"{n} turn{'s' if n > 1 else ''} · {v} cites · "
                                 f"{s} searches" + (f" · {inv} invalid" if inv else ""))
        elif node == "extract":
            cited = sum(1 for c in data["claims"] if c["citations"])
            self.detail[node] = f"{len(data['claims'])} claims · {cited} cited"
        elif node in ("audit_a", "audit_b"):
            s = [r["status"] for r in data[node]]
            self.detail[node] = (f"{s.count('verified')} ✓ · {s.count('error')} ✗ · "
                                 f"{s.count('unsupported')} ?")
        elif node == "reconcile":
            a = data["audit"]
            self.detail[node] = (f"{len(a['verified_claims'])} ✓ · "
                                 f"{len(a['factual_errors'])} ✗ · "
                                 f"{len(a['contested_claims'])} ⚖ · "
                                 f"{len(a['unsupported_claims'])} ?")
        elif node == "decide":
            self.winner = data["verdict"]["winner"]
            self.detail[node] = f"{self.winner} wins"

    def fail(self, now):
        """A crash (for example a Bedrock or network error): mark running stages failed."""
        for k, v in self.status.items():
            if v == "active":
                self.status[k] = "failed"
        if self.run_start is not None:
            self.wall = now - self.run_start

# =============================================================== diagram
COLORS = {   # fill, stroke, title, subtitle
    "pending": ("#1d1842", "#3b3478", "#8a83bd", "#6b6499"),
    "active":  ("#33288a", "#c4b5fd", "#ffffff", "#ddd6fe"),
    "done":    ("#241d58", "#8b7cf0", "#ece9ff", "#b9b0ec"),
    "failed":  ("#3a1830", "#f87171", "#fecaca", "#fca5a5"),
}
LINE = "#6d5fd6"

def _box(x, y, w, h, title, sub, state):
    fill, stroke, tc, sc = COLORS[state]
    anim = ' class="arch-active"' if state == "active" else ""
    width = 2 if state == "active" else 1.2
    ty, sy = (y + 20, y + 35) if h < 50 else (y + 23, y + 40)
    out = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" '
           f'stroke="{stroke}" stroke-width="{width}"{anim}/>'
           f'<text x="{x + 12}" y="{ty}" fill="{tc}" font-size="12.5" '
           f'font-weight="600">{escape(title)}</text>'
           f'<text x="{x + 12}" y="{sy}" fill="{sc}" font-size="10">{escape(sub)}</text>')
    cx, cy = x + w - 13, y + 13
    if state == "done":
        out += (f'<circle cx="{cx}" cy="{cy}" r="6.5" fill="#8b7cf0"/>'
                f'<path d="M{cx - 3} {cy} l2 2.2 l4 -4.4" stroke="#ffffff" stroke-width="1.5" '
                f'fill="none" stroke-linecap="round"/>')
    elif state == "active":
        out += f'<circle cx="{cx}" cy="{cy}" r="4" fill="#c4b5fd" class="arch-blink"/>'
    elif state == "failed":
        out += (f'<text x="{cx}" y="{cy + 4}" fill="#f87171" font-size="13" '
                f'font-weight="700" text-anchor="middle">!</text>')
    return out

def _chip(x, y, w, h, label, state):
    fill, stroke, tc, _ = COLORS[state]
    anim = ' class="arch-active"' if state == "active" else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h / 2}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1.2"{anim}/>'
            f'<text x="{x + w / 2}" y="{y + h / 2 + 3.5}" fill="{tc}" font-size="10.5" '
            f'text-anchor="middle">{escape(label)}</text>')

def _section(x, y, w, h, label):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="none" '
            f'stroke="#2c2566" stroke-width="1"/>'
            f'<text x="{x + 12}" y="{y + 15}" fill="#7f78b3" font-size="9.5" '
            f'letter-spacing="1.2">{label}</text>')

def _arrow(x1, y1, x2, y2):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{LINE}" '
            f'stroke-width="1.4" marker-end="url(#ah)"/>')

def _line(d):
    return f'<path d="{d}" fill="none" stroke="{LINE}" stroke-width="1.4"/>'

def render_arch_svg(p):
    s = p.status
    loop_active = "active" in (s["pro"], s["con"])
    loop_state = "active" if loop_active else ("done" if p.turns["pro"] else "pending")
    if p.winner:
        result, result_sub = "done", f"{p.winner} wins"
    elif "failed" in s.values():
        result, result_sub = "failed", "Run failed"
    else:
        result, result_sub = "pending", "Awaiting verdict"
    rnd = (f"Round {p.round} / {p.max_rounds}" if p.round else f"{p.max_rounds} rounds")

    parts = [
        '<svg viewBox="0 0 520 492" width="100%" xmlns="http://www.w3.org/2000/svg" '
        'style="font-family: sans-serif;">',
        f'<defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" '
        f'markerHeight="6" orient="auto"><path d="M1 1L8 5L1 9" fill="none" '
        f'stroke="{LINE}" stroke-width="1.6"/></marker></defs>',
        # intent: framing -> intent sources
        _section(8, 6, 504, 90, "INTENT"),
        _box(20, 30, 230, 56, "Motion framing", "Claude Haiku 4.5 · spelling + framing",
             s["frame"]),
        _box(270, 30, 230, 56, "Intent sources", "Tavily · whole motion, one search",
             s["ground"]),
        _arrow(250, 58, 268, 58),
        _arrow(260, 96, 260, 112),
        # debate loop
        _section(8, 114, 504, 126, "DEBATE LOOP · OWN RESEARCH"),
        f'<text x="500" y="129" fill="#c4b5fd" font-size="11" text-anchor="end">{rnd}</text>',
        _box(20, 138, 200, 56, "FOR agent", p.lineup["FOR"], s["pro"]),
        _box(300, 138, 200, 56, "AGAINST agent", p.lineup["AGAINST"], s["con"]),
        _arrow(220, 158, 298, 158),
        _arrow(300, 176, 222, 176),
        _chip(40, 204, 440, 26, "Each side: own Tavily research · Style tagger: DeBERTa (local)",
              loop_state),
        _arrow(260, 240, 260, 256),
        # fact-check: extractor -> two parallel auditors
        _section(8, 258, 504, 126, "FACT-CHECK"),
        '<text x="500" y="273" fill="#7f78b3" font-size="9.5" text-anchor="end" '
        'letter-spacing="1.2">PARALLEL</text>',
        _box(20, 292, 148, 60, "Claim extractor", "Claude Haiku 4.5", s["extract"]),
        _box(352, 282, 148, 46, "Auditor A", "Claude Sonnet 4.6", s["audit_a"]),
        _box(352, 334, 148, 46, "Auditor B", "Claude Haiku 4.5", s["audit_b"]),
        _arrow(168, 316, 350, 305),
        _arrow(168, 328, 350, 357),
        # fan-in to reconcile
        _line("M500 305 L506 305 L506 404"),
        _line("M426 380 L426 404"),
        _line("M506 404 L94 404"),
        _arrow(94, 404, 94, 418),
        # verdict
        _section(8, 392, 504, 92, "VERDICT"),
        _box(20, 420, 148, 56, "Reconcile", "code, no LLM", s["reconcile"]),
        _box(186, 420, 148, 56, "Judge", "Claude Sonnet 4.6", s["decide"]),
        _box(352, 420, 148, 56, "Final result", result_sub, result),
        _arrow(168, 448, 184, 448),
        _arrow(334, 448, 350, 448),
        "</svg>",
    ]
    return ('<div class="panel-card"><div class="panel-title">Architecture · live</div>'
            + "".join(parts) + "</div>")

# =============================================================== telemetry
STATUS_LABEL = {"pending": "Pending", "active": "Running", "done": "Done", "failed": "Failed"}

def render_telemetry_html(p):
    rows = []
    total = {"calls": 0, "in": 0, "out": 0}
    for key, name in STAGES:
        t = p.tokens.get(key, {"calls": 0, "in": 0, "out": 0})
        for k in total:
            total[k] += t[k]
        st_ = p.status[key]
        rows.append(f'<tr><td><span class="dot-{st_}">●</span> {escape(name)}</td>'
                    f'<td class="s-{st_}">{STATUS_LABEL[st_]}</td>'
                    f'<td>{p.seconds[key]:.1f}s</td><td>{t["calls"]}</td>'
                    f'<td>{t["in"]:,}</td><td>{t["out"]:,}</td>'
                    f'<td class="detail">{escape(p.detail[key])}</td></tr>')
    rows.append(f'<tr class="total"><td>Total (wall clock)</td><td></td>'
                f'<td>{p.wall:.1f}s</td><td>{total["calls"]}</td>'
                f'<td>{total["in"]:,}</td><td>{total["out"]:,}</td><td></td></tr>')
    head = ("<tr><th>Stage</th><th>Status</th><th>Time</th><th>Calls</th>"
            "<th>Tokens in</th><th>Tokens out</th><th>Detail</th></tr>")
    return ('<div class="panel-card"><div class="panel-title">Telemetry</div>'
            f'<table class="tele">{head}{"".join(rows)}</table>'
            '<div class="tele-note">Calls and tokens are Bedrock usage only; local '
            'HuggingFace models use none. Tavily: 1 credit for the intent search plus 1 per '
            'debater search; repeated searches come from the SQLite cache at no cost. '
            'Auditors A and B run in parallel, so their times overlap; the total is '
            'wall-clock time. Audit: ✓ verified · ✗ error · ⚖ contested · ? unsupported.'
            '</div></div>')
