"""Streamlit front end for the AI Debate Club engine (debate.py)."""
import time
from datetime import datetime
from html import escape

import streamlit as st

import memory
from panel import (LEGACY_LINEUP, PANEL_CSS, PanelState, TokenTracker, next_nodes,
                   render_arch_svg, render_telemetry_html)

DEFAULT_TOPIC = "Airbus fly-by-wire design is safer than Boeing's."   # mirrors debate.py

LINEUP = [
    ("Debaters", "Claude Haiku 4.5 · Amazon Nova 2 Lite (sides drawn per debate)"),
    ("Framing & claims", "Claude Haiku 4.5"),
    ("Intent", "Tavily: whole motion, one search"),
    ("Research", "Each side: own Tavily searches and notebook"),
    ("Fact-check", "Sonnet 4.6 + Haiku 4.5 (parallel)"),
    ("Judge", "Claude Sonnet 4.6"),
    ("Local models", "DeBERTa v3 (style tagger) · bge-small (embeddings)"),
    ("Memory", "SQLite: history, checkpoints, search cache"),
]
AVATAR = {"FOR": "🟦", "AGAINST": "🟥"}

st.set_page_config(page_title="AI Debate Club", page_icon="🎙️", layout="wide")

CSS = """
<style>
header[data-testid="stHeader"] { display: none; }
.block-container, [data-testid="stMainBlockContainer"] {
    padding-top: 0.8rem !important; padding-bottom: 3rem !important; }
.lineup { border-collapse: collapse; font-size: 0.85rem; margin: 0 0 0 auto; }
.lineup tr { background: transparent !important; }
.lineup td { border: none !important; padding: 1px 10px 1px 0; }
.lineup td.label { font-weight: 600; opacity: 0.75; white-space: nowrap; }
.brand-block { text-align: left; }
.brand-block a.home-link { text-decoration: none; color: inherit; }
.app-title { font-size: 2.4rem; font-weight: 700; line-height: 1.2; }
.brand { opacity: 0.75; margin-top: 0.1rem; }
.app-footer { position: fixed; left: 1.2rem; bottom: 0.5rem; font-size: 0.78rem;
              color: #8a83bd; z-index: 1000; }
.history-row { font-size: 0.85rem; padding: 2px 0; }
.history-row a { text-decoration: none; }
</style>
"""

# =============================================================== engine
@st.cache_resource(show_spinner="Loading local HuggingFace models (once per server start)...")
def load_engine():
    import debate   # heavy import: PyTorch plus two HuggingFace models
    return debate.app

# =============================================================== helpers
def md_safe(text):
    """Escape $ so Streamlit doesn't render text between dollar signs as LaTeX."""
    return (text or "").replace("$", "\\$")

def link(title, url):
    return f"[{md_safe(title)}](<{url}>)" if url else md_safe(title)

def bullet_list(items):
    return "\n".join(f"- {md_safe(x)}" for x in items) if items else "_None_"

def draw_panel(arch_ph, tele_ph, panel):
    arch_ph.markdown(render_arch_svg(panel), unsafe_allow_html=True)
    tele_ph.markdown(render_telemetry_html(panel), unsafe_allow_html=True)

def when(ts):
    return datetime.fromtimestamp(ts).strftime("%d %b %Y, %H:%M") if ts else ""

# =============================================================== page sections
def render_header():
    left, right = st.columns([3, 2])
    with left:
        st.markdown('<div class="brand-block"><a class="home-link" href="/" target="_self">'
                    '<div class="app-title">🎙️ AI Debate Club</div>'
                    '<div class="brand">AI Labs : AI Solutions - Ank Kumar</div>'
                    '</a></div>', unsafe_allow_html=True)
    with right:
        rows = "".join(f'<tr><td class="label">{k}:</td><td>{v}</td></tr>'
                       for k, v in LINEUP)
        st.markdown(f'<table class="lineup">{rows}</table>', unsafe_allow_html=True)

def render_footer():
    st.markdown('<div class="app-footer">© 2026 Ank Kumar. All Rights Reserved</div>',
                unsafe_allow_html=True)

def render_history():
    rows = memory.list_debates(20)
    if not rows:
        return
    with st.expander(f"History ({len(rows)} saved debates)"):
        html = "".join(
            f'<div class="history-row">{when(r["started_at"])} · '
            f'<a href="?debate={r["id"]}" target="_self">{escape(r["motion"] or "")}</a> · '
            f'{escape(r["winner"] + " wins" if r["winner"] else r["status"] or "")}</div>'
            for r in rows)
        st.markdown(html, unsafe_allow_html=True)

def render_framing(d):
    lineup = d.get("lineup") or LEGACY_LINEUP
    subjects = ", ".join(d.get("subjects") or []) or "none identified"
    st.info(f"**Reading:** {md_safe(d.get('reading', ''))}  \n"
            f"**Motion:** {md_safe(d.get('proposition', ''))}  \n"
            f"**Subjects:** {md_safe(subjects)}  \n"
            f"🟦 **FOR** ({lineup['FOR']}) **argues:** {md_safe(d.get('for_position', ''))}  \n"
            f"🟥 **AGAINST** ({lineup['AGAINST']}) **argues:** "
            f"{md_safe(d.get('against_position', ''))}")

def render_grounding(d):
    ents = [e for e in d.get("entities") or [] if e.get("url")]
    if ents:
        st.markdown("**Intent sources** · used only to read the motion, never as debate "
                    "evidence\n" + "\n".join(f"- {link(e['title'], e['url'])}" for e in ents))

def render_turn(d):
    e = d["transcript"][-1]
    model = e.get("model") or LEGACY_LINEUP[e["side"]]
    with st.chat_message(e["side"], avatar=AVATAR[e["side"]]):
        st.markdown(f"**{e['side']} · round {e['round']}** · {model}")
        st.markdown(md_safe(e["text"]))
        cites = ", ".join(e["citations"]["valid"]) or "none"
        searches = e.get("searches")
        st.caption(f"Style · emotional {e['tags']['emotional']:.2f} · cites {cites}"
                   + (f" · {searches} searches" if searches is not None else ""))
        if e["citations"]["invalid"]:
            st.caption(f":red[Invalid citations (not in its notebook): "
                       f"{', '.join(e['citations']['invalid'])}]")
        cited = e.get("cited") or []
        if cited:
            with st.expander(f"Check cited passages ({len(cited)}): "
                             + ", ".join(x["id"] for x in cited)):
                for x in cited:
                    st.markdown(f"**[{x['id']}]** [{md_safe(x['title'])}]({x['url']})")
                    st.caption(md_safe(x["text"]))
        found = e.get("sources") or []
        if found:
            with st.expander(f"Sources found this turn ({len(found)}): "
                             f"{found[0]['id']} to {found[-1]['id']}"):
                for x in found:
                    st.markdown(f"**[{x['id']}]** [{md_safe(x['title'])}]({x['url']})")
                    st.caption(md_safe(x["text"]))

def render_claims(d):
    st.divider()
    st.subheader("Fact-check")
    with st.expander(f"{len(d['claims'])} claims extracted"):
        for c in d["claims"]:
            cites = ", ".join(c["citations"]) or "uncited (opinion)"
            st.markdown(f"- **{c['speaker']}**: {md_safe(c['claim'])}  \n  _cites: {cites}_")

AUDIT_SECTIONS = [("verified_claims", "✅ Verified"),
                  ("factual_errors", "❌ Factual errors"),
                  ("contested_claims", "⚖️ Contested (checkers disagree)"),
                  ("unsupported_claims", "❓ Unsupported")]

def render_audit(d):
    st.caption("Each claim was checked only against the passages it cites, by two "
               "independent checkers (Sonnet 4.6 and Haiku 4.5).")
    for key, label in AUDIT_SECTIONS:
        st.markdown(f"**{label}**")
        st.markdown(bullet_list(d["audit"][key]))

def render_verdict(d):
    st.divider()
    st.subheader("Verdict")
    st.success(f"Winner: {d['verdict']['winner']}")
    st.markdown(md_safe(d["verdict"]["reasoning"]))

RENDERERS = {
    "frame": render_framing, "ground": render_grounding,
    "pro": render_turn, "con": render_turn, "extract": render_claims,
    "audit_a": lambda d: None, "audit_b": lambda d: None,   # shown in the panel
    "reconcile": render_audit, "decide": render_verdict,
    "brief": lambda d: None,   # only in debates saved before the cleanup
}

# =============================================================== debate run
def run_debate(graph, topic, rounds, arch_ph, tele_ph):
    panel = PanelState(rounds, topic)
    tracker = TokenTracker()
    panel.start("frame", time.perf_counter())
    draw_panel(arch_ph, tele_ph, panel)

    inputs = {"topic": topic, "round": 1, "max_rounds": rounds, "transcript": []}
    debate_id = memory.new_debate(topic, rounds)   # one id: checkpoints and history rows
    try:
        for update in graph.stream(inputs, config={"callbacks": [tracker],
                                                   "configurable": {"thread_id": debate_id}},
                                   stream_mode="updates"):
            now = time.perf_counter()
            for node, data in update.items():
                memory.record(debate_id, node, data)
                RENDERERS.get(node, lambda d: None)(data)
                if node in panel.status:
                    panel.finish(node, data, now)
                    for nxt in next_nodes(node, data, rounds, panel):
                        panel.start(nxt, now)
            panel.tokens = tracker.by_node
            draw_panel(arch_ph, tele_ph, panel)
    except Exception as exc:
        panel.tokens = tracker.by_node
        panel.fail(time.perf_counter())
        draw_panel(arch_ph, tele_ph, panel)
        memory.fail(debate_id, f"{type(exc).__name__}: {exc}")
        st.error(f"{type(exc).__name__}: {exc}")

def replay_debate(debate_id, arch_ph, tele_ph):
    """Show a saved debate exactly as it ran, from its stored step outputs.
    No Bedrock calls and no Tavily credits."""
    row, events = memory.get_debate(debate_id)
    if not row:
        st.warning("That saved debate was not found.")
        return
    result = f"{row['winner']} won" if row["winner"] else f"status: {row['status']}"
    st.caption(f"Saved debate · {when(row['started_at'])} · {result} · motion: {row['motion']}")
    st.markdown('<a href="/" target="_self">← Start a new debate</a>', unsafe_allow_html=True)
    rounds = row["rounds"] or 3
    panel = PanelState(rounds, row["motion"] or "")
    panel.start("frame", 0.0)
    for node, data in events:
        RENDERERS.get(node, lambda d: None)(data)
        if node in panel.status:
            panel.finish(node, data, 0.0)
            for nxt in next_nodes(node, data, rounds, panel):
                panel.start(nxt, 0.0)
    draw_panel(arch_ph, tele_ph, panel)

# =============================================================== page
st.markdown(CSS + PANEL_CSS, unsafe_allow_html=True)
render_header()
viewing = st.query_params.get("debate")

left, right = st.columns([3, 2], gap="large")

with right:
    with st.container(key="arch-panel"):
        arch_ph = st.empty()
        tele_ph = st.empty()

with left:
    topic = st.text_input("Motion", DEFAULT_TOPIC)
    rounds = st.slider("Rounds", 1, 5, 3)
    start = st.button("Start debate", type="primary", disabled=not topic.strip())
    render_history()

draw_panel(arch_ph, tele_ph, PanelState(rounds, topic.strip()))
render_footer()

with left:
    debate_graph = load_engine()
    results_ph = st.empty()   # one slot: cleared on Start, so no previous run can show
    if start:
        if viewing:
            st.query_params.clear()
        with results_ph.container():
            run_debate(debate_graph, topic.strip(), rounds, arch_ph, tele_ph)
    elif viewing:
        with results_ph.container():
            replay_debate(viewing, arch_ph, tele_ph)
