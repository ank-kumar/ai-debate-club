"""AI Debate Club engine.

Intent: the input is spelling-corrected and searched once on Tavily as one phrase;
Claude Haiku 4.5 frames the motion from it (both sides kept whole, FOR = the
first-named side, direct opposite positions). Every motion is debated.

Debate: two research agents (Claude Haiku 4.5 and Amazon Nova 2 Lite; which one argues
FOR is drawn per debate) search the web themselves, keep their own notebooks (F# / A#
passages) and cite them. A local DeBERTa model scores each turn's emotional style.

Fact-check: claims are extracted and checked in parallel by Sonnet 4.6 and Haiku 4.5,
each claim only against the passages it cites; verdicts are reconciled in code and
Sonnet 4.6 judges.

Graph:  frame -> ground (intent sources) -> pro <-> con (N rounds) -> extract
        -> [audit_a || audit_b] -> reconcile -> decide

Run:  python debate.py ["optional motion"]
Environment (.env next to this file):
  TAVILY_API_KEY     intent search and the debaters' research
  HF_HUB_OFFLINE=0   allow HuggingFace model downloads (default: cache only)
  DEBATE_DEVICE=cpu  run the local models on CPU instead of Apple GPU
  DEBATE_AWS_REGION  Bedrock region (default: us-east-1)
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import difflib
import operator
import random
import re
import sys
from typing import Annotated, Literal, TypedDict

import torch
from botocore.config import Config
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient
from transformers import pipeline

import memory   # SQLite: checkpoints, history, Tavily cache

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(PROJECT_DIR, ".env"))

# =============================================================== config
REGION = os.environ.get("DEBATE_AWS_REGION", "us-east-1")
DEFAULT_TOPIC = "Airbus fly-by-wire design is safer than Boeing's."

HAIKU = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
NOVA = "global.amazon.nova-2-lite-v1:0"
SONNET = "global.anthropic.claude-sonnet-4-6"

TAGGER_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEVICE = os.environ.get("DEBATE_DEVICE") or (
    "mps" if torch.backends.mps.is_available() else "cpu")

WEB_RESULTS = 5              # Tavily results per search (1 credit per search)
CHUNK_CHARS = 900            # max characters per passage
PASSAGES_PER_SEARCH = 6      # passages a debater keeps from one search
MAX_SEARCHES_PER_TURN = 2
REPEAT_THRESHOLD = 0.92      # a turn this similar to the side's last turn is re-asked
MAX_CLAIMS = 6               # up to 3 per side
STRUCTURED_RETRIES = 3

# =============================================================== models
# Fail fast on stalled Bedrock calls instead of botocore's 60 s default.
BEDROCK_CONFIG = Config(connect_timeout=10, read_timeout=45,
                        retries={"max_attempts": 3, "mode": "adaptive"})

def bedrock(model_id, max_tokens):
    return ChatBedrockConverse(model=model_id, region_name=REGION, max_tokens=max_tokens,
                               config=BEDROCK_CONFIG)

DEBATERS = {"Claude Haiku 4.5": bedrock(HAIKU, 400),
            "Amazon Nova 2 Lite": bedrock(NOVA, 400)}
extract_llm = bedrock(HAIKU, 1000)    # spelling fix, framing and claim extraction
judge_llm = bedrock(SONNET, 3000)     # auditor A and the judge
checker_llm = bedrock(HAIKU, 3000)    # auditor B: a different model, for independence

tagger = pipeline("zero-shot-classification", model=TAGGER_MODEL, device=DEVICE)
EMOTIONAL_LABEL = "This text tries to make the reader feel fear or worry."
embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)

_tavily_key = os.environ.get("TAVILY_API_KEY")
tavily = TavilyClient(_tavily_key) if _tavily_key else None

def tag(text):
    """Emotional-style score for a debate turn (0 to 1). Display only."""
    if not text.strip():
        return {"emotional": 0.0}
    r = tagger(text, candidate_labels=[EMOTIONAL_LABEL], multi_label=True,
               hypothesis_template="{}")
    return {"emotional": round(r["scores"][0], 2)}

# =============================================================== web search
_web_cache = {}   # query -> pages, for this process; SQLite keeps them across restarts

def web_search(query):
    """Tavily search (1 credit, or free from the cache). Returns (pages, error)."""
    if query in _web_cache:
        return _web_cache[query], None
    cached = memory.cache_get(query)
    if cached is not None:
        _web_cache[query] = cached
        return cached, None
    if tavily is None:
        return [], "TAVILY_API_KEY is not set"
    try:
        r = tavily.search(query, max_results=WEB_RESULTS)
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"
    pages = [{"title": x.get("title") or x.get("url", ""), "url": x.get("url", ""),
              "text": (x.get("content") or "").strip()}
             for x in r.get("results", []) if (x.get("content") or "").strip()]
    _web_cache[query] = pages
    if pages:
        memory.cache_put(query, pages)
    return pages, None

WEB_BOILERPLATE = re.compile(
    r"^(reply|pingback|trackback|copyright|cite this|copy citation|share|read more|"
    r"see (all|more)|browse nearby|rhymes for|phrases containing|examples of|"
    r"advertisement|sign up|subscribe|log in|last updated|send us feedback)\b", re.I)
SITE_CHROME = re.compile(
    r"install (our site|the app)|out of date browser|uses cookies|similar threads|share link|"
    r"follow along with the video|may not be available in some browsers|helpful links|"
    r"connect with us|site content|drug topics|consenting to our use", re.I)

def clean_web_text(text):
    """Drop page chrome from a search result: markdown headings, list numbers,
    comment and pingback lines, site navigation and cookie notices."""
    lines = []
    for ln in text.splitlines():
        ln = re.sub(r"#{1,6}\s*", "", re.sub(r"^\s*\d+\.\s+", "", ln)).strip()
        ln = " ".join(x for x in re.split(r"(?<=[.!?])\s+", ln) if not SITE_CHROME.search(x))
        if ln and not WEB_BOILERPLATE.match(ln):
            lines.append(ln)
    return "\n".join(lines)

def chunk(text, title):
    """Split text into passages of up to CHUNK_CHARS on sentence boundaries,
    dropping fragments under 80 characters."""
    passages = []
    for para in text.split("\n"):
        para = para.strip()
        if len(para) < 80:
            continue
        buf = ""
        for sent in para.replace(". ", ".|").split("|"):
            if buf and len(buf) + len(sent) > CHUNK_CHARS:
                passages.append(buf.strip())
                buf = ""
            buf += sent + " "
        if buf.strip():
            passages.append(buf.strip())
    return [{"title": title, "text": p} for p in passages]

# =============================================================== text helpers
CONTRAST = re.compile(r",\s+not\s+|,?\s+rather than\s+|,?\s+instead of\s+")
CITE_GROUP = re.compile(r"\[([^\]]*)\]")
CITE_ID = re.compile(r"[FA]\d+")   # F# = FOR's notebook, A# = AGAINST's

def trim_contrast(claim):
    """'X, not Y' -> 'X.'  Enforced in code because prompts alone did not hold."""
    return CONTRAST.split(claim, maxsplit=1)[0].rstrip(" ,.") + "."

def find_citations(text, passages):
    """Evidence IDs cited in a turn, split into known passages and unknown ones."""
    ids = list(dict.fromkeys(i for g in CITE_GROUP.findall(text) for i in CITE_ID.findall(g)))
    known = {p["id"] for p in passages}
    return {"valid": [i for i in ids if i in known], "invalid": [i for i in ids if i not in known]}

def normalize(text):
    """Lowercase, unify quotes and dashes, collapse whitespace (for quote matching)."""
    for a, b in [("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'), ("\u201d", '"'),
                 ("\u2013", "-"), ("\u2014", "-")]:
        text = text.replace(a, b)
    return " ".join(text.lower().split()).strip(" '\".")

def fmt_passage(p):
    return f"[{p['id']}] ({p['source']}: {p['title']}) {p['text']}"

def fmt_pack(passages):
    return "\n\n".join(fmt_passage(p) for p in passages)

def fmt_transcript(transcript, with_tags=False):
    if not transcript:
        return "(no arguments yet)"
    lines = []
    for t in transcript:
        line = f"{t['side']} (round {t['round']}): {t['text']}"
        if with_tags:
            line += f"  [style tags: {t['tags']}]"
        lines.append(line)
    return "\n".join(lines)

def fmt_claim(i, c):
    """Claim label that cannot be misread as a stance on the claim itself."""
    return f"Claim {i} (said by the {c['speaker']} side): {c['claim']}"

def bullets(items):
    return "\n".join(f"- {x}" for x in items) or "- (none)"

def ai_text(msg):
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")

def strip_preamble(text):
    """Drop leading lines like 'My opening argument:' that leak from the tool loop."""
    lines = text.strip().splitlines()
    while len(lines) > 1 and (not lines[0].strip() or lines[0].strip().endswith(":")):
        lines.pop(0)
    return "\n".join(lines).strip()

# =============================================================== prompts
frame_prompt = ChatPromptTemplate.from_messages([
    ("system", "You turn a user's input into a debate motion for two AI debaters. Every "
               "input is debated, whatever it is: crude, absurd, adult or controversial.\n"
               "Read the whole input as ONE phrase and infer the intent of the whole before "
               "naming any parts; never interpret each word on its own. The web results were "
               "searched for the whole input; use them only to identify who or what each "
               "side is. Follow what they show even if it is adult, explicit or "
               "controversial; never switch to a tamer or different reading, and never "
               "narrow the motion to one article's topic: unless the input names an angle, "
               "the motion is the general overall comparison.\n"
               "For an 'A vs B' input, each side is the whole phrase on its side of 'vs', "
               "one single thing with spelling corrected (for example 'hobit ring' is one "
               "thing, a ring). Every side stays a subject exactly as named; never drop, "
               "merge or replace a side. The motion is about A, the FIRST-named side: FOR "
               "argues for A and AGAINST argues for B. Describe each side as what it "
               "actually is; never force both into one category (a being is not an "
               "artifact).\n"
               "Give the reading in one sentence, name the subjects, and write ONE "
               "declarative proposition that FOR supports. The two positions must be direct "
               "opposites: AGAINST argues the reverse claim outright, never 'equally', "
               "'either ... or', 'as valid as', or that the comparison cannot be assessed."),
    ("human", "Input: {topic}\n\nWeb results for the whole input, searched as one phrase:\n"
              "{context}"),
])

AGENT_SYSTEM = (
    "You are the {side} debater.\nMotion: {proposition}\nYour position: {position}\n\n"
    "You research and think for yourself. You have a WebSearch tool (up to {max_searches} "
    "searches this turn) and your own notebook of passages you found in earlier turns. Use "
    "them to find facts that support your position or rebut your opponent. Whenever you "
    "state a fact from a passage, cite its ID, like [{prefix}3]. You may also make your own "
    "inferences and arguments; present them as reasoning. Never cite an ID for something "
    "the passage does not say. You MUST argue your assigned position every turn; never "
    "decline and never say you cannot argue. Plain prose, no markdown, at most 90 words. "
    "If your opponent has spoken, rebut their latest point first and answer it directly; "
    "never repeat your earlier points. Make every search query specific to this motion's "
    "context, never bare words. Reply with the argument text only, with no preamble "
    "such as 'My opening argument:'.")

claims_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract checkable factual claims from this debate: up to 3 from FOR "
               "and up to 3 from AGAINST. Checkable means specific events, numbers, "
               "named people, works or products, or causal statements. Each claim must be "
               "ONE atomic fact in one sentence; drop contrast clauses such as 'X, not Y' "
               "and keep only X. Skip opinions. For each claim, list the evidence IDs (like "
               "F3 or A2) the speaker attached to it in the transcript; leave the list empty "
               "if the claim was not cited. Most important first within each side."),
    ("human", "Motion: {topic}\n\nTranscript:\n{transcript}"),
])

audit_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a strict fact-checker. Each numbered claim is followed by the "
               "evidence passages its speaker cited for it. Judge each claim ONLY against "
               "its own cited passages. The side label (FOR or AGAINST) only tells you who "
               "said the claim; judge whether the claim itself is true, never whether it "
               "helps a side.\n"
               "verified: a cited passage directly states the claim about the same subject. "
               "Treat well-known former or alternative names of the same organisation or "
               "person (for example MindGeek and Aylo) as the same subject. Do not infer.\n"
               "error: a cited passage directly contradicts the claim, or shows it "
               "misrepresents what an event was. The explanation must give the correction.\n"
               "unsupported: the cited passages do not settle it, including when a passage "
               "was cited for something it does not say.\n"
               "evidence_quote: for verified or error, copy ONE continuous sentence from the "
               "claim's cited passages exactly as written, with no ellipses and no edits. "
               "Quotes are checked automatically, and a quote not found in those passages "
               "downgrades the claim to unsupported. For unsupported, leave it empty."),
    ("human", "{claims}"),
])

decide_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are the debate judge. Two independent fact-checks have been "
               "reconciled into an audit. Decide the winner using only: verified claims "
               "as evidence, factual errors and unsupported claims as penalties, and how "
               "well each side rebutted the other. Contested claims count for neither side. "
               "Each audit item starts with the side that made the claim; a verified claim "
               "helps only that side. Uncited statements are opinion and earn nothing. Use "
               "no facts beyond the audit and transcript. Ignore tone. Style tags are hints "
               "only."),
    ("human", "Motion: {topic}\n\nTranscript:\n{transcript}\n\nAudit:\n{audit}"),
])

# =============================================================== structured output
class Framing(BaseModel):
    reading: str = Field(description="What the whole input means, in one sentence")
    subjects: list[str] = Field(default_factory=list,
                                description="The sides, each exactly as the input names it")
    proposition: str = Field(description="One declarative sentence that FOR supports")
    for_position: str = Field(default="", description="One sentence: what FOR argues")
    against_position: str = Field(default="", description="One sentence: what AGAINST argues")

class WebSearch(BaseModel):
    """Search the web for evidence. Returns numbered passages you can cite by ID."""
    query: str = Field(description="What to search for, in a few words")

class Claim(BaseModel):
    speaker: Literal["FOR", "AGAINST"] = Field(
        description="The side that FIRST made this claim in the transcript")
    claim: str
    citations: list[str] = Field(default_factory=list,
                                 description="Evidence IDs cited for this claim, e.g. ['F3']")

class ClaimList(BaseModel):
    claims: list[Claim]

class ClaimCheck(BaseModel):
    claim_index: int = Field(description="The number of the claim being checked")
    status: Literal["verified", "error", "unsupported"]
    evidence_quote: str = Field(
        description="One exact sentence copied from the cited passages; empty if unsupported")
    explanation: str = Field(description="One sentence. For errors, give the correction")
    source: str = Field(description="Title of the passage the quote comes from")

class Audit(BaseModel):
    checks: list[ClaimCheck]

class Decision(BaseModel):
    # Reasoning comes before winner so the model argues before it decides.
    reasoning: str = Field(description="At most 80 words")
    winner: Literal["FOR", "AGAINST"]

def structured(llm, schema):
    """Structured output with retries: models occasionally drop required fields."""
    return llm.with_structured_output(schema).with_retry(stop_after_attempt=STRUCTURED_RETRIES)

# =============================================================== state
class DebateState(TypedDict):
    topic: str                  # the user's raw input
    intent_query: str           # spelling-corrected input, searched once for intent
    reading: str
    subjects: list[str]
    proposition: str
    for_position: str
    against_position: str
    lineup: dict                # {"FOR": model name, "AGAINST": model name}
    entities: list[dict]        # intent sources: shown, never used as debate evidence
    pack: Annotated[list[dict], operator.add]         # every passage the debaters found
    round: int
    max_rounds: int
    transcript: Annotated[list[dict], operator.add]
    claims: list[dict]
    audit_a: list[dict]
    audit_b: list[dict]
    audit: dict
    verdict: dict

# =============================================================== intent and framing
HEDGE = re.compile(r"\bequally\b|\beither\b|\bcannot be (?:fairly )?assessed\b|"
                   r"\bas (?:good|valid|strong) as\b", re.I)

def spell_fix(topic):
    """One small Haiku call: correct spelling only, keep every word and its order."""
    try:
        fixed = ai_text(extract_llm.invoke(
            "Correct only the spelling of this input. Keep every word, its order and its "
            "meaning; change nothing else. Reply with the corrected text only.\n\n" + topic)).strip()
        return fixed if fixed and len(fixed) <= 2 * len(topic) + 20 else topic
    except Exception:
        return topic

def split_sides(topic):
    """'A vs B' -> ['A', 'B']; None for any other input."""
    parts = re.split(r"\s+(?:vs\.?|versus|v\.?)\s+", topic.strip(), flags=re.I)
    return parts if len(parts) == 2 else None

def _head_pos(side, text):
    """Where the side's head word (its last word) first appears in text, allowing typos."""
    words = re.findall(r"[\w'-]+", side.lower())
    if not words:
        return None
    head = words[-1]
    for m in re.finditer(r"[\w'-]+", text):
        w = m.group(0)
        if (w == head or w.rstrip("s") == head.rstrip("s")
                or difflib.SequenceMatcher(None, w, head).ratio() >= 0.8):
            return m.start()
    return None

def first_side_first(topic, proposition):
    """For 'A vs B': True only if the motion names both sides and names A first."""
    sides = split_sides(topic)
    if not sides:
        return True
    text = proposition.lower()
    a, b = _head_pos(sides[0], text), _head_pos(sides[1], text)
    return a is not None and b is not None and a <= b

def frame_node(state):
    topic = state["topic"]
    query = spell_fix(topic)
    pages, _ = web_search(query)   # one Tavily call: the whole input as one phrase
    context = "\n\n".join(f"- {x['title']}: {x['text'][:500]}" for x in pages) or "(no results)"
    chain = frame_prompt | structured(extract_llm, Framing)
    f = chain.invoke({"topic": topic, "context": context})
    sides = split_sides(topic)
    if sides and not first_side_first(topic, f.proposition):
        f = chain.invoke({"topic": topic, "context": context + (
            f"\n\nCORRECTION: the two sides are exactly A = '{sides[0]}' and B = '{sides[1]}'. "
            "Each side is the whole phrase on its side of 'vs', one single thing, with "
            "spelling corrected. The motion must name both and be about A: FOR argues for A, "
            "AGAINST argues for B.")})
    if HEDGE.search(f.for_position or "") or HEDGE.search(f.against_position or ""):
        f2 = chain.invoke({"topic": topic, "context": context + (
            "\n\nCORRECTION: each position must be one direct, single-sided claim. AGAINST "
            "argues the reverse of the motion outright; never 'equally', 'either ... or', "
            "'as valid as', or that the comparison cannot be assessed.")})
        if first_side_first(topic, f2.proposition):
            f = f2
    prop = f.proposition.strip() or topic
    names = list(DEBATERS)
    random.shuffle(names)   # which model argues FOR is drawn per debate
    return {
        "intent_query": query,
        "reading": f.reading.strip(),
        "subjects": [x.strip() for x in f.subjects if x.strip()],
        "proposition": prop,
        "for_position": f.for_position.strip() or f"Argue that: {prop}",
        "against_position": f.against_position.strip() or f"Argue that this is false: {prop}",
        "lineup": {"FOR": names[0], "AGAINST": names[1]},
    }

def ground_node(state):
    """Intent sources: the framing search results (cached, no extra credit). Shown to
    explain the motion; never used as debate evidence."""
    pages, _ = web_search(state["intent_query"])
    return {"entities": [{"name": "Intent of the whole motion", "source": "web",
                          "title": x["title"], "url": x["url"]} for x in pages[:5]]}

# =============================================================== debaters
def new_passages(pages, side, prefix, known):
    """Chunk search results into citable passages with this side's own IDs."""
    seen, out = {p["text"] for p in known}, []
    for pg in pages:
        for c in chunk(clean_web_text(pg["text"]), pg["title"])[:2]:
            if c["text"] in seen or len(out) >= PASSAGES_PER_SEARCH:
                continue
            seen.add(c["text"])
            out.append({"id": f"{prefix}{len(known) + len(out) + 1}", **c, "url": pg["url"],
                        "source": "Web", "side": side})
    return out

def speak(side, state):
    """A research agent: searches the web itself, keeps its own notebook across turns,
    cites its own passages ([F#] or [A#]) and argues its own inferences."""
    model = state["lineup"][side]
    position = state["for_position"] if side == "FOR" else state["against_position"]
    prefix = side[0]
    notebook = [x for x in state.get("pack", []) if x.get("side") == side]
    found, searches, text = [], 0, ""
    bound = DEBATERS[model].bind_tools([WebSearch])
    msgs = [SystemMessage(AGENT_SYSTEM.format(side=side, proposition=state["proposition"],
                                              position=position, prefix=prefix,
                                              max_searches=MAX_SEARCHES_PER_TURN)),
            HumanMessage("Your notebook so far:\n"
                         + (fmt_pack(notebook) or "(empty: search to gather evidence)")
                         + "\n\nDebate so far:\n" + fmt_transcript(state["transcript"])
                         + "\n\nYour turn.")]
    for _ in range(MAX_SEARCHES_PER_TURN + 2):
        ai = bound.invoke(msgs)
        msgs.append(ai)
        calls = getattr(ai, "tool_calls", None) or []
        if not calls:
            text = ai_text(ai).strip()
            break
        for c in calls:
            if searches < MAX_SEARCHES_PER_TURN:
                searches += 1
                pages, err = web_search(str(c["args"].get("query", "")))
                new = new_passages(pages, side, prefix, notebook + found)
                found.extend(new)
                result = fmt_pack(new) or f"No usable results ({err or 'empty'})."
            else:
                result = "Search limit reached. Write your argument now."
            msgs.append(ToolMessage(content=result, tool_call_id=c["id"]))
    if not text:
        msgs.append(HumanMessage("Write your argument now, without searching."))
        text = ai_text(bound.invoke(msgs)).strip()
    text = strip_preamble(text)
    mine_prev = [t for t in state["transcript"] if t["side"] == side]
    theirs = [t for t in state["transcript"] if t["side"] != side]
    if mine_prev and theirs and text:
        a, b = embedder.encode([text, mine_prev[-1]["text"]], normalize_embeddings=True)
        if float(a @ b) >= REPEAT_THRESHOLD:   # near-copy of its last turn: re-ask once
            msgs.append(HumanMessage(
                "That repeats your previous turn. Write a new argument that directly answers "
                f"your opponent's latest point: \"{theirs[-1]['text']}\" Use new evidence from "
                "your notebook or new reasoning. Argument text only."))
            retry = strip_preamble(ai_text(bound.invoke(msgs)).strip())
            if retry:
                text = retry
    cites = find_citations(text, notebook + found)
    by_id = {x["id"]: x for x in notebook + found}
    return {"side": side, "model": model, "round": state["round"], "text": text,
            "tags": tag(text), "citations": cites, "searches": searches, "sources": found,
            "cited": [by_id[i] for i in cites["valid"]]}

def pro_node(state):
    turn = speak("FOR", state)
    return {"transcript": [turn], "pack": turn["sources"]}

def con_node(state):
    turn = speak("AGAINST", state)
    return {"transcript": [turn], "pack": turn["sources"], "round": state["round"] + 1}

def next_step(state):
    return "pro" if state["round"] <= state["max_rounds"] else "extract"

# =============================================================== fact-check
def trace_claim(claim, transcript, known, speaker):
    """For a claim that came out uncited: find the transcript sentence it came from
    (bge similarity) and take that sentence's side and any citations in it."""
    sents = [(t["side"], x) for t in transcript
             for x in re.split(r"(?<=[.!?])\s+", t["text"]) if x.strip()]
    if not sents:
        return speaker, []
    embs = embedder.encode([x for _, x in sents], normalize_embeddings=True,
                           convert_to_tensor=True)
    q = embedder.encode(claim, normalize_embeddings=True, convert_to_tensor=True)
    sims = embs @ q
    j = int(torch.argmax(sims))
    if float(sims[j]) < 0.6:   # no clear source sentence: leave it as it was
        return speaker, []
    side, sent = sents[j]
    ids = [i for g in CITE_GROUP.findall(sent) for i in CITE_ID.findall(g) if i in known]
    return side, list(dict.fromkeys(ids))

def extract_node(state):
    chain = claims_prompt | structured(extract_llm, ClaimList)
    result = chain.invoke({"topic": state["proposition"],
                           "transcript": fmt_transcript(state["transcript"])})
    known = {p["id"] for p in state["pack"]}
    claims = []
    for c in result.claims[:MAX_CLAIMS]:
        d = c.model_dump()
        d["claim"] = trim_contrast(d["claim"])
        ids = [i.strip().strip("[]") for i in d["citations"]]
        d["citations"] = [i for i in dict.fromkeys(ids) if i in known]
        if not d["citations"]:   # the extractor may drop a citation: trace it back
            d["speaker"], d["citations"] = trace_claim(d["claim"], state["transcript"],
                                                       known, d["speaker"])
        sides = {i[0] for i in d["citations"]}
        if len(sides) == 1:      # F# is FOR's notebook, A# is AGAINST's
            d["speaker"] = "FOR" if sides == {"F"} else "AGAINST"
        claims.append(d)
    return {"claims": claims}

def run_audit(llm, state):
    """One independent check: each cited claim against only the passages it cites."""
    claims = state["claims"]
    by_id = {p["id"]: p for p in state["pack"]}
    results = {i: {"claim_index": i, "status": "unsupported",
                   "note": "No citation, so it counts as opinion.", "source": ""}
               for i, c in enumerate(claims, 1) if not c["citations"]}
    checkable = [(i, c) for i, c in enumerate(claims, 1) if c["citations"]]
    if checkable:
        text = "\n\n".join(
            fmt_claim(i, c) + "\nCITED PASSAGES:\n"
            + "\n".join(fmt_passage(by_id[pid]) for pid in c["citations"])
            for i, c in checkable)
        result = (audit_prompt | structured(llm, Audit)).invoke({"claims": text})
        blocks = {i: normalize(" ".join(by_id[pid]["text"] for pid in c["citations"]))
                  for i, c in checkable}
        for chk in result.checks:
            i = chk.claim_index
            if i in results or i not in blocks:
                continue
            status, note = chk.status, chk.explanation
            if status != "unsupported":
                quote_ = normalize(chk.evidence_quote)
                if not quote_ or quote_ not in blocks[i]:
                    status = "unsupported"
                    note = "Quote not found in the cited passages; downgraded automatically. " + note
            results[i] = {"claim_index": i, "status": status, "note": note,
                          "source": chk.source}
    for i in range(1, len(claims) + 1):
        results.setdefault(i, {"claim_index": i, "status": "unsupported",
                               "note": "Not assessed by this checker.", "source": ""})
    return [results[i] for i in sorted(results)]

def audit_a_node(state):
    return {"audit_a": run_audit(judge_llm, state)}

def audit_b_node(state):
    return {"audit_b": run_audit(checker_llm, state)}

def reconcile_status(a, b):
    if a == b:
        return a
    if "unsupported" in (a, b):
        return "unsupported"
    return "contested"

AUDIT_BUCKETS = {"verified": "verified_claims", "error": "factual_errors",
                 "contested": "contested_claims", "unsupported": "unsupported_claims"}

def reconcile_node(state):
    claims = state["claims"]
    out = {v: [] for v in AUDIT_BUCKETS.values()}
    for a, b in zip(state["audit_a"], state["audit_b"]):
        c = claims[a["claim_index"] - 1]
        status = reconcile_status(a["status"], b["status"])
        if a["status"] == b["status"]:
            note = f"{a['note']} (Both checkers agree.)"
        else:
            verdict = ("Checkers disagree" if status == "contested"
                       else "Checkers split, so treated as unsupported")
            note = (f"{verdict}. Sonnet ({a['status']}): {a['note']} "
                    f"Haiku ({b['status']}): {b['note']}")
        cites = f" [{', '.join(c['citations'])}]" if c["citations"] else ""
        out[AUDIT_BUCKETS[status]].append(
            f"{c['speaker']}: {c['claim'].rstrip('.')}{cites}. {note}")
    return {"audit": out}

def decide_node(state):
    a = state["audit"]
    audit_text = (f"VERIFIED:\n{bullets(a['verified_claims'])}\n\n"
                  f"FACTUAL ERRORS:\n{bullets(a['factual_errors'])}\n\n"
                  f"CONTESTED:\n{bullets(a['contested_claims'])}\n\n"
                  f"UNSUPPORTED:\n{bullets(a['unsupported_claims'])}")
    chain = decide_prompt | structured(judge_llm, Decision)
    d = chain.invoke({"topic": state["proposition"],
                      "transcript": fmt_transcript(state["transcript"], with_tags=True),
                      "audit": audit_text})
    return {"verdict": d.model_dump()}

# =============================================================== graph
def build_graph():
    g = StateGraph(DebateState)
    for name, fn in [("frame", frame_node), ("ground", ground_node),
                     ("pro", pro_node), ("con", con_node), ("extract", extract_node),
                     ("audit_a", audit_a_node), ("audit_b", audit_b_node),
                     ("reconcile", reconcile_node), ("decide", decide_node)]:
        g.add_node(name, fn)
    g.add_edge(START, "frame")
    g.add_edge("frame", "ground")
    g.add_edge("ground", "pro")
    g.add_edge("pro", "con")
    g.add_conditional_edges("con", next_step, ["pro", "extract"])
    g.add_edge("extract", "audit_a")                     # fan-out
    g.add_edge("extract", "audit_b")
    g.add_edge(["audit_a", "audit_b"], "reconcile")      # fan-in
    g.add_edge("reconcile", "decide")
    g.add_edge("decide", END)
    return g.compile(checkpointer=memory.checkpointer())

app = build_graph()

# =============================================================== CLI
def run_cli(topic=DEFAULT_TOPIC, rounds=3):
    inputs = {"topic": topic, "round": 1, "max_rounds": rounds, "transcript": []}
    config = {"configurable": {"thread_id": memory.new_debate(topic, rounds)}}
    for update in app.stream(inputs, config=config, stream_mode="updates"):
        for node, data in update.items():
            memory.record(config["configurable"]["thread_id"], node, data)
            if node == "frame":
                print(f"READING:  {data['reading']}")
                print(f"MOTION:   {data['proposition']}")
                print(f"  FOR ({data['lineup']['FOR']}):         {data['for_position']}")
                print(f"  AGAINST ({data['lineup']['AGAINST']}): {data['against_position']}")
            elif node == "ground":
                print("\nINTENT SOURCES:")
                for e in data["entities"]:
                    print(f"  - {e['title']}  {e['url']}")
                print()
            elif node in ("pro", "con"):
                e = data["transcript"][-1]
                print(f"[{e['side']} r{e['round']} · {e['model']}] {e['text']}")
                print(f"   emotional {e['tags']['emotional']:.2f} · cites "
                      f"{', '.join(e['citations']['valid']) or 'none'} · "
                      f"{e['searches']} searches\n")
            elif node == "extract":
                print("CLAIMS TO CHECK:")
                for c in data["claims"]:
                    print(f"  - {c['speaker']}: {c['claim']}  cites: {c['citations'] or 'none'}")
            elif node in ("audit_a", "audit_b"):
                who = "Sonnet" if node == "audit_a" else "Haiku"
                print(f"\n{who} auditor: " + ", ".join(
                    f"claim {r['claim_index']}={r['status']}" for r in data[node]))
            elif node == "reconcile":
                for key in AUDIT_BUCKETS.values():
                    print(f"\n{key.upper().replace('_', ' ')}:")
                    for x in data["audit"][key]:
                        print(f"  - {x}")
            elif node == "decide":
                v = data["verdict"]
                print("\n" + "=" * 60)
                print(f"WINNER: {v['winner']}")
                print(f"REASONING: {v['reasoning']}")

if __name__ == "__main__":
    run_cli(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TOPIC)
