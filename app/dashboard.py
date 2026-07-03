"""
app/dashboard.py  —  Youtube Comments Intelligence
Run:  python -m streamlit run app/dashboard.py
"""
from __future__ import annotations
import sys, os, json, ast, traceback as _tb
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env", override=False)
except ImportError:
    pass

os.environ.setdefault("RAG_GEN_MODEL", "groq/llama-3.1-8b-instant")
os.environ.setdefault("RAG_EMBED_BACKEND", "sentence-transformers")

import streamlit as st

st.set_page_config(
    page_title="Youtube Comments Intelligence",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme (light / dark) ───────────────────────────────────────────────────────
if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "Light"

_THEMES = {
    "Light": {
        "bg": "#FAFAFA", "surface": "#FFFFFF", "surface2": "#F4F4F5",
        "text": "#111111", "muted": "#71717A", "border": "#E4E4E7",
        "accent": "#111111", "grid": "rgba(0,0,0,0.06)",
        "hover_bg": "#FFFFFF", "hover_border": "#D4D4D8",
        "pie_line": "#FFFFFF", "chart_seq": ["#F4F4F5", "#D4D4D8", "#A1A1AA", "#52525B", "#18181B"],
        "heatmap": ["#F4F4F5", "#18181B"],
        "btn_bg": "#111111", "btn_fg": "#FFFFFF",
        "input_bg": "#FFFFFF", "placeholder": "#A1A1AA",
    },
    "Dark": {
        "bg": "#0A0A0A", "surface": "#141414", "surface2": "#1C1C1C",
        "text": "#F5F5F5", "muted": "#A3A3A3", "border": "#262626",
        "accent": "#F5F5F5", "grid": "rgba(255,255,255,0.06)",
        "hover_bg": "#1C1C1C", "hover_border": "#404040",
        "pie_line": "#0A0A0A", "chart_seq": ["#262626", "#404040", "#525252", "#737373", "#A3A3A3"],
        "heatmap": ["#141414", "#F5F5F5"],
        "btn_bg": "#F5F5F5", "btn_fg": "#0A0A0A",
        "input_bg": "#1C1C1C", "placeholder": "#737373",
    },
}

def _t() -> dict:
    return _THEMES[st.session_state.ui_theme]

# ── Optional imports ───────────────────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY = True
except ImportError:
    PLOTLY = False

try:
    import pandas as pd
    PANDAS = True
except ImportError:
    PANDAS = False

from collections import Counter

# ── Constants ─────────────────────────────────────────────────────────────────
_FONT = ("system-ui,-apple-system,BlinkMacSystemFont,"
         "'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif")

_SENT_COLOR = {
    "positive": "#22C55E",
    "negative": "#EF4444",
    "neutral":  "#A1A1AA",
}

_M  = dict(t=16, b=16, l=16, r=16)


def _ct() -> dict:
    t = _t()
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t["muted"], family=_FONT, size=11),
        hoverlabel=dict(
            bgcolor=t["surface"],
            bordercolor=t["border"],
            font=dict(color=t["text"], family=_FONT, size=12),
        ),
    )


def _ax(title: str | None = None, angle: int | None = None,
        reversed: bool = False, size: int = 11) -> dict:
    t = _t()
    d: dict = dict(
        gridcolor=t["grid"],
        linecolor=t["border"],
        tickfont=dict(size=size, color=t["muted"]),
        zeroline=False,
    )
    if title:
        d["title"] = dict(text=title, font=dict(size=11, color=t["muted"]))
    if angle is not None:
        d["tickangle"] = angle
    if reversed:
        d["autorange"] = "reversed"
    return d


def _chart_seq() -> list:
    return _t()["chart_seq"]


def _colorscale() -> list:
    seq = _chart_seq()
    n = max(len(seq) - 1, 1)
    return [[i / n, c] for i, c in enumerate(seq)]


# ── Data helpers ──────────────────────────────────────────────────────────────
DATA_PATH = os.environ.get(
    "ENRICHED_CSV_PATH",
    os.path.join(os.path.dirname(__file__), "..", "data", "processed", "comments_enriched.csv"),
)

_SENT_MAP = {
    "pos": "positive", "positive": "positive", "1": "positive",
    "neg": "negative", "negative": "negative", "-1": "negative",
    "neu": "neutral",  "neutral":  "neutral",  "0":  "neutral",
}


def _parse_list(val):
    if isinstance(val, list): return val
    if not isinstance(val, str) or not val.strip(): return []
    for fn in (json.loads, ast.literal_eval):
        try: return fn(val)
        except: pass
    return []


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "sentiment" in df.columns:
        df["sentiment"] = (df["sentiment"].astype(str).str.strip().str.lower()
                           .map(lambda v: _SENT_MAP.get(v, "neutral")))
    for col in ("topic_id", "topic_id_sentiment"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
    df["_entities"] = df["entities"].apply(_parse_list) if "entities" in df.columns else [[]]*len(df)
    df["_keywords"] = df["keywords"].apply(_parse_list) if "keywords" in df.columns else [[]]*len(df)
    return df


@st.cache_resource(show_spinner="Building semantic index …")
def _get_orchestrator(path: str):
    from src.agent.orchestrator import Orchestrator
    return Orchestrator(path=path)


def _dot(sentiment: str) -> str:
    return _SENT_COLOR.get(sentiment, _t()["muted"])


def _comment_card(row) -> None:
    sid   = str(row.get("comment_id", ""))
    sent  = str(row.get("sentiment", ""))
    text  = str(row.get("text", ""))
    color = _dot(sent)
    st.markdown(
        f'<div class="ccard">'
        f'<div class="ccard-meta">'
        f'<span class="ccard-dot" style="background:{color};"></span>'
        f'#{sid} &nbsp;·&nbsp; {sent}'
        f'</div>'
        f'<div class="ccard-text">{text}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _ai_response(text: str, kind: str = "answer") -> None:
    is_fallback = text.startswith("⚠")
    base = "ai-answer" if kind == "answer" else "ai-summary"
    css  = f"{base} fallback" if is_fallback else base
    st.markdown(f'<div class="{css}">{text}</div>', unsafe_allow_html=True)


def _intent_pill(intent: str) -> None:
    st.markdown(f'<div class="intent-pill">intent · {intent}</div>', unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────
_csv_path = DATA_PATH
data_ok   = False
df        = None

if PANDAS:
    try:
        df     = load_data(_csv_path)
        data_ok = True
    except FileNotFoundError:
        pass
    except Exception:
        pass

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="nav-brand">Youtube Comments Intelligence</p>', unsafe_allow_html=True)
    st.caption("Ryan Trahan · CSCI370")
    st.divider()

    st.radio(
        "Appearance",
        ["Light", "Dark"],
        horizontal=True,
        key="ui_theme",
    )

    st.divider()
    st.markdown("**LLM**")
    try:
        from src.rag.generator import llm_status
        _llm = llm_status()
        if _llm["ok"]:
            st.success(_llm["message"])
            st.caption(_llm["model"])
        else:
            st.error(_llm["message"])
            groq_in = st.text_input("Groq API key", type="password", key="groq_key_input")
            if groq_in.strip() and st.button("Save key to .env", key="save_groq"):
                env_path = _ROOT / ".env"
                lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
                lines = [ln for ln in lines if not ln.startswith("GROQ_API_KEY=")]
                lines.append(f"GROQ_API_KEY={groq_in.strip()}")
                env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                os.environ["GROQ_API_KEY"] = groq_in.strip()
                st.success("Saved. Refresh the page (R).")
    except Exception as _e:
        st.caption(f"LLM status unavailable: {_e}")

    st.divider()
    st.markdown("**Retrieval**")
    top_k     = st.slider("Top-k results", 1, 10, 8, key="sidebar_k")
    retrieval = st.selectbox(
        "Strategy",
        ["hybrid", "mmr", "similarity", "bm25", "hyde"],
        key="sidebar_strat",
    )

    if data_ok:
        st.divider()
        st.caption(f"{len(df):,} comments loaded")

    st.divider()
    custom_path = st.text_input("Data path", value=_csv_path, key="csv_path_input")
    if custom_path != _csv_path:
        try:
            df = load_data(custom_path)
            _csv_path = custom_path
            data_ok   = True
            st.success("Loaded.")
        except Exception as e:
            st.error(str(e))

def _inject_css() -> None:
    t0 = _t()
    st.markdown(f"""
<style>
:root {{
  --bg: {t0['bg']};
  --surface: {t0['surface']};
  --surface2: {t0['surface2']};
  --text: {t0['text']};
  --muted: {t0['muted']};
  --border: {t0['border']};
  --accent: {t0['accent']};
  --grid: {t0['grid']};
  --btn-bg: {t0['btn_bg']};
  --btn-fg: {t0['btn_fg']};
  --input-bg: {t0['input_bg']};
  --placeholder: {t0['placeholder']};
  --radius: 24px;
  --radius-sm: 14px;
  --radius-pill: 999px;
  /* Streamlit native theme tokens */
  --background-color: {t0['bg']};
  --secondary-background-color: {t0['surface']};
  --text-color: {t0['text']};
  --primary-color: {t0['accent']};
}}

/* ── Streamlit app shell (light + dark) ── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section.main,
[data-testid="stAppViewContainer"] .block-container {{
  background-color: var(--bg) !important;
  color: var(--text) !important;
}}
[data-testid="stHeader"] {{
  background-color: var(--bg) !important;
  border-bottom: 1px solid var(--border);
}}
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {{
  background-color: var(--surface) !important;
  border-right: 1px solid var(--border);
}}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
  color: var(--text) !important;
}}
[data-testid="stSidebar"] hr {{
  border-color: var(--border) !important;
}}

/* Widget labels (sidebar + main) */
label[data-testid="stWidgetLabel"],
.stTextInput label, .stTextArea label, .stSelectbox label,
.stSlider label, .stRadio label, .stNumberInput label {{
  color: var(--text) !important;
}}
.stRadio label span,
.stRadio [data-baseweb="radio"] label,
.stRadio [data-baseweb="radio"] div,
.stRadio [data-baseweb="radio"] span {{
  color: var(--text) !important;
}}
.stRadio [data-baseweb="radio"] svg {{
  fill: var(--text) !important;
}}

/* Main content typography */
.block-container h1, .block-container h2, .block-container h3,
.block-container h4, .block-container h5, .block-container h6,
.block-container p, .block-container label,
.block-container [data-testid="stMarkdownContainer"] {{
  color: var(--text) !important;
}}
.block-container .stCaption,
.block-container [data-testid="stMarkdownContainer"] p {{
  color: var(--muted) !important;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 8px;
}}
.stTabs [data-baseweb="tab"] {{
  color: var(--muted) !important;
  background: transparent !important;
  border-radius: var(--radius-pill) !important;
  padding: 8px 16px !important;
}}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
  color: var(--text) !important;
  background: var(--surface2) !important;
  border-bottom: none !important;
}}
.stTabs [data-baseweb="tab-highlight"] {{
  background: transparent !important;
}}

/* Inputs & widgets */
.stTextInput input,
.stTextInput textarea,
.stNumberInput input,
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {{
  background-color: var(--input-bg) !important;
  color: var(--text) !important;
  border-color: var(--border) !important;
  border-radius: var(--radius-sm) !important;
  -webkit-text-fill-color: var(--text) !important;
}}
.stTextInput input::placeholder,
.stTextInput textarea::placeholder {{
  color: var(--placeholder) !important;
  opacity: 1 !important;
  -webkit-text-fill-color: var(--placeholder) !important;
}}
div[data-baseweb="select"] span,
div[data-baseweb="select"] div {{
  color: var(--text) !important;
}}
.stSlider [data-baseweb="slider"] div,
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"],
.stSlider [data-testid="stMarkdownContainer"] p {{
  color: var(--text) !important;
}}
.stSlider [data-baseweb="slider"] [role="slider"] {{
  background: var(--text) !important;
}}

/* Buttons — override Streamlit primary white-on-white in dark mode */
.stApp div[data-testid="stButton"] > button {{
  border-radius: var(--radius-pill) !important;
  font-weight: 600 !important;
  letter-spacing: 0.01em !important;
  padding: 0.55rem 1.4rem !important;
}}
.stApp div[data-testid="stButton"] > button[kind="primary"],
.stApp div[data-testid="stButton"] > button[data-testid="baseButton-primary"] {{
  background-color: var(--btn-bg) !important;
  color: var(--btn-fg) !important;
  border: 1px solid var(--btn-bg) !important;
}}
.stApp div[data-testid="stButton"] > button[kind="primary"] *,
.stApp div[data-testid="stButton"] > button[data-testid="baseButton-primary"] * {{
  color: var(--btn-fg) !important;
  fill: var(--btn-fg) !important;
}}
.stApp div[data-testid="stButton"] > button[kind="secondary"],
.stApp div[data-testid="stButton"] > button[data-testid="baseButton-secondary"] {{
  background-color: var(--surface) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
}}
.stApp div[data-testid="stButton"] > button[kind="secondary"] * {{
  color: var(--text) !important;
}}

/* Alerts */
[data-testid="stAlert"] {{
  border-radius: var(--radius-sm) !important;
}}
[data-testid="stAlert"] p, [data-testid="stAlert"] div {{
  color: var(--text) !important;
}}

/* Metrics, expanders, dataframe */
[data-testid="stMetric"] {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 12px 16px;
}}
[data-testid="stMetricLabel"] {{ color: var(--muted) !important; }}
[data-testid="stMetricValue"] {{ color: var(--text) !important; }}
[data-testid="stExpander"] {{
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
}}
[data-testid="stExpander"] summary {{
  color: var(--text) !important;
}}
[data-testid="stDataFrame"] {{
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
}}

div[data-testid="column"] {{
  display: flex !important;
  flex-direction: column !important;
  gap: 0 !important;
}}
div[data-testid="stPlotlyChart"] {{
  flex: 1 1 auto;
  min-height: 280px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 8px 8px 0;
  margin-bottom: 12px;
}}

.top-nav {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 4px 22px;
  margin-bottom: 8px;
  border-bottom: 1px solid var(--border);
}}
.nav-brand {{
  font-size: 0.95rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  color: var(--text);
}}
.nav-meta {{
  font-size: 0.72rem;
  color: var(--muted);
  letter-spacing: 0.04em;
}}

.hero {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 40px 44px 36px;
  margin-bottom: 28px;
}}
.hero-eyebrow {{
  font-size: 0.62rem;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 12px;
}}
.hero-title {{
  font-size: clamp(1.6rem, 3vw, 2.2rem);
  font-weight: 700;
  letter-spacing: -0.04em;
  color: var(--text);
  margin: 0 0 10px;
  line-height: 1.12;
  max-width: 16ch;
}}
.hero-sub {{
  font-size: 0.88rem;
  color: var(--muted);
  margin: 0 0 20px;
  max-width: 52ch;
  line-height: 1.6;
}}
.hero-pills {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.hero-pill {{
  font-size: 0.68rem;
  color: var(--text);
  background: var(--surface2);
  border: 1px solid var(--border);
  padding: 6px 14px;
  border-radius: var(--radius-pill);
  letter-spacing: 0.02em;
}}

.sec {{
  font-size: 0.62rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 28px 0 14px;
}}

.kpi-grid {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
  margin-bottom: 8px;
}}
@media (max-width: 900px) {{
  .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}
.kpi {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 22px 24px 18px;
}}
.kpi-label {{
  font-size: 0.6rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 10px;
}}
.kpi-value {{
  font-size: 2rem;
  font-weight: 700;
  letter-spacing: -0.04em;
  color: var(--text);
  line-height: 1;
}}
.kpi-sub {{ font-size: 0.72rem; color: var(--muted); margin-top: 8px; }}

.ai-answer, .ai-summary {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 20px 24px;
  font-size: 0.9rem;
  line-height: 1.75;
  color: var(--text);
  white-space: pre-line;
  margin: 12px 0 6px;
}}
.ai-answer.fallback, .ai-summary.fallback {{
  border-color: var(--border);
  color: var(--muted);
}}

.intent-pill {{
  display: inline-block;
  background: var(--surface2);
  border: 1px solid var(--border);
  color: var(--muted);
  font-size: 0.6rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  padding: 5px 12px;
  border-radius: var(--radius-pill);
  margin: 4px 0 14px;
}}

.ccard {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 14px 18px;
  margin-bottom: 10px;
}}
.ccard-meta {{
  font-size: 0.58rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  gap: 8px;
}}
.ccard-dot {{ width: 6px; height: 6px; border-radius: 99px; flex-shrink: 0; }}
.ccard-text {{ font-size: 0.86rem; color: var(--text); line-height: 1.65; opacity: 0.88; }}

.strat-title {{
  font-size: 0.65rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  padding-bottom: 10px;
  margin-bottom: 12px;
  border-bottom: 1px solid var(--border);
  color: var(--text);
}}
</style>
""", unsafe_allow_html=True)

_inject_css()

# ── Top nav + hero ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-nav">
  <div class="nav-brand">Youtube Comments Intelligence</div>
  <div class="nav-meta">CSCI370 · Spring 2026</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">Youtube Comments Intelligence</div>
  <div class="hero-title">Youtuber audience analytics</div>
  <div class="hero-sub">Sentiment, topics, entities, and source-grounded answers from Ryan Trahan comment data.</div>
  <div class="hero-pills">
    <span class="hero-pill">Sentiment</span>
    <span class="hero-pill">Topics</span>
    <span class="hero-pill">NER</span>
    <span class="hero-pill">RAG</span>
    <span class="hero-pill">Evaluation</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_analytics, tab_qa, tab_summary, tab_lab = st.tabs([
    "Analytics",
    "Ask",
    "Summarize",
    "Retrieval Lab",
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1  —  ANALYTICS
# ════════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    if not data_ok:
        st.info("No data loaded — run `python src/pipeline/run_full_pipeline.py` first.")
        st.stop()

    n   = len(df)
    svc = df["sentiment"].value_counts() if "sentiment" in df.columns else pd.Series(dtype=int)
    pos = int(svc.get("positive", 0))
    neg = int(svc.get("negative", 0))
    neu = int(svc.get("neutral",  0))

    # Filter out BERTopic outlier topic (-1)
    if "topic_id" in df.columns:
        dft = df[df["topic_id"] != -1].copy()
    elif "topic_label" in df.columns:
        lbl = df["topic_label"].astype(str).str.lower()
        dft = df[~lbl.str.startswith("-1") & ~lbl.str.contains("outlier")].copy()
    else:
        dft = df.copy()
    if dft.empty:
        dft = df.copy()

    # ── KPI row ──
    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi">
        <div class="kpi-label">Total Comments</div>
        <div class="kpi-value">{n:,}</div>
        <div class="kpi-sub">Ryan Trahan dataset</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Positive</div>
        <div class="kpi-value">{pos:,}</div>
        <div class="kpi-sub">{pos/n:.1%} of total</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Negative</div>
        <div class="kpi-value">{neg:,}</div>
        <div class="kpi-sub">{neg/n:.1%} of total</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Neutral</div>
        <div class="kpi-value">{neu:,}</div>
        <div class="kpi-sub">{neu/n:.1%} of total</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not PLOTLY:
        st.warning("Install plotly (`pip install plotly`) to see charts.")
    else:
        # ── Row 1: Sentiment donut  +  Top topics ──
        st.markdown('<div class="sec">Sentiment &amp; Topics</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        with c1:
            t = _t()
            fig = go.Figure(go.Pie(
                labels=svc.index.tolist(),
                values=svc.values.tolist(),
                hole=0.68,
                marker=dict(
                    colors=[_SENT_COLOR.get(s, t["muted"]) for s in svc.index],
                    line=dict(color=t["pie_line"], width=4),
                ),
                textinfo="label+percent",
                textfont=dict(size=12, color=t["muted"]),
                hovertemplate="%{label}: %{value:,}<extra></extra>",
            ))
            fig.update_layout(
                **_ct(),
                height=270,
                margin=_M,
                showlegend=False,
                annotations=[dict(
                    text=f"<b>{n:,}</b>",
                    x=0.5, y=0.5, font=dict(size=18, color=t["text"]),
                    showarrow=False,
                )],
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            if "topic_label" in dft.columns:
                tc = dft["topic_label"].value_counts().head(8)
                if not tc.empty:
                    fig = go.Figure(go.Bar(
                        x=tc.values.tolist(),
                        y=tc.index.tolist(),
                        orientation="h",
                        marker=dict(
                            color=tc.values.tolist(),
                            colorscale=_colorscale(),
                            line=dict(width=0),
                        ),
                        hovertemplate="%{y}: %{x:,}<extra></extra>",
                    ))
                    fig.update_layout(
                        **_ct(), height=270, margin=_M,
                        xaxis=_ax("Comments"),
                        yaxis=_ax(reversed=True),
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No topic data.")
            else:
                st.info("`topic_label` column not found.")

        # ── Row 2: Topics × Sentiment  +  Entities ──
        st.markdown('<div class="sec">Topics × Sentiment &amp; Entities</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)

        with c3:
            if "topic_label" in dft.columns and "sentiment" in dft.columns:
                cross = (dft.groupby(["topic_label", "sentiment"])
                           .size().reset_index(name="count"))
                fig = px.bar(
                    cross, x="topic_label", y="count", color="sentiment",
                    color_discrete_map=_SENT_COLOR, barmode="stack",
                    labels={"topic_label": "", "count": "Comments", "sentiment": ""},
                )
                fig.update_traces(marker_line_width=0)
                fig.update_layout(
                    **_ct(), height=310,
                    margin=dict(t=16, b=70, l=16, r=16),
                    xaxis=_ax(angle=-35, size=10),
                    yaxis=_ax("Comments"),
                    legend=dict(
                        orientation="h", y=1.08, x=0,
                        bgcolor="rgba(0,0,0,0)",
                        font=dict(size=11, color=_t()["muted"]),
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need `topic_label` and `sentiment` columns.")

        with c4:
            ents = Counter(
                e["text"] for row in df["_entities"]
                for e in row
                if isinstance(e, dict) and e.get("text")
            ).most_common(10)
            if ents:
                el, ev = zip(*ents)
                fig = go.Figure(go.Bar(
                    x=list(ev), y=list(el), orientation="h",
                    marker=dict(
                        color=list(ev),
                        colorscale=_colorscale(),
                        line=dict(width=0),
                    ),
                    hovertemplate="%{y}: %{x:,}<extra></extra>",
                ))
                fig.update_layout(
                    **_ct(), height=310, margin=_M,
                    xaxis=_ax("Mentions"),
                    yaxis=_ax(reversed=True),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No entity data — run NER pipeline.")

        # ── Row 3: Keywords  +  Sentiment-specific topics ──
        st.markdown('<div class="sec">Keywords &amp; Sentiment-Specific Topics</div>', unsafe_allow_html=True)
        c5, c6 = st.columns(2)

        with c5:
            kws = Counter(
                str(kw) for row in df["_keywords"] for kw in row
            ).most_common(12)
            if kws:
                kl, kv = zip(*kws)
                fig = go.Figure(go.Bar(
                    x=list(kv), y=list(kl), orientation="h",
                    marker=dict(
                        color=list(kv),
                        colorscale=_colorscale(),
                        line=dict(width=0),
                    ),
                    hovertemplate="%{y}: %{x:,}<extra></extra>",
                ))
                fig.update_layout(
                    **_ct(), height=330, margin=_M,
                    xaxis=_ax("Frequency"),
                    yaxis=_ax(reversed=True),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No keyword data — run keyword extraction pipeline.")

        with c6:
            if "topic_label_sentiment" in dft.columns:
                sl  = dft["topic_label_sentiment"].astype(str).str.lower()
                ds  = dft[~sl.str.startswith("-1") & ~sl.str.contains("outlier")]
                if ds.empty and "topic_label" in dft.columns:
                    # Per-sentiment BERTopic often marks short YouTube comments as
                    # outliers (-1). Fall back to overall topics split by sentiment.
                    tl = dft["topic_label"].astype(str).str.lower()
                    ds = dft[~tl.str.startswith("-1") & ~tl.str.contains("outlier")]
                    topic_col = "topic_label"
                    st.caption("Per-sentiment topics were mostly outliers — showing overall topics by sentiment instead.")
                else:
                    topic_col = "topic_label_sentiment"

                if not ds.empty:
                    stc = (ds.groupby(["sentiment", topic_col])
                             .size().reset_index(name="count"))
                    fig = px.bar(
                        stc, x=topic_col, y="count",
                        color="sentiment", color_discrete_map=_SENT_COLOR,
                        barmode="group",
                        labels={topic_col: "", "count": "Comments", "sentiment": ""},
                    )
                    fig.update_traces(marker_line_width=0)
                    fig.update_layout(
                        **_ct(), height=330,
                        margin=dict(t=16, b=70, l=16, r=16),
                        xaxis=_ax(angle=-35, size=10),
                        yaxis=_ax("Comments"),
                        legend=dict(
                            orientation="h", y=1.08, x=0,
                            bgcolor="rgba(0,0,0,0)",
                            font=dict(size=11, color=_t()["muted"]),
                        ),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No topic data available — re-run BERTopic with src/pipeline/run_full_pipeline.py")
            else:
                st.info("`topic_label_sentiment` column not found.")

    # ── Raw data table ──
    st.divider()
    with st.expander("Raw data table"):
        show_cols = [c for c in ["comment_id", "text", "sentiment", "topic_label", "keywords"] if c in df.columns]
        st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

    # ── Evaluation ──
    st.divider()
    st.markdown('<div class="sec">MLflow Evaluation</div>', unsafe_allow_html=True)
    st.caption(
        "Fast mode skips BERTScore (~1-3 min). Enable BERTScore for full generation "
        "metrics your professor expects (~3-8 min on CPU, first run downloads roberta-base)."
    )

    eval_col1, eval_col2 = st.columns(2)
    with eval_col1:
        include_bert = st.checkbox("Include BERTScore", value=False, key="eval_bertscore")
    with eval_col2:
        bert_ref_mode = st.selectbox(
            "BERTScore reference",
            ["both", "context", "gold"],
            format_func=lambda x: {
                "both": "Both (context + gold)",
                "context": "Retrieved comments (recommended)",
                "gold": "Gold reference answers",
            }[x],
            key="eval_bert_ref",
            disabled=not include_bert,
        )

    if st.button("Run evaluation", type="primary", key="eval_btn"):
        spinner_msg = (
            "Running full evaluation (retrieval + Groq answers + BERTScore) …"
            if include_bert else
            "Running evaluation (retrieval + Groq answers + metrics) …"
        )
        with st.spinner(spinner_msg):
            try:
                from src.evaluation.run_eval import run_full_evaluation
                results = run_full_evaluation(
                    path=_csv_path,
                    use_mock=not data_ok,
                    fast=not include_bert,
                    include_bertscore=include_bert,
                    bert_reference_mode=bert_ref_mode,
                )
                st.session_state["eval_results"] = results
                st.success("Evaluation complete.")
            except Exception as e:
                st.error(str(e))
                with st.expander("Traceback"):
                    st.code(_tb.format_exc())

    if "eval_results" in st.session_state:
        res = st.session_state["eval_results"]
        st.markdown(f"**Run label:** `{res.get('label', 'eval')}`")
        if res.get("include_bertscore"):
            st.caption(
                f"BERTScore mode: **{res.get('bert_reference_mode', 'both')}** | "
                f"model: **distilbert-base-uncased**"
            )
            if res.get("bertscore_used_fallback"):
                st.warning(
                    "BERTScore package/model failed — showing **semantic similarity fallback** "
                    "(MiniLM cosine). Install/fix with: `pip install bert-score`"
                )
                if res.get("bertscore_error"):
                    st.caption(f"Details: {res['bertscore_error']}")
            elif res.get("generation", {}).get("bert_f1", 0) == 0:
                st.warning(
                    "BERTScore ran but returned 0. Run in terminal: "
                    "`pip install bert-score` then re-run evaluation."
                )
        else:
            st.caption("BERTScore was skipped (fast mode). Re-run with **Include BERTScore** checked.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Retrieval metrics**")
            for k, v in sorted(res.get("retrieval", {}).items()):
                st.metric(k.replace("_", " ").title(), f"{v:.3f}")
        with c2:
            st.markdown("**Generation metrics**")
            for k, v in sorted(res.get("generation", {}).items()):
                label = k.replace("_", " ").title()
                if k == "bert_f1":
                    label = "BERTScore F1 (context)"
                elif k == "bert_f1_gold":
                    label = "BERTScore F1 (gold)"
                st.metric(label, f"{v:.3f}")

        per_query = res.get("per_query", [])
        if per_query:
            with st.expander("Per-query breakdown (for report / professor)"):
                import pandas as pd
                pq_df = pd.DataFrame(per_query)
                show_cols = [c for c in pq_df.columns if c != "hypothesis"]
                st.dataframe(pq_df[show_cols], use_container_width=True, hide_index=True)

        with st.expander("Sample Q/A from evaluation"):
            for q, a in zip(res.get("queries", []), res.get("hypotheses", [])):
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}")
                st.divider()


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2  —  ASK AI
# ════════════════════════════════════════════════════════════════════════════════
with tab_qa:
    st.markdown("#### Ask a question about the comments")
    st.caption("Intent classifier -> hybrid/MMR retrieval -> DSPy answer generation")

    q_input = st.text_input(
        "Question",
        placeholder="e.g. What do viewers think about Ryan's storytelling?",
        key="qa_input",
    )

    if st.button("Ask", type="primary", key="qa_submit"):
        if not q_input.strip():
            st.warning("Please enter a question.")
        elif not data_ok:
            st.error("No data loaded. Check the CSV path in the sidebar.")
        else:
            with st.spinner("Retrieving and generating …"):
                try:
                    orch = _get_orchestrator(_csv_path)
                    resp = orch.handle(q_input, k=top_k, retrieval=retrieval)

                    _ai_response(resp["answer"], kind="answer")
                    _intent_pill(resp["intent"])

                    if resp.get("citations"):
                        cited = df[df["comment_id"].isin(resp["citations"])]
                        with st.expander(f"{len(cited)} source comments", expanded=True):
                            for _, row in cited.iterrows():
                                _comment_card(row)

                        try:
                            from src.evaluation.generation_eval import faithfulness_score
                            ctx = " ".join(cited["text"].astype(str).tolist())
                            score = faithfulness_score(resp["answer"], ctx)
                            st.metric("Answer faithfulness", f"{score:.1%}")
                            st.caption("How well the answer matches retrieved comments (wording + meaning).")
                        except Exception:
                            pass

                except Exception as e:
                    st.error(f"Error: {e}")
                    with st.expander("Traceback"):
                        st.code(_tb.format_exc())


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3  —  SUMMARIZE
# ════════════════════════════════════════════════════════════════════════════════
with tab_summary:
    st.markdown("#### Summarize viewer opinion on a topic")
    st.caption("Retrieves top-k comments → DSPy getSummary")

    col_a, col_b = st.columns([5, 1])
    with col_a:
        s_input = st.text_input(
            "Topic",
            placeholder="e.g. What do viewers say about Ryan's travel vlogs?",
            key="sum_input",
        )
    with col_b:
        sum_k = st.slider("k", 2, 12, 6, key="sum_k")

    if st.button("Summarize", type="primary", key="sum_submit"):
        if not s_input.strip():
            st.warning("Please enter a topic or question.")
        elif not data_ok:
            st.error("No data loaded.")
        else:
            with st.spinner("Retrieving and summarising …"):
                try:
                    orch = _get_orchestrator(_csv_path)
                    resp = orch.handle(s_input, k=sum_k)

                    _ai_response(resp["answer"], kind="summary")
                    _intent_pill(resp["intent"])

                    if resp.get("citations"):
                        cited = df[df["comment_id"].isin(resp["citations"])]
                        with st.expander(f"{len(cited)} retrieved comments"):
                            for _, row in cited.iterrows():
                                _comment_card(row)

                        try:
                            from src.evaluation.generation_eval import faithfulness_score
                            ctx = " ".join(cited["text"].astype(str).tolist())
                            score = faithfulness_score(resp["answer"], ctx)
                            st.metric("Summary faithfulness", f"{score:.1%}")
                            st.caption("How well the summary matches retrieved comments (wording + meaning).")
                        except Exception:
                            pass

                except Exception as e:
                    st.error(f"Error: {e}")
                    with st.expander("Traceback"):
                        st.code(_tb.format_exc())


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4  —  RETRIEVAL LAB
# ════════════════════════════════════════════════════════════════════════════════
with tab_lab:
    st.markdown("#### Retrieval Strategy Comparison")
    st.caption("Compare hybrid (BM25+FAISS RRF), MMR, BM25, similarity, and HyDE side-by-side")

    lab_q = st.text_input(
        "Query",
        value="What do viewers say about Ryan Trahan's storytelling?",
        key="lab_input",
    )

    if st.button("Compare strategies", type="primary", key="lab_submit"):
        if not data_ok:
            st.error("No data loaded.")
        else:
            with st.spinner("Running retrieval strategies …"):
                try:
                    from src.rag.generator import get_vector_store, _search

                    store = get_vector_store(_csv_path)
                    strategies = {
                        "Hybrid":     "hybrid",
                        "BM25":       "bm25",
                        "MMR":        "mmr",
                        "Similarity": "similarity",
                        "HyDE":       "hyde",
                    }

                    results: dict[str, list] = {}
                    for name, mode in strategies.items():
                        results[name] = _search(store, mode, lab_q, k=top_k)

                    # Side-by-side columns
                    st.markdown('<div class="sec">Results</div>', unsafe_allow_html=True)
                    cols = st.columns(5)
                    for col, name in zip(cols, strategies):
                        with col:
                            st.markdown(
                                f'<div class="strat-title">{name}</div>',
                                unsafe_allow_html=True,
                            )
                            for doc in results[name]:
                                sent  = doc.get("sentiment", "")
                                score = doc.get("score", 0)
                                st.markdown(
                                    f'<div class="ccard">'
                                    f'<div class="ccard-meta">'
                                    f'<span class="ccard-dot" style="background:{_dot(sent)};"></span>'
                                    f'#{doc["comment_id"]} &nbsp;·&nbsp; {score:.3f}'
                                    f'</div>'
                                    f'<div class="ccard-text">{doc["text"]}</div>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

                    # Overlap heatmap
                    if PLOTLY:
                        st.markdown('<div class="sec">Result Overlap</div>', unsafe_allow_html=True)
                        id_sets = {n: {d["comment_id"] for d in docs} for n, docs in results.items()}
                        all_ids = sorted(set().union(*id_sets.values()))
                        if all_ids:
                            ov = pd.DataFrame(
                                {n: [cid in ids for cid in all_ids] for n, ids in id_sets.items()},
                                index=[f"#{i}" for i in all_ids],
                            )
                            fig = px.imshow(
                                ov.astype(int).T,
                                color_continuous_scale=_t()["heatmap"],
                                aspect="auto",
                                labels={"x": "Comment", "y": "Strategy", "color": "Retrieved"},
                            )
                            fig.update_layout(
                                **_ct(), height=140, margin=_M,
                                coloraxis_showscale=False,
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption("Dark = retrieved · shared columns = consensus between strategies")

                    # Per-strategy faithfulness
                    st.markdown('<div class="sec">Query Coverage</div>', unsafe_allow_html=True)
                    try:
                        from src.evaluation.generation_eval import faithfulness_score
                        mc = st.columns(5)
                        for col, (name, docs) in zip(mc, results.items()):
                            if docs:
                                ctx   = " ".join(d["text"] for d in docs)
                                score = faithfulness_score(lab_q, ctx)
                                col.metric(name, f"{score:.1%}")
                    except Exception:
                        pass

                except Exception as e:
                    st.error(f"Error: {e}")
                    with st.expander("Traceback"):
                        st.code(_tb.format_exc())
