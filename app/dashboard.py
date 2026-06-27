"""
app/dashboard.py  —  Ryan Trahan YouTube Intelligence Engine
Run:  python -m streamlit run app/dashboard.py
"""
from __future__ import annotations
import sys, os, json, ast, traceback as _tb
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

st.set_page_config(
    page_title="YT Intelligence · Ryan Trahan",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Scoped CSS — never touches Streamlit internals ────────────────────────────
st.markdown("""
<style>
/* ── Base font ── */
html, body, [class*="css"] {
    font-family: system-ui, -apple-system, BlinkMacSystemFont,
                 "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #13141a 0%, #0a0b0e 50%, #15102a 100%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 30px 36px 26px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    top: -100px; right: -80px;
    width: 360px; height: 360px;
    background: radial-gradient(circle, rgba(124,58,237,.1) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-size: 0.6rem; font-weight: 700;
    letter-spacing: .14em; text-transform: uppercase;
    color: #7C3AED; margin-bottom: 10px;
}
.hero-title {
    font-size: 1.75rem; font-weight: 700; letter-spacing: -0.035em;
    color: #f0f0f5; margin: 0 0 6px; line-height: 1.15;
}
.hero-sub { font-size: 0.82rem; color: #44474f; margin: 0; }
.hero-tags { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 18px; }
.hero-tag {
    font-size: 0.66rem; color: #44474f;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    padding: 3px 10px; border-radius: 6px; letter-spacing: .03em;
}

/* ── Section label ── */
.sec {
    font-size: 0.58rem; font-weight: 700;
    letter-spacing: .13em; text-transform: uppercase;
    color: #44474f; margin: 24px 0 12px;
    display: flex; align-items: center; gap: 10px;
}
.sec::after {
    content: ''; flex: 1; height: 1px;
    background: rgba(255,255,255,0.05);
}

/* ── KPI grid ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-bottom: 4px; }
.kpi {
    background: #111318;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 20px 22px 16px;
}
.kpi-accent { height: 2px; border-radius: 2px; margin-bottom: 16px; }
.kpi-label {
    font-size: 0.58rem; font-weight: 700;
    letter-spacing: .13em; text-transform: uppercase;
    color: #44474f; margin-bottom: 8px;
}
.kpi-value {
    font-size: 2.1rem; font-weight: 700; letter-spacing: -0.045em;
    color: #f0f0f5; line-height: 1;
}
.kpi-sub { font-size: 0.7rem; color: #2c2f38; margin-top: 6px; }

/* ── AI response cards ── */
.ai-answer {
    background: #0e140e;
    border: 1px solid rgba(50,215,75,.12);
    border-left: 3px solid #30d158;
    border-radius: 12px;
    padding: 18px 22px;
    font-size: 0.88rem; line-height: 1.78;
    color: #a8c8ac; white-space: pre-line;
    margin: 10px 0 4px;
}
.ai-answer.fallback {
    background: #14110a;
    border-left-color: #ff9f0a;
    color: #c0a870;
}
.ai-summary {
    background: #0d1018;
    border: 1px solid rgba(94,130,255,.12);
    border-left: 3px solid #5e82ff;
    border-radius: 12px;
    padding: 18px 22px;
    font-size: 0.88rem; line-height: 1.78;
    color: #a0b0d0; white-space: pre-line;
    margin: 10px 0 4px;
}
.ai-summary.fallback {
    background: #14110a;
    border-left-color: #ff9f0a;
    color: #c0a870;
}

/* ── Intent pill ── */
.intent-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(124,58,237,.08);
    border: 1px solid rgba(124,58,237,.18);
    color: #7C3AED; font-size: 0.6rem; font-weight: 700;
    letter-spacing: .1em; text-transform: uppercase;
    padding: 3px 10px; border-radius: 99px; margin: 4px 0 14px;
}

/* ── Comment card ── */
.ccard {
    background: #111318;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 12px 16px; margin-bottom: 8px;
}
.ccard-meta {
    font-size: 0.58rem; font-weight: 700;
    letter-spacing: .1em; text-transform: uppercase;
    color: #2e3240; margin-bottom: 6px;
    display: flex; align-items: center; gap: 8px;
}
.ccard-dot { width: 6px; height: 6px; border-radius: 99px; flex-shrink: 0; }
.ccard-text { font-size: 0.84rem; color: #7a8090; line-height: 1.65; }

/* ── Retrieval lab strategy header ── */
.strat-title {
    font-size: 0.65rem; font-weight: 700;
    letter-spacing: .1em; text-transform: uppercase;
    padding-bottom: 10px; margin-bottom: 12px;
    border-bottom: 2px solid currentColor;
}
</style>
""", unsafe_allow_html=True)

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
    "positive": "#30d158",
    "negative": "#ff453a",
    "neutral":  "#636366",
}

# Base Plotly layout applied to every chart
# NOTE: margin is intentionally excluded — each chart sets its own to avoid
# duplicate-keyword errors when charts also pass margin explicitly.
_CT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(color="#44474f", family=_FONT, size=11),
    hoverlabel=dict(
        bgcolor="#1a1c24",
        bordercolor="rgba(255,255,255,0.08)",
        font=dict(color="#e8e8f0", family=_FONT, size=12),
    ),
)
_M  = dict(t=16, b=16, l=16, r=16)   # default margin
_MB = dict(t=16, b=70, l=16, r=16)   # margin with room for angled x-axis labels

# Purple sequential for single-variable charts
_PURPLE_SEQ = ["#1e1040", "#3b1a7a", "#5e2fbf", "#7C3AED", "#a78bfa"]
_BLUE_SEQ   = ["#0a1030", "#1a3070", "#2a50c0", "#5e82ff", "#99b4ff"]
_GREEN_SEQ  = ["#061a0a", "#0e3a18", "#1a6030", "#30d158", "#86efac"]
_AMBER_SEQ  = ["#1a0e00", "#3a2000", "#7a4500", "#ff9f0a", "#fcd34d"]


def _ax(title: str | None = None, angle: int | None = None,
        reversed: bool = False, size: int = 11) -> dict:
    d: dict = dict(
        gridcolor="rgba(255,255,255,0.04)",
        linecolor="rgba(255,255,255,0.04)",
        tickfont=dict(size=size, color="#44474f"),
        zeroline=False,
    )
    if title:   d["title"] = dict(text=title, font=dict(size=11, color="#44474f"))
    if angle is not None: d["tickangle"] = angle
    if reversed: d["autorange"] = "reversed"
    return d


def _chart(fig, height: int = 280) -> None:
    fig.update_layout(**_CT, height=height, margin=_M)
    st.plotly_chart(fig, use_container_width=True)


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
    return {"positive": "#30d158", "negative": "#ff453a", "neutral": "#636366"}.get(sentiment, "#636366")


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
    st.markdown("### ▶ YT Intelligence")
    st.caption("Ryan Trahan · CSCI370 Spring 2026")
    st.divider()

    st.markdown("**Retrieval**")
    top_k     = st.slider("Top-k results", 1, 10, 4, key="sidebar_k")
    retrieval = st.selectbox("Strategy", ["mmr", "similarity", "hyde"], key="sidebar_strat")

    st.divider()
    st.markdown("**System**")

    def _status(ok: bool, label: str) -> None:
        st.markdown(f"{'🟢' if ok else '🔴'} {label}")

    _status(PANDAS,  "pandas")
    _status(PLOTLY,  "plotly")
    _status(data_ok, "data loaded")

    try:    import faiss;  _status(True,  "faiss-cpu")
    except: _status(False, "faiss-cpu")

    try:    import dspy;   _status(True,  "dspy")
    except: _status(False, "dspy")

    try:
        import requests as _r
        _status(_r.get("http://localhost:11434", timeout=1).ok, "Ollama")
    except:
        _status(False, "Ollama (offline)")

    if data_ok:
        st.divider()
        st.markdown("**Dataset**")
        st.caption(f"{len(df):,} comments")
        if "sentiment" in df.columns:
            vc = df["sentiment"].value_counts()
            for s, c in vc.items():
                st.caption(f"{_dot(s).replace('#','')} {s}: {c:,}")

    st.divider()
    custom_path = st.text_input("CSV path", value=_csv_path, key="csv_path_input")
    if custom_path != _csv_path:
        try:
            df = load_data(custom_path)
            _csv_path = custom_path
            data_ok   = True
            st.success("Loaded.")
        except Exception as e:
            st.error(str(e))

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">CSCI370 · Spring 2026 · NLP + RAG</div>
  <div class="hero-title">Ryan Trahan Intelligence Engine</div>
  <div class="hero-sub">End-to-end YouTube comment analysis — sentiment, topics, entities, and source-grounded AI answers</div>
  <div class="hero-tags">
    <span class="hero-tag">VADER + RoBERTa</span>
    <span class="hero-tag">spaCy NER</span>
    <span class="hero-tag">KeyBERT + YAKE</span>
    <span class="hero-tag">BERTopic</span>
    <span class="hero-tag">FAISS + BM25</span>
    <span class="hero-tag">MMR · HyDE · Hybrid</span>
    <span class="hero-tag">DSPy</span>
    <span class="hero-tag">MLflow</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_analytics, tab_qa, tab_summary, tab_lab = st.tabs([
    "📊  Analytics",
    "💬  Ask AI",
    "📝  Summarize",
    "🔬  Retrieval Lab",
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
        <div class="kpi-accent" style="background:#7C3AED;"></div>
        <div class="kpi-label">Total Comments</div>
        <div class="kpi-value">{n:,}</div>
        <div class="kpi-sub">Ryan Trahan dataset</div>
      </div>
      <div class="kpi">
        <div class="kpi-accent" style="background:#30d158;"></div>
        <div class="kpi-label">Positive</div>
        <div class="kpi-value">{pos:,}</div>
        <div class="kpi-sub">{pos/n:.1%} of total</div>
      </div>
      <div class="kpi">
        <div class="kpi-accent" style="background:#ff453a;"></div>
        <div class="kpi-label">Negative</div>
        <div class="kpi-value">{neg:,}</div>
        <div class="kpi-sub">{neg/n:.1%} of total</div>
      </div>
      <div class="kpi">
        <div class="kpi-accent" style="background:#636366;"></div>
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
            fig = go.Figure(go.Pie(
                labels=svc.index.tolist(),
                values=svc.values.tolist(),
                hole=0.68,
                marker=dict(
                    colors=[_SENT_COLOR.get(s, "#636366") for s in svc.index],
                    line=dict(color="#0a0b0e", width=5),
                ),
                textinfo="label+percent",
                textfont=dict(size=12, color="#8a8d98"),
                hovertemplate="%{label}: %{value:,}<extra></extra>",
            ))
            fig.update_layout(
                **_CT,
                height=270,
                margin=_M,
                showlegend=False,
                annotations=[dict(
                    text=f"<b>{n:,}</b>",
                    x=0.5, y=0.5, font=dict(size=18, color="#f0f0f5"),
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
                            colorscale=_PURPLE_SEQ,
                            line=dict(width=0),
                        ),
                        hovertemplate="%{y}: %{x:,}<extra></extra>",
                    ))
                    fig.update_layout(
                        **_CT, height=270, margin=_M,
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
                    **_CT, height=310,
                    margin=dict(t=16, b=70, l=16, r=16),
                    xaxis=_ax(angle=-35, size=10),
                    yaxis=_ax("Comments"),
                    legend=dict(
                        orientation="h", y=1.08, x=0,
                        bgcolor="rgba(0,0,0,0)",
                        font=dict(size=11, color="#44474f"),
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
                        colorscale=_AMBER_SEQ,
                        line=dict(width=0),
                    ),
                    hovertemplate="%{y}: %{x:,}<extra></extra>",
                ))
                fig.update_layout(
                    **_CT, height=310, margin=_M,
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
                        colorscale=_GREEN_SEQ,
                        line=dict(width=0),
                    ),
                    hovertemplate="%{y}: %{x:,}<extra></extra>",
                ))
                fig.update_layout(
                    **_CT, height=330, margin=_M,
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
                if not ds.empty:
                    stc = (ds.groupby(["sentiment", "topic_label_sentiment"])
                             .size().reset_index(name="count"))
                    fig = px.bar(
                        stc, x="topic_label_sentiment", y="count",
                        color="sentiment", color_discrete_map=_SENT_COLOR,
                        barmode="group",
                        labels={"topic_label_sentiment": "", "count": "Comments", "sentiment": ""},
                    )
                    fig.update_traces(marker_line_width=0)
                    fig.update_layout(
                        **_CT, height=330,
                        margin=dict(t=16, b=70, l=16, r=16),
                        xaxis=_ax(angle=-35, size=10),
                        yaxis=_ax("Comments"),
                        legend=dict(
                            orientation="h", y=1.08, x=0,
                            bgcolor="rgba(0,0,0,0)",
                            font=dict(size=11, color="#44474f"),
                        ),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No sentiment-specific topic data.")
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
    st.caption("Start MLflow with `python -m mlflow server --port 5000` before running.")
    if st.button("Run evaluation", type="primary", key="eval_btn"):
        with st.spinner("Running evaluation pipeline …"):
            try:
                from src.evaluation.run_eval import run_full_evaluation
                run_full_evaluation(path=_csv_path, use_mock=True)
                st.success("Done — [open MLflow dashboard](http://localhost:5000)")
            except Exception as e:
                st.error(str(e))
                with st.expander("Traceback"):
                    st.code(_tb.format_exc())


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2  —  ASK AI
# ════════════════════════════════════════════════════════════════════════════════
with tab_qa:
    st.markdown("#### Ask a question about the comments")
    st.caption("Intent classifier → MMR retrieval → DSPy answer generation")

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
    st.caption("Run the same query through MMR, Similarity, and HyDE side-by-side")

    lab_q = st.text_input(
        "Query",
        value="What do viewers say about Ryan Trahan's storytelling?",
        key="lab_input",
    )

    if st.button("Compare strategies", type="primary", key="lab_submit"):
        if not data_ok:
            st.error("No data loaded.")
        else:
            with st.spinner("Running all three retrieval strategies …"):
                try:
                    from src.rag.generator import get_vector_store

                    store = get_vector_store(_csv_path)
                    strategies = {
                        "MMR":        (store.mmr_search,        "#7C3AED"),
                        "Similarity": (store.similarity_search, "#5e82ff"),
                        "HyDE":       (store.hyde_search,       "#ff9f0a"),
                    }

                    results: dict[str, list] = {}
                    for name, (fn, _) in strategies.items():
                        results[name] = fn(lab_q, k=top_k)

                    # Side-by-side columns
                    st.markdown('<div class="sec">Results</div>', unsafe_allow_html=True)
                    cols = st.columns(3)
                    for col, (name, (_, color)) in zip(cols, strategies.items()):
                        with col:
                            st.markdown(
                                f'<div class="strat-title" style="color:{color};">{name}</div>',
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
                                color_continuous_scale=["#111318", "#7C3AED"],
                                aspect="auto",
                                labels={"x": "Comment", "y": "Strategy", "color": "Retrieved"},
                            )
                            fig.update_layout(
                                **_CT, height=140, margin=_M,
                                coloraxis_showscale=False,
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption("Purple = retrieved   ·   shared columns = consensus between strategies")

                    # Per-strategy faithfulness
                    st.markdown('<div class="sec">Query Coverage</div>', unsafe_allow_html=True)
                    try:
                        from src.evaluation.generation_eval import faithfulness_score
                        mc = st.columns(3)
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
