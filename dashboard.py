"""
dashboard.py
Interaktywny dashboard Dash — wizualizacja analizy NLP Half-Life 2

Uruchomienie:  python dashboard.py
Otwórz:        http://127.0.0.1:8050
"""

import json
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc  # pip install dash-bootstrap-components
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

load_dotenv()


# ─────────────────────────────────────────────
# Wczytanie danych
# ─────────────────────────────────────────────

def load_data() -> dict:
    data = {}

    if Path("data/full_analysis.csv").exists():
        data["df"] = pd.read_csv("data/full_analysis.csv")
    else:
        # Dane demo jeśli pipeline nie był uruchomiony
        data["df"] = _generate_demo_data()

    if Path("data/topic_info.csv").exists():
        data["topic_info"] = pd.read_csv("data/topic_info.csv")

    if Path("data/topic_words.json").exists():
        with open("data/topic_words.json") as f:
            data["topic_words"] = json.load(f)
        data["topic_motifs"] = generate_topic_motifs(data["topic_words"])

    if Path("data/character_similarity.csv").exists():
        data["similarity"] = pd.read_csv("data/character_similarity.csv", index_col=0)

    if Path("data/topic_evaluation.json").exists():
        with open("data/topic_evaluation.json") as f:
            data["eval_metrics"] = json.load(f)

    if Path("data/topic_timeline.csv").exists():
        data["timeline"] = pd.read_csv("data/topic_timeline.csv")

    return data


def _fallback_topic_motif(words: list[str]) -> str:
    """Prosty fallback opisu motywu bez wywołania API."""
    if not words:
        return "Motyw ogólny dialogów bez wyraźnych słów kluczowych."
    head = ", ".join(words[:3])
    return f"Motyw rozmów wokół: {head}."


def _load_topic_motif_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_topic_motif_cache(cache_path: Path, payload: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def generate_topic_motifs(topic_words: dict) -> dict[str, str]:
    """
    Generuje krótkie opisy motywów tematów na podstawie słów kluczowych.
    Używa OpenAI (jeśli dostępny klucz), a wyniki cache'uje do pliku.
    """
    if not topic_words:
        return {}

    cache_path = Path("data/topic_motifs.json")
    cache = _load_topic_motif_cache(cache_path)
    motifs: dict[str, str] = {}
    pending: list[tuple[str, list[str]]] = []

    for raw_tid, words in topic_words.items():
        tid = str(raw_tid)
        safe_words = [str(w) for w in words][:8]
        cached = cache.get(tid, {})
        if (
            isinstance(cached, dict)
            and cached.get("keywords") == safe_words
            and isinstance(cached.get("motif"), str)
            and cached.get("motif").strip()
        ):
            motifs[tid] = cached["motif"].strip()
        else:
            pending.append((tid, safe_words))

    can_use_openai = bool(OpenAI) and bool(os.getenv("OPENAI_API_KEY"))
    client = OpenAI() if can_use_openai else None

    for tid, words in pending:
        motif = ""
        if client is not None:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.3,
                    max_tokens=80,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Jestes analitykiem narracji gier. "
                                "Na podstawie slow kluczowych podaj 1 zdanie po polsku "
                                "opisujace motyw tematu. Bez list, bez cudzyslowu."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Slowa kluczowe tematu: {', '.join(words)}",
                        },
                    ],
                )
                motif = (response.choices[0].message.content or "").strip()
            except Exception:
                motif = ""

        if not motif:
            motif = _fallback_topic_motif(words)

        motifs[tid] = motif

    updated_cache = {}
    for raw_tid, words in topic_words.items():
        tid = str(raw_tid)
        safe_words = [str(w) for w in words][:8]
        updated_cache[tid] = {"keywords": safe_words, "motif": motifs.get(tid, _fallback_topic_motif(safe_words))}

    _save_topic_motif_cache(cache_path, updated_cache)
    return motifs


def _generate_demo_data() -> pd.DataFrame:
    """Dane demo gdy pipeline nie był uruchomiony."""
    import random
    random.seed(42)

    chapters = ["2a", "2b", "2c", "2d", "2e", "2f", "2g", "2h", "2i", "2j", "2k", "2l", "2m", "2n"]
    chapter_names = [
        "Point Insertion", "A Red Letter Day", "Route Kanal", "Water Hazard",
        "Black Mesa East", "We Don't Go To Ravenholm", "Highway 17", "Sandtraps",
        "Nova Prospekt", "Entanglement", "Anti-citizen One", "Follow Freeman!",
        "Our Benefactors", "Dark Energy",
    ]
    characters = ["G-Man", "Alyx", "Dr. Breen", "Barney", "Dr. Kleiner",
                  "Eli", "Dr. Mossman", "Father Gregori", "Vortigaunt", "Citizen"]
    sentiments = ["positive", "neutral", "negative"]
    topics = list(range(8))

    rows = []
    for i in range(300):
        ch_idx = random.randint(0, len(chapters) - 1)
        rows.append({
            "chapter_id": chapters[ch_idx],
            "chapter_name": chapter_names[ch_idx],
            "character": random.choice(characters),
            "text": f"Sample dialogue line {i}",
            "sentiment": random.choice(sentiments),
            "sentiment_score": round(random.uniform(0.5, 1.0), 3),
            "topic": random.choice(topics),
            "topic_prob": round(random.uniform(0.3, 1.0), 3),
            "is_main_character": random.choice([True, False]),
            "line_index": i,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Kolory i styl HL2
# ─────────────────────────────────────────────

HL2_COLORS = {
    "bg": "#0a0e14",
    "surface": "#111820",
    "accent": "#e05c00",       # pomarańczowy HEV
    "accent2": "#1a6b9a",      # niebieski kombajn
    "text": "#c8cdd5",
    "text_dim": "#6b7585",
    "positive": "#3ddc84",
    "neutral": "#ffd166",
    "negative": "#ef476f",
    "grid": "#1e2833",
}

CHARACTER_COLORS = {
    "G-Man": "#9b5de5",
    "Alyx": "#3ddc84",
    "Dr. Breen": "#ef476f",
    "Barney": "#ffd166",
    "Dr. Kleiner": "#00bbf9",
    "Eli": "#f15bb5",
    "Dr. Mossman": "#fee440",
    "Father Gregori": "#ff6b35",
    "Vortigaunt": "#06d6a0",
    "Citizen": "#8ecae6",
    "Resistance": "#73d2de",
    "Civil Patrol": "#d62828",
    "Radio": "#aaaaaa",
}

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=HL2_COLORS["bg"],
        plot_bgcolor=HL2_COLORS["surface"],
        font=dict(color=HL2_COLORS["text"], family="Courier New, monospace"),
        xaxis=dict(gridcolor=HL2_COLORS["grid"], zerolinecolor=HL2_COLORS["grid"]),
        yaxis=dict(gridcolor=HL2_COLORS["grid"], zerolinecolor=HL2_COLORS["grid"]),
        colorway=list(CHARACTER_COLORS.values()),
    )
)


# ─────────────────────────────────────────────
# Wykresy
# ─────────────────────────────────────────────

def fig_dialogue_distribution(df: pd.DataFrame) -> go.Figure:
    """Liczba linii dialogowych per postać."""
    counts = df["character"].value_counts().head(12).reset_index()
    counts.columns = ["character", "count"]

    colors = [CHARACTER_COLORS.get(c, HL2_COLORS["accent"]) for c in counts["character"]]

    fig = go.Figure(go.Bar(
        x=counts["count"],
        y=counts["character"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=counts["count"],
        textposition="outside",
        textfont=dict(color=HL2_COLORS["text"]),
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="Liczba linii dialogowych per postać", font=dict(size=14, color=HL2_COLORS["accent"])),
        height=420,
        margin=dict(l=120, r=40, t=60, b=40),
        yaxis=dict(autorange="reversed", **PLOTLY_TEMPLATE["layout"]["yaxis"]),
    )
    return fig


def fig_sentiment_timeline(df: pd.DataFrame) -> go.Figure:
    """Sentyment w czasie (per rozdział)."""
    chapter_order = ["2a","2b","2c","2d","2e","2f","2g","2h","2i","2j","2k","2l","2m","2n"]
    chapter_labels = {
        "2a": "Point\nInsertion", "2b": "Red Letter\nDay", "2c": "Route\nKanal",
        "2d": "Water\nHazard", "2e": "Black Mesa\nEast", "2f": "Ravenholm",
        "2g": "Highway 17", "2h": "Sandtraps", "2i": "Nova\nProspekt",
        "2j": "Entanglement", "2k": "Anti-citizen\nOne", "2l": "Follow\nFreeman",
        "2m": "Our\nBenefactors", "2n": "Dark\nEnergy",
    }

    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
    df2 = df.copy()
    df2["sentiment_val"] = df2["sentiment"].map(sentiment_map).fillna(0)

    grouped = df2.groupby("chapter_id")["sentiment_val"].mean().reindex(chapter_order).dropna()

    colors = [HL2_COLORS["positive"] if v > 0.1
              else HL2_COLORS["negative"] if v < -0.1
              else HL2_COLORS["neutral"]
              for v in grouped.values]

    fig = go.Figure()

    # Obszar pod krzywą
    fig.add_trace(go.Scatter(
        x=list(grouped.index),
        y=grouped.values,
        fill="tozeroy",
        mode="lines+markers",
        line=dict(color=HL2_COLORS["accent"], width=2),
        marker=dict(color=colors, size=8, line=dict(color=HL2_COLORS["bg"], width=1)),
        name="Średni sentyment",
        hovertemplate="<b>%{x}</b><br>Sentyment: %{y:.2f}<extra></extra>",
    ))

    fig.add_hline(y=0, line_dash="dash", line_color=HL2_COLORS["text_dim"], line_width=1)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="Timeline sentymentu per rozdział", font=dict(size=14, color=HL2_COLORS["accent"])),
        xaxis=dict(
            ticktext=[chapter_labels.get(c, c) for c in grouped.index],
            tickvals=list(grouped.index),
            **PLOTLY_TEMPLATE["layout"]["xaxis"],
        ),
        yaxis=dict(title="Sentyment (−1 neg / +1 poz)", **PLOTLY_TEMPLATE["layout"]["yaxis"]),
        height=320,
        margin=dict(l=60, r=40, t=60, b=80),
        showlegend=False,
    )
    return fig


def fig_sentiment_per_character(df: pd.DataFrame) -> go.Figure:
    """Rozkład sentymentu dla każdej głównej postaci."""
    main_chars = list(CHARACTER_COLORS.keys())
    df_main = df[df["character"].isin(main_chars)]

    sentiment_pivot = (
        df_main.groupby(["character", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["positive", "neutral", "negative"], fill_value=0)
    )

    fig = go.Figure()
    for sentiment, color in [
        ("positive", HL2_COLORS["positive"]),
        ("neutral", HL2_COLORS["neutral"]),
        ("negative", HL2_COLORS["negative"]),
    ]:
        if sentiment in sentiment_pivot.columns:
            fig.add_trace(go.Bar(
                name=sentiment.capitalize(),
                x=sentiment_pivot.index.tolist(),
                y=sentiment_pivot[sentiment].tolist(),
                marker_color=color,
            ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="Rozkład sentymentu per postać", font=dict(size=14, color=HL2_COLORS["accent"])),
        barmode="stack",
        height=380,
        margin=dict(l=40, r=40, t=60, b=80),
        legend=dict(bgcolor=HL2_COLORS["surface"], bordercolor=HL2_COLORS["grid"]),
        xaxis=dict(tickangle=-30, **PLOTLY_TEMPLATE["layout"]["xaxis"]),
    )
    return fig


def fig_topic_timeline(df: pd.DataFrame, topic_words: dict) -> go.Figure:
    """Heatmapa: rozdziały × tematy."""
    chapter_order = ["2a","2b","2c","2d","2e","2f","2g","2h","2i","2j","2k","2l","2m","2n"]

    if "topic" not in df.columns:
        return go.Figure().update_layout(template=PLOTLY_TEMPLATE, title="Brak danych BERTopic")

    df_valid = df[df["topic"] != -1].copy()
    pivot = (
        df_valid.groupby(["chapter_id", "topic"])
        .size()
        .unstack(fill_value=0)
        .reindex(chapter_order, fill_value=0)
    )

    # Etykiety tematów
    topic_labels = {
        int(k): f"T{k}: {', '.join(v[:2])}"
        for k, v in topic_words.items()
    } if topic_words else {}

    y_labels = [topic_labels.get(int(c), f"Temat {c}") for c in pivot.columns]

    fig = go.Figure(go.Heatmap(
        z=pivot.values.T,
        x=pivot.index.tolist(),
        y=y_labels,
        colorscale=[[0, HL2_COLORS["surface"]], [0.5, HL2_COLORS["accent2"]], [1, HL2_COLORS["accent"]]],
        showscale=True,
        hovertemplate="Rozdział: %{x}<br>Temat: %{y}<br>Liczba linii: %{z}<extra></extra>",
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="Tematy BERTopic × rozdziały (heatmapa)", font=dict(size=14, color=HL2_COLORS["accent"])),
        height=400,
        margin=dict(l=200, r=40, t=60, b=60),
        xaxis=dict(tickangle=-30, **PLOTLY_TEMPLATE["layout"]["xaxis"]),
    )
    return fig


def fig_character_similarity(sim_df: pd.DataFrame) -> go.Figure:
    """Heatmapa podobieństwa postaci (cosine similarity embeddingów)."""
    fig = go.Figure(go.Heatmap(
        z=sim_df.values,
        x=sim_df.columns.tolist(),
        y=sim_df.index.tolist(),
        colorscale=[[0, HL2_COLORS["bg"]], [0.5, HL2_COLORS["accent2"]], [1, HL2_COLORS["accent"]]],
        showscale=True,
        text=np.round(sim_df.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=10, color=HL2_COLORS["text"]),
        hovertemplate="%{y} ↔ %{x}<br>Similarity: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="Podobieństwo stylu postaci (cosine similarity embeddingów)", font=dict(size=14, color=HL2_COLORS["accent"])),
        height=450,
        margin=dict(l=120, r=40, t=60, b=120),
        xaxis=dict(tickangle=-45, **PLOTLY_TEMPLATE["layout"]["xaxis"]),
    )
    return fig


def fig_chapter_activity(df: pd.DataFrame) -> go.Figure:
    """Aktywność dialogowa per rozdział + breakdown postaci."""
    chapter_order = ["2a","2b","2c","2d","2e","2f","2g","2h","2i","2j","2k","2l","2m","2n"]
    chapter_labels = {
        "2a": "Point Insertion", "2b": "Red Letter Day", "2c": "Route Kanal",
        "2d": "Water Hazard", "2e": "Black Mesa East", "2f": "Ravenholm",
        "2g": "Highway 17", "2h": "Sandtraps", "2i": "Nova Prospekt",
        "2j": "Entanglement", "2k": "Anti-citizen One", "2l": "Follow Freeman",
        "2m": "Our Benefactors", "2n": "Dark Energy",
    }
    top_chars = df["character"].value_counts().head(6).index.tolist()

    fig = go.Figure()
    for char in top_chars:
        char_df = df[df["character"] == char]
        counts = char_df.groupby("chapter_id").size().reindex(chapter_order, fill_value=0)
        fig.add_trace(go.Bar(
            name=char,
            x=chapter_order,
            y=counts.values,
            marker_color=CHARACTER_COLORS.get(char, "#888"),
        ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="Aktywność dialogowa per rozdział", font=dict(size=14, color=HL2_COLORS["accent"])),
        barmode="stack",
        height=340,
        margin=dict(l=40, r=40, t=60, b=60),
        legend=dict(bgcolor=HL2_COLORS["surface"], bordercolor=HL2_COLORS["grid"], orientation="h", y=-0.2),
        xaxis=dict(
            tickvals=chapter_order,
            ticktext=[chapter_labels.get(ch, ch) for ch in chapter_order],
            tickangle=-20,
            **PLOTLY_TEMPLATE["layout"]["xaxis"],
        ),
    )
    return fig


def card_metrics(eval_metrics: dict) -> list:
    """Karty z metrykami ewaluacji tematów."""
    if not eval_metrics:
        return [html.P("Brak danych ewaluacji.", style={"color": HL2_COLORS["text_dim"]})]

    items = [
        ("Topic Diversity", eval_metrics.get("topic_diversity", "—"), "Unikalność słów kluczowych tematów"),
        ("Topic Coherence", eval_metrics.get("mean_intra_topic_coherence", "—"), "Śr. cos. similarity w tematach"),
        ("Coverage", f"{int(eval_metrics.get('topic_coverage', 0)*100)}%", "% linii z przypisanym tematem"),
        ("Liczba tematów", eval_metrics.get("n_topics", "—"), "Wykrytych przez BERTopic"),
        ("Szum (−1)", eval_metrics.get("n_noise_docs", "—"), "Linie bez przypisanego tematu"),
    ]

    cards = []
    for label, value, desc in items:
        cards.append(
            html.Div([
                html.Div(str(value), style={
                    "fontSize": "28px", "fontWeight": "bold",
                    "color": HL2_COLORS["accent"], "fontFamily": "Courier New",
                }),
                html.Div(label, style={"color": HL2_COLORS["text"], "fontSize": "13px", "marginTop": "2px"}),
                html.Div(desc, style={"color": HL2_COLORS["text_dim"], "fontSize": "11px", "marginTop": "4px"}),
            ], style={
                "background": HL2_COLORS["surface"],
                "border": f"1px solid {HL2_COLORS['grid']}",
                "borderLeft": f"3px solid {HL2_COLORS['accent']}",
                "padding": "16px 20px",
                "borderRadius": "4px",
                "minWidth": "160px",
                "flex": "1",
            })
        )
    return cards


# ─────────────────────────────────────────────
# Layout aplikacji Dash
# ─────────────────────────────────────────────

def create_app(data: dict) -> dash.Dash:
    df = data["df"]
    topic_words = data.get("topic_words", {})
    topic_motifs = data.get("topic_motifs", {})
    sim_df = data.get("similarity")
    eval_metrics = data.get("eval_metrics", {})

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG],
        suppress_callback_exceptions=True,
    )

    # ── Sidebar ──────────────────────────────────────────────────────────
    sidebar = html.Div([
        html.Div("HL2 NLP", style={
            "fontSize": "22px", "fontWeight": "bold", "letterSpacing": "4px",
            "color": HL2_COLORS["accent"], "fontFamily": "Courier New",
            "padding": "24px 20px 8px",
        }),
        html.Div("NARRATIVE ANALYSIS", style={
            "fontSize": "10px", "letterSpacing": "3px",
            "color": HL2_COLORS["text_dim"], "padding": "0 20px 24px",
        }),
        html.Hr(style={"borderColor": HL2_COLORS["grid"], "margin": "0"}),

        # Nav items
        *[
            html.Div(label, id=f"nav-{tab_id}", style={
                "padding": "14px 24px",
                "cursor": "pointer",
                "color": HL2_COLORS["text"],
                "fontSize": "12px",
                "letterSpacing": "1px",
                "borderLeft": "3px solid transparent",
            })
            for tab_id, label in [
                ("overview", "► PRZEGLĄD"),
                ("sentiment", "► SENTYMENT"),
                ("topics", "► TEMATY (BERTopic)"),
                ("characters", "► POSTACI"),
                ("generator", "► GENERATOR POSTACI"),
            ]
        ],

        html.Hr(style={"borderColor": HL2_COLORS["grid"], "margin": "16px 0 8px"}),
        html.Div([
            html.Div(f"Linie dialogowe: {len(df)}", style={"color": HL2_COLORS["text_dim"], "fontSize": "11px"}),
            html.Div(f"Postaci: {df['character'].nunique()}", style={"color": HL2_COLORS["text_dim"], "fontSize": "11px"}),
            html.Div(f"Rozdziałów: {df['chapter_id'].nunique()}", style={"color": HL2_COLORS["text_dim"], "fontSize": "11px"}),
        ], style={"padding": "8px 24px"}),
    ], style={
        "width": "220px",
        "minHeight": "100vh",
        "background": HL2_COLORS["surface"],
        "borderRight": f"1px solid {HL2_COLORS['grid']}",
        "position": "fixed",
        "top": 0, "left": 0, "bottom": 0,
        "overflowY": "auto",
    })

    # ── Główna treść ──────────────────────────────────────────────────────
    content = html.Div(
        id="page-content",
        style={
            "marginLeft": "220px",
            "padding": "32px 40px",
            "minHeight": "100vh",
            "background": HL2_COLORS["bg"],
            "color": HL2_COLORS["text"],
            "fontFamily": "Courier New, monospace",
        }
    )

    dcc_store = dcc.Store(id="current-tab", data="overview")

    app.layout = html.Div([dcc_store, sidebar, content])

    # ── Callbacks ─────────────────────────────────────────────────────────

    nav_ids = ["nav-overview", "nav-sentiment", "nav-topics", "nav-characters", "nav-generator"]

    @app.callback(
        Output("current-tab", "data"),
        [Input(nav_id, "n_clicks") for nav_id in nav_ids],
        prevent_initial_call=True,
    )
    def switch_tab(*args):
        ctx = callback_context
        if not ctx.triggered:
            return "overview"
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        tab_map = {
            "nav-overview": "overview",
            "nav-sentiment": "sentiment",
            "nav-topics": "topics",
            "nav-characters": "characters",
            "nav-generator": "generator",
        }
        return tab_map.get(triggered_id, "overview")

    @app.callback(
        Output("page-content", "children"),
        Input("current-tab", "data"),
    )
    def render_page(tab):
        if tab == "overview":
            return render_overview(df, eval_metrics, topic_words)
        elif tab == "sentiment":
            return render_sentiment(df)
        elif tab == "topics":
            return render_topics(df, topic_words, topic_motifs)
        elif tab == "characters":
            return render_characters(df, sim_df)
        elif tab == "generator":
            return render_generator()
        return html.Div("Wybierz sekcję z menu.")

    # Generator callback (single line)
    @app.callback(
        Output("generator-output", "children"),
        Input("btn-generate", "n_clicks"),
        [
            State("gen-character", "value"),
            State("gen-situation", "value"),
            State("gen-sentiment", "value"),
        ],
        prevent_initial_call=True,
    )
    def generate_line(n_clicks, character, situation, sentiment):
        if not character or not situation:
            return html.Div("Uzupełnij postać i sytuację.", style={"color": HL2_COLORS["negative"]})

        try:
            from character_generator import CharacterGenerator
            gen = CharacterGenerator(api_key=os.environ.get("OPENAI_API_KEY"), df=df)
            line = gen.generate(character, situation, sentiment or "neutral")
            return html.Div([
                html.Div(f"[{character.upper()}]", style={
                    "color": CHARACTER_COLORS.get(character, HL2_COLORS["accent"]),
                    "fontWeight": "bold", "marginBottom": "8px",
                }),
                html.Div(f'"{line}"', style={
                    "fontSize": "16px", "fontStyle": "italic",
                    "color": HL2_COLORS["text"],
                    "borderLeft": f"3px solid {CHARACTER_COLORS.get(character, HL2_COLORS['accent'])}",
                    "paddingLeft": "16px",
                }),
            ])
        except Exception as e:
            return html.Div([
                html.Div("Błąd generowania:", style={"color": HL2_COLORS["negative"]}),
                html.Code(str(e), style={"fontSize": "11px", "color": HL2_COLORS["text_dim"]}),
            ])

    # Generator callback (chat mode)
    @app.callback(
        Output("chat-history-store", "data"),
        Output("chat-window", "children"),
        Output("chat-input", "value"),
        Input("btn-chat-send", "n_clicks"),
        Input("btn-chat-clear", "n_clicks"),
        State("gen-character", "value"),
        State("gen-sentiment", "value"),
        State("chat-input", "value"),
        State("chat-history-store", "data"),
        prevent_initial_call=True,
    )
    def chat_with_character(send_clicks, clear_clicks, character, sentiment, user_message, history):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update

        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        history = history or []

        if triggered_id == "btn-chat-clear":
            welcome = html.Div(
                "Czat wyczyszczony. Napisz pierwszą wiadomość.",
                style={"color": HL2_COLORS["text_dim"], "fontSize": "12px"},
            )
            return [], [welcome], ""

        if not character:
            error_msg = html.Div("Najpierw wybierz postać.", style={"color": HL2_COLORS["negative"]})
            return history, [error_msg], user_message or ""

        if not user_message or not user_message.strip():
            return history, no_update, user_message

        user_text = user_message.strip()
        history.append({"role": "user", "text": user_text})

        try:
            from character_generator import CharacterGenerator

            gen = CharacterGenerator(api_key=os.environ.get("OPENAI_API_KEY"), df=df)
            recent_history = history[-8:]
            convo_context = "\n".join(
                f"{'Player' if msg['role'] == 'user' else character}: {msg['text']}"
                for msg in recent_history
            )
            situation = f"Conversation so far:\n{convo_context}\n\nReply to the latest Player line naturally."
            reply = gen.generate(character, situation, sentiment or "neutral")
            history.append({"role": "assistant", "text": reply})
        except Exception as e:
            history.append({"role": "assistant", "text": f"[Błąd]: {e}"})

        bubbles = []
        for msg in history[-12:]:
            is_user = msg["role"] == "user"
            bubbles.append(html.Div(
                msg["text"],
                style={
                    "alignSelf": "flex-end" if is_user else "flex-start",
                    "maxWidth": "85%",
                    "padding": "10px 12px",
                    "marginBottom": "8px",
                    "background": HL2_COLORS["accent2"] if is_user else HL2_COLORS["surface"],
                    "border": f"1px solid {HL2_COLORS['grid']}",
                    "borderLeft": f"3px solid {HL2_COLORS['accent']}" if not is_user else f"3px solid {HL2_COLORS['accent2']}",
                    "color": HL2_COLORS["text"],
                    "fontSize": "13px",
                    "lineHeight": "1.45",
                },
            ))

        return history, bubbles, ""

    return app


# ─────────────────────────────────────────────
# Strony
# ─────────────────────────────────────────────

TITLE_STYLE = {"fontSize": "20px", "fontWeight": "bold", "color": "#e05c00",
               "letterSpacing": "2px", "marginBottom": "24px", "fontFamily": "Courier New"}
SUBTITLE_STYLE = {"fontSize": "14px", "color": "#6b7585", "marginBottom": "16px",
                  "fontFamily": "Courier New", "letterSpacing": "1px"}


def section(title: str, content):
    return html.Div([
        html.H3(title, style=TITLE_STYLE),
        content,
    ], style={"marginBottom": "40px"})


def render_overview(df, eval_metrics, topic_words):
    return html.Div([
        html.H2("ANALIZA NARRACYJNA — HALF-LIFE 2", style={
            "fontSize": "24px", "fontWeight": "bold", "letterSpacing": "4px",
            "color": HL2_COLORS["accent"], "marginBottom": "8px",
        }),
        html.Div("BERTopic • Sentiment Analysis • NER • Character Embeddings",
                 style={"color": HL2_COLORS["text_dim"], "marginBottom": "32px", "fontSize": "12px"}),

        # Metryki ewaluacji
        html.Div("METRYKI EWALUACJI TEMATÓW", style=SUBTITLE_STYLE),
        html.Div(card_metrics(eval_metrics), style={
            "display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "32px",
        }),

        # Rozkład dialogów
        dcc.Graph(figure=fig_dialogue_distribution(df), config={"displayModeBar": False}),

        html.Div(style={"height": "32px"}),
        dcc.Graph(figure=fig_chapter_activity(df), config={"displayModeBar": False}),
    ])


def render_sentiment(df):
    return html.Div([
        html.H2("ANALIZA SENTYMENTU", style=TITLE_STYLE),

        dcc.Graph(figure=fig_sentiment_timeline(df), config={"displayModeBar": False}),
        html.Div(style={"height": "24px"}),
        dcc.Graph(figure=fig_sentiment_per_character(df), config={"displayModeBar": False}),

        # Filtr postaci
        html.Div("FILTRUJ PO POSTACI", style={**SUBTITLE_STYLE, "marginTop": "32px"}),
        dcc.Dropdown(
            id="sentiment-char-filter",
            options=[{"label": c, "value": c} for c in sorted(df["character"].unique())],
            multi=True,
            placeholder="Wszystkie postaci...",
            style={"background": HL2_COLORS["surface"], "color": HL2_COLORS["text"]},
        ),
        dcc.Graph(id="sentiment-filtered-graph", config={"displayModeBar": False}),
    ])


def render_topics(df, topic_words, topic_motifs=None):
    topic_motifs = topic_motifs or {}
    return html.Div([
        html.H2("TOPIC MODELING — BERTopic", style=TITLE_STYLE),

        # Opisy motywów + słowa kluczowe
        html.Div("WYKRYTE TEMATY: MOTYW + SŁOWA KLUCZOWE", style=SUBTITLE_STYLE),
        html.Div([
            html.Div([
                html.Div(f"Temat {tid}", style={"color": HL2_COLORS["accent"], "fontWeight": "bold", "fontSize": "12px"}),
                html.Div(
                    topic_motifs.get(str(tid), _fallback_topic_motif(words)),
                    style={"color": HL2_COLORS["text"], "fontSize": "13px", "marginTop": "8px", "lineHeight": "1.5"},
                ),
                html.Div(
                    f"Słowa kluczowe: {', '.join(words)}",
                    style={"color": HL2_COLORS["text_dim"], "fontSize": "12px", "marginTop": "8px"},
                ),
            ], style={
                "padding": "10px 16px",
                "background": HL2_COLORS["surface"],
                "border": f"1px solid {HL2_COLORS['grid']}",
                "borderLeft": f"3px solid {HL2_COLORS['accent2']}",
                "borderRadius": "3px",
                "marginBottom": "8px",
            })
            for tid, words in (topic_words.items() if topic_words else {})
        ], style={"marginBottom": "32px"}),

        dcc.Graph(figure=fig_topic_timeline(df, topic_words), config={"displayModeBar": False}),
    ])


def render_characters(df, sim_df):
    content = [
        html.H2("ANALIZA POSTACI", style=TITLE_STYLE),
    ]

    if sim_df is not None:
        content.append(dcc.Graph(figure=fig_character_similarity(sim_df), config={"displayModeBar": False}))
    else:
        content.append(html.P(
            "Macierz podobieństwa niedostępna — uruchom nlp_pipeline.py",
            style={"color": HL2_COLORS["text_dim"]}
        ))

    # Statystyki per postać
    content.append(html.Div("STATYSTYKI PER POSTAĆ", style={**SUBTITLE_STYLE, "marginTop": "32px"}))

    char_stats = []
    for char in df["character"].value_counts().head(10).index:
        char_df = df[df["character"] == char]
        sentiments = char_df["sentiment"].value_counts().to_dict() if "sentiment" in char_df else {}
        char_stats.append(html.Div([
            html.Div(char, style={
                "color": CHARACTER_COLORS.get(char, HL2_COLORS["text"]),
                "fontWeight": "bold", "width": "150px", "flexShrink": "0",
            }),
            html.Div(f"{len(char_df)} linii", style={"width": "80px", "color": HL2_COLORS["text"]}),
            html.Div([
                html.Span(f"✓{sentiments.get('positive', 0)}", style={"color": HL2_COLORS["positive"], "marginRight": "8px"}),
                html.Span(f"~{sentiments.get('neutral', 0)}", style={"color": HL2_COLORS["neutral"], "marginRight": "8px"}),
                html.Span(f"✗{sentiments.get('negative', 0)}", style={"color": HL2_COLORS["negative"]}),
            ]),
        ], style={"display": "flex", "alignItems": "center", "gap": "16px", "padding": "10px 0",
                  "borderBottom": f"1px solid {HL2_COLORS['grid']}"}))

    content.append(html.Div(char_stats))
    return html.Div(content)


def render_generator():
    return html.Div([
        html.H2("GENERATOR POSTACI", style=TITLE_STYLE),
        html.Div("Generowanie wypowiedzi i czat w stylu postaci HL2 — OpenAI API",
                 style={"color": HL2_COLORS["text_dim"], "marginBottom": "28px", "fontSize": "12px"}),

        html.Div([
            html.Div([
                html.Label("POSTAĆ", style={"color": HL2_COLORS["text_dim"], "fontSize": "11px", "letterSpacing": "2px"}),
                dcc.Dropdown(
                    id="gen-character",
                    options=[
                        {"label": c, "value": c}
                        for c in ["G-Man", "Alyx", "Dr. Breen", "Barney", "Dr. Kleiner",
                                  "Eli", "Father Gregori", "Vortigaunt", "Dr. Mossman"]
                    ],
                    placeholder="Wybierz postać...",
                    style={"marginTop": "6px"},
                ),
            ], style={"marginBottom": "20px"}),

            html.Div([
                html.Label("SYTUACJA", style={"color": HL2_COLORS["text_dim"], "fontSize": "11px", "letterSpacing": "2px"}),
                dcc.Textarea(
                    id="gen-situation",
                    placeholder="Opisz sytuację, np. 'Gordon właśnie pokonał strażnika Combine'",
                    style={
                        "width": "100%", "height": "80px", "marginTop": "6px",
                        "background": HL2_COLORS["surface"], "color": HL2_COLORS["text"],
                        "border": f"1px solid {HL2_COLORS['grid']}", "padding": "10px",
                        "fontFamily": "Courier New", "resize": "vertical",
                    },
                ),
            ], style={"marginBottom": "20px"}),

            html.Div([
                html.Label("TON", style={"color": HL2_COLORS["text_dim"], "fontSize": "11px", "letterSpacing": "2px"}),
                dcc.RadioItems(
                    id="gen-sentiment",
                    options=[
                        {"label": " Pozytywny", "value": "positive"},
                        {"label": " Neutralny", "value": "neutral"},
                        {"label": " Negatywny", "value": "negative"},
                    ],
                    value="neutral",
                    inline=True,
                    style={"color": HL2_COLORS["text"], "marginTop": "8px", "gap": "16px"},
                ),
            ], style={"marginBottom": "24px"}),

            html.Button("GENERUJ WYPOWIEDŹ", id="btn-generate", n_clicks=0, style={
                "background": HL2_COLORS["accent"],
                "color": "#fff",
                "border": "none",
                "padding": "12px 28px",
                "fontFamily": "Courier New",
                "letterSpacing": "2px",
                "fontSize": "13px",
                "cursor": "pointer",
                "marginBottom": "28px",
            }),

            html.Div("CZAT Z POSTACIĄ", style={**SUBTITLE_STYLE, "marginBottom": "10px"}),
            dcc.Store(id="chat-history-store", data=[]),
            html.Div(id="chat-window", children=[
                html.Div(
                    "Napisz wiadomość i kliknij WYŚLIJ, aby rozpocząć rozmowę.",
                    style={"color": HL2_COLORS["text_dim"], "fontSize": "12px"},
                )
            ], style={
                "minHeight": "220px",
                "maxHeight": "320px",
                "overflowY": "auto",
                "display": "flex",
                "flexDirection": "column",
                "padding": "12px",
                "background": HL2_COLORS["bg"],
                "border": f"1px solid {HL2_COLORS['grid']}",
                "borderRadius": "4px",
                "marginBottom": "10px",
            }),
            dcc.Textarea(
                id="chat-input",
                placeholder="Napisz do postaci...",
                style={
                    "width": "100%", "height": "70px", "marginTop": "6px",
                    "background": HL2_COLORS["surface"], "color": HL2_COLORS["text"],
                    "border": f"1px solid {HL2_COLORS['grid']}", "padding": "10px",
                    "fontFamily": "Courier New", "resize": "vertical",
                },
            ),
            html.Div([
                html.Button("WYŚLIJ", id="btn-chat-send", n_clicks=0, style={
                    "background": HL2_COLORS["accent2"],
                    "color": "#fff",
                    "border": "none",
                    "padding": "10px 18px",
                    "fontFamily": "Courier New",
                    "letterSpacing": "1px",
                    "fontSize": "12px",
                    "cursor": "pointer",
                }),
                html.Button("WYCZYŚĆ CZAT", id="btn-chat-clear", n_clicks=0, style={
                    "background": "transparent",
                    "color": HL2_COLORS["text_dim"],
                    "border": f"1px solid {HL2_COLORS['grid']}",
                    "padding": "10px 14px",
                    "fontFamily": "Courier New",
                    "letterSpacing": "1px",
                    "fontSize": "12px",
                    "cursor": "pointer",
                }),
            ], style={"display": "flex", "gap": "10px", "marginTop": "10px", "marginBottom": "24px"}),
        ], style={"maxWidth": "600px"}),

        html.Div(id="generator-output", style={
            "minHeight": "80px",
            "padding": "20px",
            "background": HL2_COLORS["surface"],
            "border": f"1px solid {HL2_COLORS['grid']}",
            "borderRadius": "4px",
            "maxWidth": "700px",
        }),

        html.Div([
            html.Div("JAK TO DZIAŁA", style={"color": HL2_COLORS["accent"], "fontWeight": "bold", "marginBottom": "12px", "fontSize": "13px"}),
            html.Div([
                html.P("1. Profil postaci: opis stylu + przykładowe linie z oryginalnego skryptu"),
                html.P("2. Dane stylistyczne: średnia długość zdania, częstotliwość pytań/wykrzyknień"),
                html.P("3. OpenAI API generuje wypowiedź i odpowiedzi czatu w stylu postaci"),
                html.P("4. Ustaw ton: pozytywny / neutralny / negatywny"),
                html.P("5. Wymagana zmienna środowiskowa: OPENAI_API_KEY"),
            ], style={"color": HL2_COLORS["text_dim"], "fontSize": "12px", "lineHeight": "1.8"}),
        ], style={
            "marginTop": "32px", "padding": "20px",
            "background": HL2_COLORS["surface"],
            "border": f"1px solid {HL2_COLORS['grid']}",
            "borderLeft": f"3px solid {HL2_COLORS['accent2']}",
            "maxWidth": "700px",
        }),
    ])


# ─────────────────────────────────────────────
# Callback dla filtru sentymentu
# ─────────────────────────────────────────────

def register_callbacks(app, df):
    @app.callback(
        Output("sentiment-filtered-graph", "figure"),
        Input("sentiment-char-filter", "value"),
    )
    def update_sentiment_filter(selected_chars):
        filtered = df if not selected_chars else df[df["character"].isin(selected_chars)]
        return fig_sentiment_per_character(filtered)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("[Dashboard] Wczytywanie danych...")
    data = load_data()
    app = create_app(data)
    register_callbacks(app, data["df"])
    print("[Dashboard] Uruchamianie serwera na http://127.0.0.1:8050")
    app.run(debug=True, port=8050)