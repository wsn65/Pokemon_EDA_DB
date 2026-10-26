import random
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import learning_curve, train_test_split

st.set_page_config(
    page_title="Pokémon Data Insights",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent

# ---------- Palette ----------
BG = "#F9F7F2"
CARD = "#FFFEFC"
BORDER = "#E5E2DA"
# Pokémon palette: red, dark red, blue, yellow, gold
POKE_RED = "#FF0000"
POKE_DARK_RED = "#CC0000"
POKE_BLUE = "#3B4CCA"
POKE_YELLOW = "#FFDE00"
POKE_GOLD = "#B3A125"
POKE_CYCLE = [POKE_RED, POKE_DARK_RED, POKE_BLUE, POKE_YELLOW, POKE_GOLD]

BLUE = POKE_BLUE
DARK_BLUE = "#2F3DA3"
BLUE_TINT = "#E9EBFB"
YELLOW = POKE_YELLOW
RED = POKE_DARK_RED
TEXT = "#0F172A"
MUTED = "#64748B"
# Display font: rounded, chunky headings (Fredoka, loaded from Google Fonts; falls back to Nunito / system fonts)
DISPLAY_FONT = "'Fredoka', 'Nunito', system-ui, sans-serif"
PLOTLY_DISPLAY_FONT = "Fredoka, Nunito, sans-serif"
# Pixel font for the small grey labels (Google Fonts, loaded in the <style> block below)
PIXEL_FONT = "'Press Start 2P', ui-monospace, monospace"
PLOTLY_PIXEL_FONT = "Press Start 2P, monospace"
SHOW_INSIGHT_FACTS = False  # True adds the computed one-line fact above the notebook text in every "Details"

CAT_COLORS = {"Single Type": BLUE, "Dual Type": YELLOW}
# Poké Ball donut: red top half, white bottom half, dark outline
BALL_COLORS = {"Dual Type": "#EE1515", "Single Type": "#FFFEFC"}
BALL_TEXT = {"Dual Type": "#FFFFFF", "Single Type": "#1F2937"}
EXP_COLORS = {"Low": BLUE, "Medium": YELLOW, "High": RED}
EXP_ORDER = {"exp_group": ["Low", "Medium", "High"]}
# One color per stat, from the Pokémon red / blue / yellow palette
STAT_COLORS = {"attack": "#FF0000", "defense": "#3B4CCA", "special_attack": "#CC0000",
               "special_defense": "#7C8BE0", "hp": "#FFDE00", "speed": "#B3A125"}

# Same six stats, same order as the notebook
STATS = ["attack", "defense", "special_attack", "special_defense", "hp", "speed"]
LABEL = {"attack": "Attack", "defense": "Defense", "special_attack": "Special Attack",
         "special_defense": "Special Defense", "hp": "HP", "speed": "Speed"}

# ---------- Style ----------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@500;600;700&family=Press+Start+2P&display=swap');
:root {{ color-scheme: light; }}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background-color: {BG} !important; color: {TEXT}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ max-width: 1280px; padding-top: 2rem; }}
section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {{ background-color: {CARD} !important; }}
section[data-testid="stSidebar"] {{ border-right: 1px solid {BORDER}; }}
h1, h2, h3 {{ color: {BLUE} !important; }}
h1, h2, h3, [data-testid="stHeading"], [data-testid="stHeading"] * {{ font-family: {DISPLAY_FONT} !important; font-weight: 600 !important; letter-spacing: .01em; }}
.kpi-value, .ins-value, .p-name, .p-tile .v, [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {{ font-family: {DISPLAY_FONT} !important; font-weight: 600; }}
button[data-baseweb="tab"], button[data-baseweb="tab"] * {{ font-family: {DISPLAY_FONT} !important; font-size: 16px; }}
p, label, li, [data-testid="stMarkdownContainer"] {{ color: {TEXT}; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {{ color: {MUTED} !important; }}
[data-testid="stWidgetLabel"] p {{ color: {MUTED}; font-weight: 400 !important; font-family: {PIXEL_FONT} !important; font-size: 10px !important; line-height: 1.6 !important; }}
[data-testid="stRadio"] label[data-baseweb="radio"] [data-testid="stMarkdownContainer"] p {{ font-family: {PIXEL_FONT} !important; font-size: 10px !important; font-weight: 400 !important; line-height: 1.6 !important; }}
[data-testid="stMetric"] {{
    background: {CARD}; border: 1px solid {BORDER}; border-top: 3px solid {YELLOW};
    border-radius: 12px; padding: 14px; box-shadow: 0 1px 2px rgba(15,23,42,.06);
}}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {{ color: {MUTED} !important; }}
[data-testid="stMetricValue"] {{ color: {BLUE}; }}
hr {{ border-color: {YELLOW} !important; opacity: .8; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {CARD}; border-color: {BORDER}; border-radius: 12px; box-shadow: 0 1px 2px rgba(15,23,42,.05);
}}
button[data-baseweb="tab"] {{ color: {MUTED}; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color: {BLUE}; font-weight: 600; }}
div[data-baseweb="tab-highlight"] {{ background-color: {BLUE} !important; }}
div[data-baseweb="tab-border"] {{ background-color: {BORDER} !important; }}
div[data-baseweb="select"] > div {{ background-color: {CARD} !important; border-color: {BORDER} !important; }}
div[data-baseweb="select"] * {{ color: {TEXT} !important; }}
div[data-baseweb="select"] svg {{ fill: {MUTED} !important; }}
span[data-baseweb="tag"] {{ background-color: {BLUE_TINT} !important; }}
span[data-baseweb="tag"] * {{ color: {DARK_BLUE} !important; font-weight: 600; }}
div[data-baseweb="popover"] ul {{ background: {CARD} !important; }}
div[data-baseweb="popover"] li {{ color: {TEXT} !important; }}
div[data-baseweb="popover"] li:hover {{ background: #F3F0E9 !important; }}
[data-testid="stExpander"] {{ border: 1px solid {BORDER}; border-radius: 12px; background: {CARD}; }}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {{ color: {BLUE} !important; font-weight: 600; }}
.kpi {{
    background: {CARD}; border: 1px solid {BORDER}; border-top: 3px solid {YELLOW};
    border-radius: 12px; padding: 14px 18px; box-shadow: 0 1px 2px rgba(15,23,42,.06); height: 100%;
}}
.kpi-label {{ font-family: {PIXEL_FONT}; font-size: 10px; font-weight: 400; line-height: 1.5; color: {MUTED}; }}
.kpi-value {{ font-size: 34px; font-weight: 700; color: {BLUE}; line-height: 1.2; margin-top: 2px; }}
.kpi-note {{ font-size: 13px; color: {MUTED}; margin-top: 2px; }}
.ins-value {{ font-size: 26px; font-weight: 700; color: {BLUE}; line-height: 1.25; margin-top: 2px; }}
.ins-fact {{ font-size: 13px; font-weight: 500; color: {MUTED}; margin-top: 4px; }}
.ins-take {{ font-size: 12.5px; color: {MUTED}; margin-top: 6px; padding-top: 6px; border-top: 1px dashed {BORDER}; line-height: 1.45; }}
[class*="st-key-insd_"] [data-testid="stExpander"] {{ border: none !important; background: transparent !important; box-shadow: none !important; }}
[class*="st-key-insd_"] [data-testid="stExpander"] summary {{ padding: 0 4px; min-height: 0; }}
[class*="st-key-insd_"] [data-testid="stExpander"] summary, [class*="st-key-insd_"] [data-testid="stExpander"] summary * {{ color: {MUTED} !important; font-weight: 500 !important; font-size: 12.5px !important; }}
.about {{ max-width: 760px; margin: 6px auto 10px; text-align: center; }}
.about-row {{ padding: 12px 0; border-bottom: 1px solid {BORDER}; }}
.about-row:last-child {{ border-bottom: none; }}
.about-label {{
    display: inline-block; font-size: 12px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
    color: #3B2F00; background: {YELLOW}; border-radius: 999px; padding: 2px 12px; margin-bottom: 6px;
}}
.about-text {{ font-size: 16px; color: {TEXT}; line-height: 1.5; }}
.about-text a {{ color: {BLUE}; font-weight: 600; text-decoration: none; }}
.about-text a:hover {{ text-decoration: underline; }}
section[data-testid="stSidebar"] .about {{ margin: 0 auto; }}
section[data-testid="stSidebar"] .about-row {{ padding: 8px 0; }}
section[data-testid="stSidebar"] .about-text {{ font-size: 14px; }}
.p-name {{ font-size: 34px; font-weight: 700; color: {BLUE}; text-transform: capitalize; line-height: 1.1; }}
.p-id {{ font-size: 15px; color: {MUTED}; font-weight: 600; margin: 2px 0 10px; }}
.p-chip {{
    display: inline-block; font-size: 13px; font-weight: 700; text-transform: capitalize;
    padding: 3px 14px; border-radius: 999px; margin: 0 6px 6px 0; letter-spacing: .02em;
}}
.p-chip.egg {{ background: {BLUE_TINT}; color: {DARK_BLUE}; font-weight: 600; }}
.p-tiles {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin: 8px 0 10px; }}
.p-tile {{ background: {CARD}; border: 1px solid {BORDER}; border-top: 3px solid {YELLOW}; border-radius: 10px; padding: 8px 12px; }}
.p-tile .l {{ font-family: {PIXEL_FONT}; font-size: 9px; font-weight: 400; line-height: 1.6; color: {MUTED}; }}
.p-tile .v {{ font-size: 20px; font-weight: 700; color: {BLUE}; line-height: 1.3; }}
.p-label {{ font-family: {PIXEL_FONT}; font-size: 9px; font-weight: 400; letter-spacing: .04em; text-transform: uppercase; color: {MUTED}; margin: 10px 0 6px; }}
.pokeball {{
    width: 34px; height: 34px; border-radius: 50%; border: 3px solid #1F2937; box-sizing: border-box; position: relative;
    background: linear-gradient(#EE1515 46%, #1F2937 46% 54%, #FFFEFC 54%); margin: 0 auto 8px;
}}
.pokeball::after {{
    content: ""; position: absolute; left: 50%; top: 50%; width: 12px; height: 12px; transform: translate(-50%, -50%);
    border-radius: 50%; background: #FFFEFC; border: 3px solid #1F2937; box-sizing: border-box;
}}
.pokeball.throw-a {{ animation: ball-a 1.3s ease-in-out; }}
.pokeball.throw-b {{ animation: ball-b 1.3s ease-in-out; }}
@keyframes ball-a {{ 0% {{ transform: translateY(-26px) rotate(0deg); }} 30% {{ transform: translateY(0) rotate(0deg); }}
    45% {{ transform: rotate(-28deg); }} 60% {{ transform: rotate(28deg); }} 75% {{ transform: rotate(-16deg); }} 100% {{ transform: rotate(0deg); }} }}
@keyframes ball-b {{ 0% {{ transform: translateY(-26px) rotate(0deg); }} 30% {{ transform: translateY(0) rotate(0deg); }}
    45% {{ transform: rotate(-28deg); }} 60% {{ transform: rotate(28deg); }} 75% {{ transform: rotate(-16deg); }} 100% {{ transform: rotate(0deg); }} }}
.summary {{
    background: {BLUE_TINT}; border-left: 4px solid {YELLOW}; border-radius: 8px;
    padding: 10px 14px; font-size: 14px; color: #1E3A6E; margin: 4px 0 10px;
}}
</style>
""", unsafe_allow_html=True)


# ---------- Helpers ----------
def themed(fig, height=None):
    fig.update_layout(
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", color=TEXT),
        title_font_family=PLOTLY_DISPLAY_FONT,
        legend=dict(title_text="", font=dict(color=TEXT)),
    )
    if fig.layout.margin.l is None:  # a figure can set its own margins; otherwise use the default
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER)
    # Softer bars: rounded corners and a little more space between bars
    try:
        fig.update_traces(marker_cornerradius=8, selector=dict(type="bar"))
    except Exception:
        pass  # older Plotly versions don't support rounded corners
    if any(t.type == "bar" for t in fig.data) and fig.layout.bargap is None:
        fig.update_layout(bargap=0.22)
    if height:
        fig.update_layout(height=height)
    return fig


_chart_counter = [0]  # resets on every script run, so keys stay stable between reruns


def show(fig, height=400):
    _chart_counter[0] += 1
    key = f"chart_{_chart_counter[0]}"  # unique key: two identical charts can no longer clash
    try:
        st.plotly_chart(themed(fig, height), width="stretch", key=key)
    except TypeError:
        st.plotly_chart(themed(fig, height), use_container_width=True, key=key)


def show_df(d):
    try:
        st.dataframe(d, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(d, use_container_width=True, hide_index=True)


def kpi(label, value):
    return (f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div></div>')


_ins_counter = [0]


def insight(col, label, value, fact, take):
    """Compact insight card next to a chart; the details stay hidden behind a small, quiet toggle."""
    _ins_counter[0] += 1
    col.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="ins-value">{value}</div></div>',
                 unsafe_allow_html=True)
    try:
        box = col.container(key=f"insd_{_ins_counter[0]}")
    except TypeError:  # older Streamlit without container keys: the toggle just keeps the default look
        box = col.container()
    with box:
        with st.expander("Details"):
            show_fact = bool(fact) and SHOW_INSIGHT_FACTS
            fact_html = f'<div class="ins-fact" style="margin-top:0">{fact}</div>' if show_fact else ""
            take_style = "" if show_fact else ' style="margin-top:0;padding-top:0;border-top:none"'
            st.markdown(f'{fact_html}<div class="ins-take"{take_style}>{take}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def url_works(url):
    """True when the image URL answers with an image (checked once per URL, from the machine running the app)."""
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}, method="GET")
        with urllib.request.urlopen(req, timeout=6) as r:
            return r.status == 200 and "image" in r.headers.get("Content-Type", "")
    except Exception:
        return False


def pokemon_image(row):
    """First image that actually exists: local images/ folder, the dataset's url_image, then PokeAPI artwork."""
    folder = BASE_DIR / "images"
    if folder.exists():
        pid = int(row["id"])
        for stem in (f"{pid:03d}", str(pid), str(row["pokemon"]), str(row["pokemon"]).title()):
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                p = folder / f"{stem}{ext}"
                if p.exists():
                    return str(p)
    if isinstance(row["url_image"], str) and url_works(row["url_image"]):
        return row["url_image"]
    # `id` in this dataset is the PokeAPI Pokémon id, so it also works for alternate forms
    fallback = ("https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/"
                f"official-artwork/{int(row['id'])}.png")
    return fallback if url_works(fallback) else None


def type_chip(name):
    color = TYPE_COLORS.get(name, BLUE)
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    ink = "#1F2937" if (0.299 * r + 0.587 * g + 0.114 * b) > 160 else "#FFFFFF"
    return f'<span class="p-chip" style="background:{color};color:{ink}">{name}</span>'


def summary(text):
    st.markdown(f'<div class="summary">{text}</div>', unsafe_allow_html=True)


def hist_kde(series, bins, color=BLUE):
    """Histogram with a KDE curve, like seaborn's histplot(kde=True)."""
    s = series.dropna()
    counts, edges = np.histogram(s, bins=bins)
    width = edges[1] - edges[0]
    traces = [go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts, width=width, showlegend=False,
                     marker=dict(color=color, opacity=0.8, line=dict(color="white", width=1)),
                     hovertemplate="%{x:.0f}: %{y} Pokémon<extra></extra>")]
    try:
        xs = np.linspace(s.min(), s.max(), 200)
        ys = gaussian_kde(s)(xs) * len(s) * width
        traces.append(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=DARK_BLUE, width=2), showlegend=False, hoverinfo="skip"))
    except Exception:
        pass
    return traces


DIST_EDGES = np.arange(0, 270, 10)  # same bins for every distribution chart


def stat_dist(series, color, name, bar_opacity, legend=True):
    """Bars + KDE curve with the same bins and style for every stat."""
    s = series.dropna()
    counts, _ = np.histogram(s, bins=DIST_EDGES)
    centers = (DIST_EDGES[:-1] + DIST_EDGES[1:]) / 2
    traces = [go.Bar(x=centers, y=counts, width=10, name=name, showlegend=legend,
                     legendgroup=name,
                     marker=dict(color=color, opacity=bar_opacity, line=dict(color="white", width=1)),
                     hovertemplate=name + " %{x:.0f}: %{y} Pokémon<extra></extra>")]
    try:
        xs = np.linspace(max(0, s.min()), s.max(), 200)
        ys = gaussian_kde(s)(xs) * len(s) * 10
        traces.append(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=2.5),
                                 name=name, showlegend=False, legendgroup=name, hoverinfo="skip"))
    except Exception:
        pass
    return traces


HEAT_SCALE = [[0.0, "#6F7FE0"], [0.5, "#FFFBEA"], [1.0, "#F26B6B"]]  # Pokémon blue -> cream -> Pokémon red, a bit lighter


def heatmap(c, height=420):
    fig = px.imshow(c, text_auto=".2f", zmin=-1, zmax=1, aspect="auto", color_continuous_scale=HEAT_SCALE)
    fig.update_coloraxes(colorbar=dict(title="r", thickness=12))
    return fig


def ball_donut(series, center, avg=False):
    """Poké Ball donut for Single vs Dual type: counts (value + %) or average stat (value only)."""
    series = series.reindex([c for c in ("Dual Type", "Single Type") if c in series.index])
    labels = list(series.index)
    fmt = dict(texttemplate="%{value:.1f}") if avg else dict(textinfo="value+percent")
    fig = go.Figure(go.Pie(
        labels=labels, values=series.values, hole=0.6, sort=False, direction="clockwise",
        marker=dict(colors=[BALL_COLORS[c] for c in labels], line=dict(color="#1F2937", width=4)),
        textfont=dict(size=15, color=[BALL_TEXT[c] for c in labels]), insidetextorientation="horizontal",
        hovertemplate=("%{label}: %{value:.1f} average stat<extra></extra>" if avg
                       else "%{label}: %{value} Pokémon (%{percent})<extra></extra>"),
        **fmt,
    ))
    fig.update_layout(
        annotations=[dict(text=center, x=0.5, y=0.5, showarrow=False, font=dict(size=18, color=TEXT))],
        legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"),
    )
    return fig


def bar_counts(counts, xname, yname="Count", color=BLUE, height=380, horizontal=False):
    d = counts.reset_index()
    d.columns = [xname, yname]
    fig = px.bar(d, x=xname, y=yname, text=yname)
    fig.update_traces(marker_color=color, textposition="outside", cliponaxis=False)
    return fig


def colored_bar(counts, xname, colors, yname="Count"):
    """Bar chart where bars cycle through the Pokémon palette (red, dark red, blue, yellow, gold)."""
    d = counts.reset_index()
    d.columns = [xname, yname]
    d[xname] = d[xname].astype(str)
    palette = {cat: POKE_CYCLE[i % len(POKE_CYCLE)] for i, cat in enumerate(d[xname])}
    fig = px.bar(d, x=xname, y=yname, text=yname, color=xname, color_discrete_map=palette)
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_xaxes(type="category")
    fig.update_layout(showlegend=False)
    return fig


# One color per generation, inspired by the main-series games
GEN_COLORS = {"1": "#EF3B2D", "2": "#D4A017", "3": "#2E9E6B", "4": "#7090F0",
              "5": "#4B5563", "6": "#2D6CDF", "7": "#F59E0B", "8": "#A855F7", "9": "#E11D74"}


# ---------- Data ----------
@st.cache_data
def load():
    df = pd.read_csv(BASE_DIR / "clean_pokemon.csv")
    df["generation_id"] = df["generation_id"].astype(int)
    return df


try:
    df = load()
except FileNotFoundError:
    st.error("`clean_pokemon.csv` was not found. Keep app.py in the same folder as the CSV files "
             "and run `streamlit run app.py` from that folder.")
    st.stop()

# ---------- Header ----------
def find_title_logo():
    """Logo from the images/ folder: title / logo / banner / pokemon first, otherwise any non-Pokémon image there."""
    folder = BASE_DIR / "images"
    if not folder.exists():
        return None
    files = {p.stem.lower(): p for p in sorted(folder.iterdir())
             if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")}
    for stem in ("title", "logo", "pokemon_logo", "pokemon-logo", "banner", "pokemon"):
        if stem in files:
            return files[stem]
    names = {str(n).lower() for n in df["pokemon"]}
    for stem, p in files.items():
        if not stem.isdigit() and stem not in names:
            return p
    return None


@st.cache_data(show_spinner=False)
def image_data_uri(path, mtime):
    import base64
    mime = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp"}[Path(path).suffix.lower()]
    return f"data:image/{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


_logo = find_title_logo()
if _logo:
    # mix-blend-mode: multiply hides the white JPG background on the light page
    _logo_html = (f'<img src="{image_data_uri(str(_logo), _logo.stat().st_mtime)}" alt="Pokémon" '
                  'style="width:min(360px,80%);height:auto;mix-blend-mode:multiply">')
else:
    _logo_html = f'<div style="font-size:56px;font-weight:800;color:{BLUE}">Pokémon</div>'
st.markdown(
    f'<div style="text-align:center;margin:0 0 4px">{_logo_html}'
    f'<div style="font-family:{DISPLAY_FONT};font-size:40px;font-weight:700;letter-spacing:.35em;text-indent:.35em;'
    f'text-transform:uppercase;color:{BLUE};line-height:1.1;margin-top:4px">Dashboard</div>'
    f'<div style="font-family:{PIXEL_FONT};font-size:10px;line-height:1.6;color:{MUTED};margin-top:12px">Explore the Pokémon world through data</div>'
    f'</div>',
    unsafe_allow_html=True,
)
st.divider()

# ---------- Sidebar: dataset overview + filters ----------
st.sidebar.header("Dataset overview")
st.sidebar.markdown(
    """
<div class="about">
  <div class="about-row"><div class="about-label">Dataset</div><div class="about-text">Pokemon</div></div>
  <div class="about-row"><div class="about-label">Source</div><div class="about-text">
<a href="https://github.com/rfordatascience/tidytuesday/blob/main/data/2025/2025-04-01/readme.md" target="_blank">TidyTuesday Pokemon dataset</a></div></div>
  <div class="about-row"><div class="about-label">Description</div><div class="about-text">
This dataset contains information about Pokémon including their physical characteristics, types and battle statistics.</div></div>
</div>
""",
    unsafe_allow_html=True,
)

st.sidebar.header("Filters")
gens = sorted(df["generation_id"].unique())
sel_gen = st.sidebar.multiselect("Generation", gens, default=gens)
sel_cat = st.sidebar.multiselect("Type category", list(CAT_COLORS), default=list(CAT_COLORS))
all_types = sorted(df["type_1"].unique())
sel_type = st.sidebar.multiselect("Primary type", all_types, default=all_types, format_func=str.title)
all_eggs = sorted(df["egg_group_1"].unique())
sel_egg = st.sidebar.multiselect("Egg group", all_eggs, default=all_eggs, format_func=str.title)
f = df[df["generation_id"].isin(sel_gen) & df["type_cat"].isin(sel_cat)
       & df["type_1"].isin(sel_type) & df["egg_group_1"].isin(sel_egg)]
st.sidebar.caption(f"{len(f):,} of {len(df):,} Pokémon selected")
st.sidebar.caption("The ML Model tab always uses the full dataset, as in the notebook.")

TYPE_COLORS = df.groupby("type_1")["color_1"].first().to_dict()

empty = f.empty
if empty:
    st.warning("No Pokémon match the current filters. Widen the filters in the sidebar.")
else:
    n_types = len(set(f["type_1"]) | set(f["type_2"].dropna()))
    k = st.columns(4)
    k[0].markdown(kpi("Total Pokémon", f"{len(f):,}"), unsafe_allow_html=True)
    k[1].markdown(kpi("Pokémon Types", f"{n_types}"), unsafe_allow_html=True)
    k[2].markdown(kpi("Generations", f"{f['generation_id'].nunique()}"), unsafe_allow_html=True)
    k[3].markdown(kpi("Dual-Type Pokémon", f"{(f['type_cat'] == 'Dual Type').mean():.0%}"), unsafe_allow_html=True)
    st.write("")

tabs = st.tabs(["Overview", "Pokémon Types", "Battle Statistics", "Base Experience", "Physical Characteristics", "ML Model"])


# =============== Overview ===============
def throw_pokeball():
    """Throw a Poké Ball: a random Pokémon from the whole dataset becomes the selected profile."""
    ss = st.session_state
    row = df.sample(1).iloc[0]
    ss["profile_pick"] = row["pokemon"]
    ss["ball_throws"] = ss.get("ball_throws", 0) + 1
    msg = f"Gotcha! {row['pokemon'].title()} was caught!"
    if row["avg_stat"] >= df["avg_stat"].quantile(0.95):
        msg += " Wow, it is in the top 5% of average stats!"
    ss["ball_msg"] = msg


def clear_ball_msg():
    st.session_state.pop("ball_msg", None)


with tabs[0]:
    # ---- Pokémon profile: pick any Pokémon (independent of the sidebar filters) ----
    with st.container(border=True):
        names = sorted(df["pokemon"])
        default = names.index("pikachu") if "pikachu" in names else 0
        pick_col, throw_col, ball_col, _ = st.columns([1.5, 0.9, 0.3, 0.8], vertical_alignment="bottom")
        chosen = pick_col.selectbox("Pokémon profile — search a Pokémon by name", names, index=default,
                                    format_func=lambda n: n.title(), key="profile_pick", on_change=clear_ball_msg)
        throws = st.session_state.get("ball_throws", 0)
        try:
            throw_col.button("Throw a Pokéball", on_click=throw_pokeball, key="throw_ball", width="stretch")
        except TypeError:
            throw_col.button("Throw a Pokéball", on_click=throw_pokeball, key="throw_ball", use_container_width=True)
        ball_col.markdown(f'<div class="pokeball{" throw-" + ("a" if throws % 2 else "b") if throws else ""}"></div>',
                          unsafe_allow_html=True)
        if st.session_state.get("ball_msg"):
            st.markdown(f'<div style="font-family:{PIXEL_FONT};font-size:10px;line-height:1.6;color:{BLUE};margin:2px 0 8px">'
                        f'{st.session_state["ball_msg"]}</div>', unsafe_allow_html=True)
        p = df[df["pokemon"] == chosen].iloc[0]
        img_col, info_col = st.columns([1.2, 2], gap="large", vertical_alignment="center")
        with img_col:
            img = pokemon_image(p)
            if img:
                st.image(img, width=400)
            else:
                st.markdown(f'<div style="background:{BLUE_TINT};border-radius:16px;height:400px;display:flex;'
                            f'align-items:center;justify-content:center;color:{MUTED};text-align:center">'
                            'No image found.<br>Add it to the images/ folder as 025.png</div>', unsafe_allow_html=True)
        with info_col:
            chips = type_chip(p["type_1"]) + (type_chip(p["type_2"]) if isinstance(p["type_2"], str) else "")
            eggs = f'<span class="p-chip egg">{p["egg_group_1"]}</span>' + (
                f'<span class="p-chip egg">{p["egg_group_2"]}</span>' if isinstance(p["egg_group_2"], str) else "")
            head_l, head_r = st.columns([1.1, 1.3], gap="medium")
            with head_l:
                st.markdown(
                    f'<div class="p-name">{p["pokemon"]}</div>'
                    f'<div class="p-id">#{int(p["id"])} · Generation {int(p["generation_id"])} · {p["type_cat"]}</div>'
                    f'<div class="p-label">Type</div>{chips}'
                    f'<div class="p-label">Egg group</div>{eggs}',
                    unsafe_allow_html=True,
                )
            with head_r:
                st.markdown(
                    f'<div class="p-tiles">'
                    f'<div class="p-tile"><div class="l">Height</div><div class="v">{p["height"]} m</div></div>'
                    f'<div class="p-tile"><div class="l">Weight</div><div class="v">{p["weight"]} kg</div></div>'
                    f'<div class="p-tile"><div class="l">Base experience</div><div class="v">{int(p["base_experience"])}</div></div>'
                    f'<div class="p-tile"><div class="l">Average stat</div><div class="v">{p["avg_stat"]:.1f}</div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            sdf = pd.DataFrame({"Stat": [LABEL[s] for s in STATS], "Value": [int(p[s]) for s in STATS],
                                "Dataset average": [df[s].mean() for s in STATS]})
            fig = go.Figure()
            fig.add_trace(go.Bar(x=sdf["Value"], y=sdf["Stat"], orientation="h", text=sdf["Value"],
                                 marker_color=[STAT_COLORS[s] for s in STATS], name=p["pokemon"].title(),
                                 textposition="outside", cliponaxis=False,
                                 hovertemplate="%{y}: %{x}<extra></extra>"))
            fig.add_trace(go.Scatter(x=sdf["Dataset average"], y=sdf["Stat"], mode="markers", name="Dataset average",
                                     marker=dict(symbol="line-ns", size=18, line=dict(color="#1F2937", width=3)),
                                     hovertemplate="Average: %{x:.0f}<extra></extra>"))
            # Stat names sit above each bar (not on the left), so the bars start at the same left edge as the name
            fig.update_yaxes(autorange="reversed", title="", showticklabels=False)
            fig.update_xaxes(title="Base stat", range=[0, 270])
            for stat_label in sdf["Stat"]:
                fig.add_annotation(x=0, y=stat_label, xref="x", yref="y", text=stat_label, showarrow=False,
                                   xanchor="left", yanchor="bottom", yshift=11,
                                   font=dict(size=9, color=MUTED, family=PLOTLY_PIXEL_FONT))
            fig.update_layout(legend=dict(orientation="h", y=1.12, x=0), margin=dict(l=0, r=0, t=40, b=10), bargap=0.5)
            show(fig, 340)

    st.write("")
    if not empty:
        view = st.radio("Show Pokémon by", ["Type", "Generation"], horizontal=True, key="ov_view")
        if view == "Type":
            st.subheader("Pokémon per primary type")
            fig = colored_bar(f["type_1"].value_counts(), "type_1", TYPE_COLORS)
            fig.update_xaxes(title="type_1")
        else:
            st.subheader("Pokémon per generation")
            fig = colored_bar(f["generation_id"].value_counts().sort_index(), "generation_id", GEN_COLORS)
            fig.update_xaxes(title="generation_id")
        show(fig, 380)

        st.write("")
        view2 = st.radio("Show average stat by", ["Type", "Generation"], horizontal=True, key="ov_view2")
        if view2 == "Type":
            st.subheader("Average stat by primary type")
            a = f.groupby("type_1")["avg_stat"].mean().sort_values(ascending=False).round(2)
            fig = colored_bar(a, "Primary Type", TYPE_COLORS, "avg_stat")
            fig.update_xaxes(title="Primary Type")
        else:
            st.subheader("Average stat by generation")
            a = f.groupby("generation_id")["avg_stat"].mean().sort_index().round(2)
            fig = colored_bar(a, "generation_id", GEN_COLORS, "avg_stat")
            fig.update_xaxes(title="generation_id")
        fig.update_traces(texttemplate="%{y:.1f}")
        show(fig, 380)

        st.write("")
        left, right = st.columns([1, 1.6])
        with left:
            st.subheader("Single vs dual type")
            t = f["type_cat"].value_counts().reset_index()
            t.columns = ["Category", "Pokémon"]
            fig = go.Figure(go.Pie(
                labels=t["Category"], values=t["Pokémon"], hole=0.6, sort=False, direction="clockwise",
                marker=dict(colors=[BALL_COLORS[c] for c in t["Category"]], line=dict(color="#1F2937", width=4)),
                textinfo="value+percent", textfont=dict(size=15, color=[BALL_TEXT[c] for c in t["Category"]]),
                insidetextorientation="horizontal",
                hovertemplate="%{label}: %{value} Pokémon (%{percent})<extra></extra>",
            ))
            fig.update_layout(
                annotations=[dict(text=f"<b>{len(f):,}</b><br>Pokémon", x=0.5, y=0.5, showarrow=False,
                                  font=dict(size=18, color=TEXT))],
                legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"),
            )
            show(fig, 400)
        with right:
            st.subheader("Stat distributions")
            pick_col, _ = st.columns([1, 1.6])
            pick = pick_col.selectbox("Show distribution of", [LABEL[s] for s in STATS], key="ov_dist")
            s = next(k for k, v in LABEL.items() if v == pick)
            fig = go.Figure()
            for tr in stat_dist(f[s], STAT_COLORS[s], pick, 0.85, legend=False):
                fig.add_trace(tr)
            fig.update_layout(barmode="overlay", bargap=0.02)
            fig.update_xaxes(title="Stat value", range=[0, 260])
            fig.update_yaxes(title="Count")
            show(fig, 400)


# =============== Pokémon Types ===============
with tabs[1]:
    if not empty:
        T = lambda x: str(x).title()
        t1 = f["type_1"].value_counts()
        avg_t = f.groupby("type_1")["avg_stat"].mean().sort_values(ascending=False).round(2)
        e1 = f["egg_group_1"].value_counts()
        e2 = f["egg_group_2"].dropna().value_counts()
        combos = (f.dropna(subset=["egg_group_2"]).groupby(["egg_group_1", "egg_group_2"]).size()
                  .sort_values(ascending=False).reset_index(name="Count"))
        avg_c = f.groupby("type_cat")["avg_stat"].mean()
        dual_n = int((f["type_cat"] == "Dual Type").sum())
        WIDE = [3.2, 1]          # chart | insight card
        TWO = [1.5, 1.5, 1.1]    # chart | chart | insight cards

        st.subheader("Primary types")
        view = st.radio("Show primary types by", ["Number of Pokémon", "Average stat"], horizontal=True, key="types_view")
        chart, card = st.columns(WIDE, vertical_alignment="center")
        if view == "Number of Pokémon":
            with chart:
                show(colored_bar(f["type_1"].value_counts(), "type_1", TYPE_COLORS), 380)
            insight(card, "Most common primary type", T(t1.index[0]),
                    f"{t1.iloc[0]} Pokémon · least common: {T(t1.index[-1])} ({t1.iloc[-1]})",
                    "The dataset is not balanced: some types appear far more often than others.")
        else:
            with chart:
                fig = colored_bar(avg_t, "Primary Type", TYPE_COLORS, "avg_stat")
                fig.update_xaxes(title="Primary Type")
                fig.update_traces(texttemplate="%{y:.1f}")
                show(fig, 380)
            insight(card, "Highest average stat", f"{T(avg_t.index[0])} · {avg_t.iloc[0]:.1f}",
                    f"Lowest: {T(avg_t.index[-1])} ({avg_t.iloc[-1]:.1f})",
                    f"{T(avg_t.index[0])} types tend to have higher overall stats than other types.")

        st.subheader("Single vs dual type")
        view2 = st.radio("Show single vs dual type by", ["Number of Pokémon", "Average stat"], horizontal=True, key="cat_view")
        c1, c2, card = st.columns([1.5, 1.3, 1.1], vertical_alignment="center")
        if view2 == "Number of Pokémon":
            counts = f["type_cat"].value_counts()
            with c1:
                fig = bar_counts(counts, "type_cat", "Count")
                fig.update_layout(title="Number of Pokémon")
                show(fig, 360)
            with c2:
                show(ball_donut(counts, f"<b>{len(f):,}</b><br>Pokémon"), 360)
            more = "Dual" if dual_n >= len(f) - dual_n else "Single"
            close = abs(dual_n / len(f) - 0.5) < 0.1
            insight(card, "Dual-type share", f"{dual_n / len(f):.0%}",
                    f"{dual_n} of {len(f)} Pokémon are dual type",
                    f"{more} Type Pokémon are more common here" + (", but the two groups are close in size." if close else "."))
        else:
            m = avg_c.round(2)
            with c1:
                fig = bar_counts(m, "type_cat", "avg_stat")
                fig.update_layout(title="Average stat")
                fig.update_traces(marker_color=YELLOW)
                show(fig, 360)
            with c2:
                show(ball_donut(m, "<b>Average</b><br>stat", avg=True), 360)
            ranked = m.sort_values(ascending=False)
            top = ranked.index[0]
            other = ranked.index[1] if len(ranked) > 1 else None
            insight(card, "Higher average stat", f"{top} · {ranked.iloc[0]:.1f}",
                    f"{other}: {ranked.iloc[1]:.1f}" if other else "Only one type category in this selection",
                    f"{top} Pokémon score a bit higher on average, but not every {top} Pokémon beats every {other}."
                    if other else "Widen the filters to compare single and dual type.")

        st.subheader("Egg groups")
        c1, c2, cards = st.columns(TWO, vertical_alignment="center")
        with c1:
            fig = bar_counts(f["egg_group_1"].value_counts(), "egg_group_1", "Count")
            fig.update_layout(title="egg_group_1")
            show(fig, 380)
        with c2:
            fig = bar_counts(f["egg_group_2"].dropna().value_counts(), "egg_group_2", "Count")
            fig.update_layout(title="egg_group_2")
            fig.update_traces(marker_color=YELLOW)
            show(fig, 380)
        insight(cards, "Most common egg group", T(e1.index[0]),
                f"{e1.iloc[0]} Pokémon · least common: {T(e1.index[-1])} ({e1.iloc[-1]})",
                "No-eggs and indeterminate are special groups: those Pokémon cannot breed.")
        insight(cards, "Most common 2nd egg group", T(e2.index[0]) if len(e2) else "–",
                f"{e2.iloc[0]} Pokémon · least common: {T(e2.index[-1])} ({e2.iloc[-1]})" if len(e2) else "No second egg groups",
                f"Only {f['egg_group_2'].notna().mean():.0%} of Pokémon have a second egg group.")

        st.subheader("Egg group combinations")
        if combos.empty:
            st.info("No dual egg-group Pokémon in the current selection.")
        else:
            combos["Combination"] = combos["egg_group_1"] + " + " + combos["egg_group_2"]
            chart, card = st.columns(WIDE, vertical_alignment="center")
            with chart:
                fig = px.bar(combos, x="Combination", y="Count", text="Count")
                fig.update_traces(marker_color=BLUE, textposition="outside", cliponaxis=False)
                fig.update_xaxes(tickangle=-45, title="egg_group_1 + egg_group_2")
                show(fig, 520)
            insight(card, "Most common egg combination",
                    f"{T(combos['egg_group_1'][0])} + {T(combos['egg_group_2'][0])}",
                    f"{combos['Count'][0]} Pokémon · {len(combos)} combinations in total",
                    "Some egg-group combinations are much more frequent than others.")


# =============== Battle Statistics ===============
with tabs[2]:
    if not empty:
        st.subheader("Distributions")
        DIST = {  # stat: (bins, takeaway from the notebook)
            "attack": (20, "Most Pokémon have moderate attack values between 45 and 100. The distribution is slightly skewed "
                           "to the right, which means a small number of Pokémon have high attack values."),
            "defense": (20, "Most Pokémon have defense values between 40 and 100. The distribution of defense values is right skewed, "
                            "which indicates that a small number of Pokémon have high defense values."),
            "speed": (20, "There are two peaks in the Speed histogram, around 50–60 and 90–100. This means there are two common "
                          "speed ranges, with a small number of Pokémon having high speed values."),
            "hp": (25, "Most Pokémon have HP values between 50 and 80. The distribution is skewed to the right, which means a "
                       "small number of Pokémon have high HP values."),
        }
        pick = st.radio("Show distribution of", [LABEL[k] for k in DIST], horizontal=True, key="bs_dist")
        stat = next(k for k in DIST if LABEL[k] == pick)
        bins, note = DIST[stat]
        chart, card = st.columns([3.2, 1], vertical_alignment="center")
        with chart:
            fig = go.Figure(hist_kde(f[stat], bins, STAT_COLORS[stat]))
            fig.update_xaxes(title=LABEL[stat])
            fig.update_yaxes(title="Count")
            fig.update_layout(title=f"{LABEL[stat]} Distribution")
            show(fig, 400)
        q1, med, q3 = f[stat].quantile([0.25, 0.5, 0.75])
        insight(card, f"Typical {LABEL[stat]}", f"Median {med:.0f}",
                "", note)

        st.subheader("Boxplot of the stats")
        BOX_TAKE = {
            "all": "Attack and Speed show the greatest variability, while HP has a narrower distribution with many outliers. "
                   "Defense falls between them, with a few extreme values.",
            "attack": "Attack shows one of the greatest variabilities among the stats.",
            "speed": "Speed, like Attack, shows great variability.",
            "hp": "HP has a narrower distribution with many outliers.",
            "defense": "Defense falls between the others, with a few extreme values.",
        }
        OUTLIER_NOTE = " These outliers may represent Pokémon with unusually high stats, not data errors."
        box_pick = st.radio("Show boxplot of", ["All stats"] + [LABEL[k] for k in DIST], horizontal=True, key="bs_box")
        box_stats = list(DIST) if box_pick == "All stats" else [next(k for k in DIST if LABEL[k] == box_pick)]
        chart, card = st.columns([3.2, 1], vertical_alignment="center")
        with chart:
            fig = go.Figure()
            for k in box_stats:  # one color per stat, same palette as the rest of the app
                fig.add_trace(go.Box(y=f[k], name=LABEL[k], marker_color=STAT_COLORS[k], line_color=STAT_COLORS[k],
                                     boxpoints="outliers", text=f["pokemon"],
                                     hovertemplate="%{text}<br>%{y}<extra></extra>"))
            fig.update_layout(showlegend=False)
            if len(box_stats) == 1:
                fig.update_layout(boxgap=0.75)  # a single box would otherwise stretch across the whole chart
            show(fig, 420)
        spread, outs = {}, {}
        for k in box_stats:
            q1, q3 = f[k].quantile([0.25, 0.75])
            spread[k] = q3 - q1
            outs[k] = int(((f[k] > q3 + 1.5 * (q3 - q1)) | (f[k] < q1 - 1.5 * (q3 - q1))).sum())
        if len(box_stats) > 1:
            widest = max(spread, key=spread.get)
            insight(card, "Most variable stat", LABEL[widest],
                    "",
                    BOX_TAKE["all"] + OUTLIER_NOTE)
        else:
            k = box_stats[0]
            q1, q3 = f[k].quantile([0.25, 0.75])
            insight(card, f"{LABEL[k]} spread", f"{spread[k]:.0f} points",
                    "",
                    BOX_TAKE[k] + OUTLIER_NOTE)

        st.subheader("Correlation matrix of the stats")
        CORR_NOTES = {  # pair: note from the notebook ({r} is computed from the data)
            ("defense", "special_defense"): "Defense and Special Defense have a moderate positive relationship (r = {r}): Pokémon "
                                            "with high defense tend to have high special defense, but one does not necessarily cause the other.",
            ("special_attack", "special_defense"): "Special Attack and Special Defense also show a moderate positive linear relationship (r = {r}).",
            ("special_attack", "speed"): "Special Attack and Speed show a moderate positive relationship (r = {r}).",
            ("attack", "defense"): "Attack and Defense show a moderate positive linear relationship (r = {r}).",
            ("speed", "defense"): "Speed and Defense show almost no linear relationship (r = {r}).",
        }
        sel_stats = list(STATS)  # always compare all six stats
        if len(sel_stats) < 2:
            st.info("Pick at least two stats to see how they relate.")
        else:
            cm = f[sel_stats].corr()
            if cm.isna().any().any():
                st.info("Not enough Pokémon in the current selection to compute correlations.")
            else:
                pairs = [(x, y, cm.loc[x, y]) for i, x in enumerate(sel_stats) for y in sel_stats[i + 1:]]
                ranked = sorted(pairs, key=lambda p: -abs(p[2]))
                top_pairs = ranked[:2]  # the scatter plots show the two clearest relationships
                sa, sb, sr = ranked[0]
                wa, wb, wr = ranked[-1]

                # 1) Heatmap on the left, its scatter plots stacked on the right (same height as the heatmap)
                chart, scat = st.columns([1.3, 1], gap="large", vertical_alignment="center")
                with chart:
                    st.markdown("##### Correlation heatmap")
                    show(heatmap(cm.rename(index=LABEL, columns=LABEL)), 520)
                with scat:
                    st.markdown("##### Strongest relationships")
                    h = 240 if len(top_pairs) == 2 else 480
                    for (x, y, r), color in zip(top_pairs, (BLUE, RED)):
                        fig = px.scatter(f, x=x, y=y, hover_name="pokemon", labels={x: LABEL[x], y: LABEL[y]})
                        fig.update_traces(marker=dict(color=color, size=6, opacity=0.7))
                        slope, intercept = np.polyfit(f[x], f[y], 1)
                        xs = np.array([f[x].min(), f[x].max()])
                        fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", showlegend=False,
                                                 line=dict(color=TEXT, width=2, dash="dash"), hoverinfo="skip"))
                        fig.update_layout(title=f"{LABEL[x]} vs {LABEL[y]} (r = {r:.2f})")
                        show(fig, h)

                # 3) Insight card underneath
                notes = [t.format(r=f"{cm.loc[x, y]:.2f}") for (x, y), t in CORR_NOTES.items()
                         if x in sel_stats and y in sel_stats]
                scatter_names = " and ".join(f"{LABEL[x]} vs {LABEL[y]}" for x, y, _ in top_pairs)
                notes.append(f"The scatter plots ({scatter_names}) support the heatmap. A few outliers appear away from the main cluster.")
                insight(st.container(), "Strongest relationship", f"r = {sr:.2f}",
                        "",
                        "<br>".join(f"• {n}" for n in notes))


# =============== Base Experience ===============
with tabs[3]:
    if not empty:
        # ---- Distribution: chart + insight card, same layout as the Battle Statistics distributions ----
        st.subheader("Base experience distribution")
        chart, card = st.columns([3.2, 1], vertical_alignment="center")
        with chart:
            fig = go.Figure(hist_kde(f["base_experience"], 30))
            fig.update_xaxes(title="Base experience")
            fig.update_yaxes(title="Count")
            show(fig, 400)
        q1, med, q3 = f["base_experience"].quantile([0.25, 0.5, 0.75])
        insight(card, "Typical base experience", f"Median {med:.0f}",
                f"Half of Pokémon have base experience between {q1:.0f} and {q3:.0f}",
                "Most Pokémon have a base experience between 35 and 60 or between 150 and 190. "
                "The distribution is right skewed, with a few high outliers.")

        # ---- Relationship with the average stat: heatmap + scatter side by side, insight card underneath ----
        st.subheader("Base experience vs average stat")
        if len(f) < 3:
            st.info("Not enough Pokémon in the current selection to compute the correlation.")
        else:
            cm = f[["avg_stat", "base_experience"]].corr()
            r = cm.loc["avg_stat", "base_experience"]
            chart, scat = st.columns([1, 1.3], gap="large", vertical_alignment="center")
            with chart:
                st.markdown("##### Correlation heatmap")
                names = {"avg_stat": "Average stat", "base_experience": "Base experience"}
                fig = heatmap(cm.rename(index=names, columns=names))
                fig.update_xaxes(constrain="domain")
                fig.update_yaxes(scaleanchor="x", scaleratio=1, constrain="domain")  # square cells
                show(fig, 420)
            with scat:
                st.markdown("##### Scatter plot")
                fig = px.scatter(f, x="avg_stat", y="base_experience", hover_name="pokemon",
                                 labels={"avg_stat": "Average stat", "base_experience": "Base experience"})
                fig.update_traces(marker=dict(color=BLUE, size=7, opacity=0.7))
                slope, intercept = np.polyfit(f["avg_stat"], f["base_experience"], 1)
                xs = np.array([f["avg_stat"].min(), f["avg_stat"].max()])
                fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", showlegend=False,
                                         line=dict(color=TEXT, width=2, dash="dash"), hoverinfo="skip"))
                fig.update_layout(title=f"Base experience vs average stat (r = {r:.2f})")
                show(fig, 420)
            strength = "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak"
            direction = "positive" if r >= 0 else "negative"
            insight(st.container(), "Correlation", f"r = {r:.2f}",
                    "Base experience & average stat",
                    f"There is a {strength} {direction} correlation between base experience and the average stat. "
                    "Three distinct diagonal patterns are visible in the scatter plot.")


# =============== Physical Characteristics ===============
with tabs[4]:
    if not empty:
        # ---- Scatter and bar chart side by side, insight cards on the right ----
        st.subheader("Weight vs height and average by type category")
        measure = st.radio("Bar chart: show average", ["Height", "Weight"], horizontal=True, key="phys_view")
        is_h = measure == "Height"
        col_name, unit, adj = ("height", "m", "taller") if is_h else ("weight", "kg", "heavier")
        m = f.groupby("type_cat")[col_name].mean().round(2 if is_h else 1)
        c1, c2, cards = st.columns([1.5, 1.5, 1.1], vertical_alignment="center")
        with c1:
            fig = px.scatter(f, x="weight", y="height", color="type_cat", hover_name="pokemon",
                             color_discrete_map=CAT_COLORS,
                             labels={"type_cat": "Type category", "weight": "Weight (kg)", "height": "Height (m)"})
            fig.update_traces(marker=dict(size=7, opacity=0.8))
            if len(f) >= 3:
                slope, intercept = np.polyfit(f["weight"], f["height"], 1)
                xs = np.array([f["weight"].min(), f["weight"].max()])
                fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", showlegend=False,
                                         line=dict(color=TEXT, width=2, dash="dash"), hoverinfo="skip"))
            fig.update_layout(title="Weight vs height", legend=dict(orientation="h", y=-0.2, x=0))
            show(fig, 420)
        with c2:
            fig = bar_counts(m, "type_cat", col_name)
            fig.update_layout(title=f"Average {col_name} ({unit})")
            if not is_h:
                fig.update_traces(marker_color=YELLOW)
            show(fig, 420)
        if len(f) >= 3:
            r = f["weight"].corr(f["height"])
            strength = "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak"
            direction = "positive" if r >= 0 else "negative"
            take = f"There is a {strength} {direction} relationship between height and weight."
            if r >= 0.4:
                take += " Taller Pokémon tend to be heavier, but some outliers show that the relationship is not perfect."
            insight(cards, "Height–weight correlation", f"r = {r:.2f}",
                    "Dashed line shows the overall trend", take)
        ranked = m.sort_values(ascending=False)
        top = ranked.index[0]
        other = ranked.index[1] if len(ranked) > 1 else None
        insight(cards, f"{adj.title()} type category", f"{top} · {ranked.iloc[0]} {unit}",
                f"{other}: {ranked.iloc[1]} {unit}" if other else "Only one type category in this selection",
                f"{top} Pokémon are {adj} than {other} Pokémon on average." if other
                else "Widen the filters to compare single and dual type.")


# =============== ML Model ===============
# Same as the notebook: Linear Regression, 70% train / 30% test, random_state=42.
@st.cache_data
def fit_model():
    X, Y = df[STATS], df["base_experience"]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    model = LinearRegression().fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    sizes, tr, va = learning_curve(model, x_train, y_train, cv=10)
    return dict(
        r2=r2_score(y_test, y_pred), mse=mse, rmse=float(np.sqrt(mse)),
        res=pd.DataFrame({"pokemon": df.loc[y_test.index, "pokemon"], "Actual": y_test, "Predicted": y_pred}),
        lc=pd.DataFrame({"Training set size": sizes, "Training score": tr.mean(axis=1), "Validation score": va.mean(axis=1)}),
    )


with tabs[5]:
    M = fit_model()
    st.markdown("**Linear Regression** · features: the six battle stats · target: base experience · "
                "70% training / 30% testing · random_state = 42")
    k = st.columns(3)
    k[0].metric("R2", f"{M['r2']:.2f}")
    k[1].metric("Mean squared error", f"{M['mse']:.2f}")
    k[2].metric("Root mean squared error", f"{M['rmse']:.2f}")
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Actual vs predicted")
        res = M["res"]
        lo, hi = float(min(res["Actual"].min(), res["Predicted"].min())), float(max(res["Actual"].max(), res["Predicted"].max()))
        fig = px.scatter(res, x="Actual", y="Predicted", hover_name="pokemon")
        fig.update_traces(marker=dict(color=BLUE, size=7, opacity=0.75))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=RED, dash="dash"), name="Perfect prediction"))
        show(fig, 420)
    with c2:
        st.subheader("Learning curve")
        lc = M["lc"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lc["Training set size"], y=lc["Training score"], mode="lines+markers", name="Training score", line=dict(color=BLUE)))
        fig.add_trace(go.Scatter(x=lc["Training set size"], y=lc["Validation score"], mode="lines+markers", name="Validation score", line=dict(color=YELLOW)))
        fig.update_xaxes(title="Training set size")
        fig.update_yaxes(title="R2 score")
        show(fig, 420)
    summary(f"The model explains about {M['r2'] * 100:.0f}% of the differences in base experience on Pokémon it did not train on, "
            f"and its predictions are off by about {M['rmse']:.0f} experience points on average (RMSE). "
            "In the actual vs predicted plot, points close to the dashed line are well predicted.")