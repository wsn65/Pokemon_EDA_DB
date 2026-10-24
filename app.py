import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Pokémon Data Insights", page_icon="⚡", layout="wide")

# ---------- Style ----------
st.markdown("""
<style>
.stApp { background-color: #172554; }
h1, h2, h3 { color: #FFCB05 !important; }
[data-testid="stMetric"] {
    background: #1e3a8a; border: 1px solid #FFCB05;
    border-radius: 12px; padding: 14px;
}
[data-testid="stMetricValue"] { color: #FFCB05; }
hr { border-color: #FFCB05 !important; }
</style>
""", unsafe_allow_html=True)

# ---------- Data ----------
@st.cache_data
def load():
    df = pd.read_csv("clean_pokemon.csv")
    if "avg_stat" not in df:
        stats = ["hp", "attack", "defense", "special_attack", "special_defense", "speed"]
        df["avg_stat"] = df[stats].mean(axis=1)
    if "type_cat" not in df:
        df["type_cat"] = df["type_2"].isna().map({True: "Single Type", False: "Dual Type"})
    if "exp_group" not in df:
        ratio = df["base_experience"] / df["avg_stat"]
        df["exp_group"] = pd.cut(ratio, [0, 1.7, 2.4, 100], labels=["Low", "Medium", "High"])
    return df

df = load()

# ---------- Header ----------
logo, title = st.columns([1, 12], vertical_alignment="center")
logo.image("https://upload.wikimedia.org/wikipedia/commons/5/53/Pok%C3%A9_Ball_icon.svg", width=80)
title.title("Pokémon Data Insights")
title.caption("Explore the Pokémon world through data")
st.divider()

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
gens = sorted(df["generation_id"].dropna().unique())
sel_gen = st.sidebar.multiselect("Generation", gens, default=gens)
sel_type = st.sidebar.multiselect("Type category", df["type_cat"].unique(), default=list(df["type_cat"].unique()))

f = df[df["generation_id"].isin(sel_gen) & df["type_cat"].isin(sel_type)]

# ---------- KPI cards ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Pokémon", f"{len(f):,}")
c2.metric("Avg stat", f"{f['avg_stat'].mean():.1f}")
c3.metric("Avg base exp", f"{f['base_experience'].mean():.0f}")
c4.metric("Dual type", f"{(f['type_cat'] == 'Dual Type').mean():.0%}")

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["Distributions", "Relationships", "Types"])

with tab1:
    stat = st.selectbox("Stat", ["hp", "attack", "defense", "special_attack", "special_defense", "speed", "base_experience"])
    fig = px.histogram(f, x=stat, nbins=30, marginal="box", color_discrete_sequence=["#FFCB05"])
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = px.scatter(f, x="avg_stat", y="base_experience", color="exp_group",
                     hover_name="pokemon", opacity=0.75,
                     title="Base experience vs average stat")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Three distinct lines: each group has its own base experience rate.")

with tab3:
    left, right = st.columns(2)
    counts = f["type_cat"].value_counts().reset_index()
    left.plotly_chart(px.bar(counts, x="type_cat", y="count", title="Single vs dual type",
                             color_discrete_sequence=["#FFCB05"]), use_container_width=True)
    right.plotly_chart(px.box(f, x="type_cat", y="avg_stat", title="Average stat by type category",
                              color_discrete_sequence=["#FFCB05"]), use_container_width=True)

# ---------- Data ----------
with st.expander("View data"):
    st.dataframe(f, use_container_width=True)