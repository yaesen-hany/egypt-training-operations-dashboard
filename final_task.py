# =============================================================================
# Egypt Training Program Dashboard
# Libraries: streamlit, pandas, numpy, altair, plotly
# =============================================================================
import altair as alt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =============================================================================
# SECTION 1: PAGE CONFIGURATION (first Streamlit statement)
# =============================================================================
st.set_page_config(
    page_title="Egypt Training Program Dashboard",
    page_icon="📊",
    layout="wide",
)

# =============================================================================
# SECTION 2: CONSTANTS
# =============================================================================
START_DATE = "2026-09-01"
END_DATE = "2026-12-31"
N_RECORDS = 60
TRACKS = ["AI", "Data", "Coding"]
CHANNELS = ["Online", "Offline", "Hybrid"]
METRIC_OPTIONS = ["profit", "revenue", "participants"]

# (city, governorate, latitude, longitude)
CITIES = [
    ("Cairo", "Cairo", 30.0444, 31.2357),
    ("Giza", "Giza", 30.0131, 31.2089),
    ("Alexandria", "Alexandria", 31.2001, 29.9187),
    ("Aswan", "Aswan", 24.0889, 32.8998),
    ("Mansoura", "Dakahlia", 31.0409, 31.3785),
    ("Assiut", "Assiut", 27.1809, 31.1837),
    ("Tanta", "Gharbia", 30.7865, 31.0004),
    ("Luxor", "Luxor", 25.6872, 32.6396),
]


# =============================================================================
# SECTION 3: DATA GENERATION (deterministic, cached)
# =============================================================================
@st.cache_data
def load_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = N_RECORDS

    # Dates: exactly n records, sorted, pinned to the first and last day
    all_days = pd.date_range(START_DATE, END_DATE, freq="D")
    interior = all_days[1:-1]
    idx = np.sort(rng.choice(len(interior), size=n - 2, replace=False))
    dates = pd.DatetimeIndex([all_days[0], *interior[idx], all_days[-1]])

    # Cities (paired governorate + coordinates)
    city_idx = rng.integers(0, len(CITIES), size=n)
    city = np.array([CITIES[i][0] for i in city_idx])
    governorate = np.array([CITIES[i][1] for i in city_idx])
    latitude = np.array([CITIES[i][2] for i in city_idx])
    longitude = np.array([CITIES[i][3] for i in city_idx])

    # Categorical + numeric fields
    track = rng.choice(TRACKS, size=n)
    channel = rng.choice(CHANNELS, size=n)
    participants = rng.integers(20, 61, size=n)
    completed = np.rint(participants * rng.uniform(0.55, 1.0, size=n)).astype(int)
    completed = np.clip(completed, 0, participants)
    satisfaction = np.round(rng.uniform(3.0, 5.0, size=n), 1)

    # Financials: revenue > cost > 0 guaranteed
    cost = (participants * rng.integers(150, 401, size=n)).astype(int)
    revenue = (cost * rng.uniform(1.15, 1.9, size=n)).astype(int)
    revenue = np.maximum(revenue, cost + 1)

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "city": city,
            "governorate": governorate,
            "latitude": latitude,
            "longitude": longitude,
            "track": track,
            "channel": channel,
            "participants": participants,
            "completed": completed,
            "satisfaction_score": satisfaction,
            "cost": cost,
            "revenue": revenue,
        }
    )

    # Derived columns (zero-division safe)
    p = df["participants"].to_numpy()
    safe_den = np.where(p > 0, p, 1)
    df["completion_rate"] = np.where(p > 0, df["completed"].to_numpy() / safe_den, 0.0)
    df["profit"] = df["revenue"] - df["cost"]
    df["month"] = df["date"].dt.to_period("M").astype(str)

    return df.copy()


# =============================================================================
# SECTION 4: SIDEBAR FILTER PIPELINE
# =============================================================================
def render_sidebar_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    min_d = df["date"].min().date()
    max_d = df["date"].max().date()

    selected = st.sidebar.date_input(
        "Date range",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d,
        key="filter_date_range",
    )
    if isinstance(selected, (tuple, list)) and len(selected) == 2:
        start_d, end_d = selected
    else:
        start_d, end_d = min_d, max_d  # graceful fallback

    track_opts = sorted(df["track"].unique().tolist())
    city_opts = sorted(df["city"].unique().tolist())
    channel_opts = sorted(df["channel"].unique().tolist())

    sel_tracks = st.sidebar.multiselect(
        "Track", options=track_opts, default=track_opts, key="filter_multiselect_track"
    )
    sel_cities = st.sidebar.multiselect(
        "City", options=city_opts, default=city_opts, key="filter_multiselect_city"
    )
    sel_channels = st.sidebar.multiselect(
        "Channel",
        options=channel_opts,
        default=channel_opts,
        key="filter_multiselect_channel",
    )
    min_sat = st.sidebar.slider(
        "Minimum satisfaction",
        min_value=3.0,
        max_value=5.0,
        value=3.0,
        step=0.1,
        key="filter_slider_min_satisfaction",
    )
    min_sat = round(float(min_sat), 1)

    # Overview metric selector lives in the sidebar so it is always rendered,
    # even when st.stop() fires, which keeps its session state alive.
    st.sidebar.divider()
    st.sidebar.header("Overview chart")
    st.sidebar.selectbox(
        "Bar chart metric",
        options=METRIC_OPTIONS,
        format_func=lambda m: m.capitalize(),
        key="overview_bar_metric",
    )

    # Filter convergence (empty multiselect => isin([]) => zero rows)
    mask = (
        (df["date"] >= pd.Timestamp(start_d))
        & (df["date"] <= pd.Timestamp(end_d))
        & df["track"].isin(sel_tracks)
        & df["city"].isin(sel_cities)
        & df["channel"].isin(sel_channels)
        & (df["satisfaction_score"] >= min_sat - 1e-9)
    )
    return df.loc[mask].copy()


# =============================================================================
# SECTION 5: KPI HELPERS
# =============================================================================
def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) > 0 else 0.0


def render_kpis(fdf: pd.DataFrame, df: pd.DataFrame) -> None:
    total_participants = int(fdf["participants"].sum())
    total_profit = int(fdf["profit"].sum())
    avg_completion = safe_mean(fdf["completion_rate"])

    # Per-row normalized benchmarks (fair subset vs. global comparison)
    d_participants = safe_mean(fdf["participants"]) - safe_mean(df["participants"])
    d_completion = (avg_completion - safe_mean(df["completion_rate"])) * 100
    d_profit = safe_mean(fdf["profit"]) - safe_mean(df["profit"])

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Total Participants",
        f"{total_participants:,}",
        delta=f"{d_participants:+.1f} per record vs. global avg",
    )
    c2.metric(
        "Avg Completion Rate",
        f"{avg_completion * 100:.1f}%",
        delta=f"{d_completion:+.1f} pp vs. global avg",
    )
    c3.metric(
        "Total Profit",
        f"EGP {total_profit:,}",
        delta=f"{d_profit:+,.0f} EGP per record vs. global avg",
    )


# =============================================================================
# SECTION 6: TAB RENDERERS
# =============================================================================
def render_overview(fdf: pd.DataFrame, df: pd.DataFrame) -> None:
    render_kpis(fdf, df)
    st.divider()

    metric = st.session_state["overview_bar_metric"]
    left, right = st.columns(2)

    with left:
        st.subheader(f"{metric.capitalize()} by City")
        by_city = fdf.groupby("city", as_index=False)[metric].sum()
        st.bar_chart(by_city, x="city", y=metric, use_container_width=True)

    with right:
        st.subheader("Participants by Track")
        by_track = fdf.groupby("track", as_index=False)["participants"].sum()
        fig = px.pie(by_track, names="track", values="participants", hole=0.5)
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True, key="overview_donut_chart")


def render_trends(fdf: pd.DataFrame) -> None:
    st.subheader("Monthly Participants")
    monthly = fdf.groupby("month")["participants"].sum().sort_index()
    st.line_chart(monthly, use_container_width=True)

    st.subheader("Cumulative Participants by Track")
    pivot = (
        fdf.pivot_table(
            index="date", columns="track", values="participants", aggfunc="sum"
        )
        .fillna(0)
        .sort_index()
        .cumsum()
    )
    st.area_chart(pivot, use_container_width=True)

    st.subheader("Participants vs. Completed by Track")
    by_track = fdf.groupby("track", as_index=False)[["participants", "completed"]].sum()
    melted = pd.melt(
        by_track,
        id_vars="track",
        value_vars=["participants", "completed"],
        var_name="metric",
        value_name="value",
    )
    chart = (
        alt.Chart(melted)
        .mark_bar()
        .encode(
            x=alt.X("track:N", title=None),
            xOffset="metric:N",
            y=alt.Y("value:Q", title="Count"),
            color=alt.Color("metric:N", title="Metric"),
            tooltip=["track:N", "metric:N", "value:Q"],
        )
    )
    st.altair_chart(chart, use_container_width=True, key="trends_grouped_bar")


def render_geography(fdf: pd.DataFrame) -> None:
    geo = fdf.groupby(["city", "governorate", "latitude", "longitude"], as_index=False).agg(
        sessions=("date", "count"),
        participants=("participants", "sum"),
        completed=("completed", "sum"),
        profit=("profit", "sum"),
        avg_satisfaction=("satisfaction_score", "mean"),
    )
    safe_den = geo["participants"].where(geo["participants"] > 0, 1)
    geo["completion_pct"] = np.where(
        geo["participants"] > 0, geo["completed"] / safe_den * 100, 0.0
    ).round(1)
    geo["avg_satisfaction"] = geo["avg_satisfaction"].round(2)
    geo = geo.rename(columns={"latitude": "lat", "longitude": "lon"})
    geo["map_size"] = geo["participants"].astype(float) * 40.0  # meters, scaled by participants

    st.subheader("Coverage Map")
    st.map(geo, latitude="lat", longitude="lon", size="map_size", use_container_width=True)

    st.subheader("City Summary")
    table = geo[
        ["city", "governorate", "sessions", "participants", "completion_pct", "profit", "avg_satisfaction"]
    ].sort_values("participants", ascending=False)
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        key="geo_summary_table",
        column_config={
            "completion_pct": st.column_config.NumberColumn("Completion %", format="%.1f%%"),
            "profit": st.column_config.NumberColumn("Profit (EGP)", format="%d"),
        },
    )


def render_explorer(fdf: pd.DataFrame) -> None:
    rows, cols = fdf.shape
    m1, m2 = st.columns(2)
    m1.metric("Rows", f"{rows:,}")
    m2.metric("Columns", f"{cols:,}")

    st.subheader("Head")
    st.dataframe(fdf.head(), use_container_width=True, key="explorer_head_table")

    st.subheader("Describe")
    st.dataframe(fdf.describe().round(2), use_container_width=True, key="explorer_describe_table")

    st.subheader("Full Data")
    st.dataframe(fdf, use_container_width=True, key="explorer_full_table")

    st.subheader("Editable Copy")
    edited = st.data_editor(
        fdf.copy(),
        num_rows="fixed",
        use_container_width=True,
        key="explorer_data_editor",
    )

    csv_bytes = edited.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="training_activity.csv",
        mime="text/csv",
        key="explorer_download_csv",
    )


# =============================================================================
# SECTION 7: MAIN
# =============================================================================
def main() -> None:
    if "overview_bar_metric" not in st.session_state:
        st.session_state["overview_bar_metric"] = "profit"

    st.title("📊 Egypt Training Program Dashboard")
    st.caption("Performance, trends, and geographic coverage across cities and channels.")

    df = load_data()
    fdf = render_sidebar_and_filter(df)

    # Fail-safe guard: nothing below runs on an empty selection
    if fdf.empty:
        st.warning("No records match the current filters. Adjust the sidebar to continue.")
        st.stop()

    tab_overview, tab_trends, tab_geo, tab_data = st.tabs(
        ["Overview", "Trends", "Geography", "Data Explorer"]
    )
    with tab_overview:
        render_overview(fdf, df)
    with tab_trends:
        render_trends(fdf)
    with tab_geo:
        render_geography(fdf)
    with tab_data:
        render_explorer(fdf)


main()