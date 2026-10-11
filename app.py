import importlib.util
import os
import subprocess
import sys
from io import BytesIO


def _ensure_runtime_dependencies() -> None:
    core_packages = {
        "streamlit": "streamlit==1.40.2",
        "pandas": "pandas==2.2.3",
        "numpy": "numpy==2.1.3",
        "altair": "altair==5.4.1",
        "plotly": "plotly==5.24.1",
    }
    optional_packages = {
        "folium": "folium==0.18.0",
        "streamlit_folium": "streamlit-folium==0.24.0",
        "reportlab": "reportlab==4.2.5",
        "sklearn": "scikit-learn==1.5.2",
        "seaborn": "seaborn==0.13.2",
        "PIL": "pillow==11.3.0",
    }
    missing_core = [
        requirement
        for module, requirement in core_packages.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing_core:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_core])

    missing_optional = [
        requirement
        for module, requirement in optional_packages.items()
        if importlib.util.find_spec(module) is None
    ]
    for requirement in missing_optional:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])
        except subprocess.CalledProcessError:
            print(f"Optional package could not be installed: {requirement}")


_ensure_runtime_dependencies()

import altair as alt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Optional GenAI Integration
try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    GENAI_AVAILABLE = False

try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_MAP_AVAILABLE = True
except ImportError:
    folium = None
    st_folium = None
    FOLIUM_MAP_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None
    PIL_AVAILABLE = False

# =============================================================================
# SECTION 1: PAGE CONFIGURATION (Strictly the first Streamlit call)
# =============================================================================
st.set_page_config(
    page_title="Egypt Training Operations Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# SECTION 2: CONSTANTS & REFERENCE GEODATA
# =============================================================================
# Optional: Set the GEMINI_API_KEY environment variable or add it to
# .streamlit/secrets.toml. If left unset, the assistant automatically
# uses the local records engine seamlessly.
_GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")

_TRACKS: list[str] = ["AI", "Data", "Coding"]
_CHANNELS: list[str] = ["Online", "Offline", "Hybrid"]
_METRICS: list[str] = ["profit", "revenue", "participants"]

# Canonical paired Egyptian geodata: (City, Governorate, Latitude, Longitude)
_CITIES_REF: list[tuple[str, str, float, float]] = [
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
# SECTION 3: SESSION STATE INITIALIZATION
# =============================================================================
if "overview_bar_metric" not in st.session_state:
    st.session_state["overview_bar_metric"] = "profit"

if "geo_map_engine" not in st.session_state:
    st.session_state["geo_map_engine"] = "Native Streamlit Map"

if "last_filter_sig" not in st.session_state:
    st.session_state["last_filter_sig"] = None
if "last_balloons_sig" not in st.session_state:
    st.session_state["last_balloons_sig"] = None

if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = [
        {
            "role": "assistant",
            "content": (
                "Welcome! 👋 I am your Operations Intelligence Assistant. "
                "I can analyze real-time cohort performance, curriculum tracks, "
                "geographic locations, and delivery channels using active dashboard metrics. "
                "How can I assist you today?"
            ),
        }
    ]


# =============================================================================
# SECTION 4: DATA INGESTION & ROBUST FALLBACK GENERATOR
# =============================================================================
@st.cache_data
def load_dataset() -> pd.DataFrame:
    file_path = "training_activity.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df["date"] = pd.to_datetime(df["date"])
    else:
        rng = np.random.default_rng(42)
        n_rows = 60
        calendar_days = pd.date_range("2026-09-01", "2026-12-31", freq="D")
        interior = calendar_days[1:-1]
        sampled_interior = np.sort(rng.choice(len(interior), size=n_rows - 2, replace=False))
        session_dates = pd.DatetimeIndex([calendar_days[0], *interior[sampled_interior], calendar_days[-1]])

        city_picks = rng.integers(0, len(_CITIES_REF), size=n_rows)
        participants = rng.integers(20, 61, size=n_rows)
        cost = (participants * rng.integers(160, 390, size=n_rows)).astype(int)
        revenue = np.maximum((cost * rng.uniform(1.18, 1.85, size=n_rows)).astype(int), cost + 1)
        df = pd.DataFrame({
            "date": pd.to_datetime(session_dates),
            "city": np.array([_CITIES_REF[i][0] for i in city_picks]),
            "governorate": np.array([_CITIES_REF[i][1] for i in city_picks]),
            "latitude": np.array([_CITIES_REF[i][2] for i in city_picks]),
            "longitude": np.array([_CITIES_REF[i][3] for i in city_picks]),
            "track": rng.choice(_TRACKS, size=n_rows),
            "channel": rng.choice(_CHANNELS, size=n_rows),
            "participants": participants,
            "completed": np.clip(np.rint(participants * rng.uniform(0.55, 1.0, size=n_rows)).astype(int), 0, participants),
            "satisfaction_score": np.round(rng.uniform(3.0, 5.0, size=n_rows), 1),
            "cost": cost,
            "revenue": revenue,
        })

    p = df["participants"].to_numpy()
    df["completion_rate"] = np.where(p > 0, df["completed"].to_numpy() / np.where(p > 0, p, 1), 0.0)
    df["profit"] = df["revenue"] - df["cost"]
    df["profit_margin_pct"] = np.where(df["revenue"] > 0, (df["profit"] / df["revenue"]) * 100, 0.0)
    df["cost_per_participant"] = np.where(p > 0, df["cost"] / np.where(p > 0, p, 1), 0.0)
    df["month"] = df["date"].dt.to_period("M").astype(str)

    return df.sort_values("date").reset_index(drop=True).copy()


# =============================================================================
# SECTION 5: SIDEBAR FILTERS & RUNTIME CONVERGENCE
# =============================================================================
def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.header("🎯 Operational Filters")

        min_calendar = df["date"].min().date()
        max_calendar = df["date"].max().date()
        all_tracks = sorted(df["track"].unique().tolist())
        all_cities = sorted(df["city"].unique().tolist())
        all_channels = sorted(df["channel"].unique().tolist())

        def reset_filters() -> None:
            st.session_state["filter_date_range"] = (min_calendar, max_calendar)
            st.session_state["filter_multiselect_tracks"] = all_tracks
            st.session_state["filter_multiselect_cities"] = all_cities
            st.session_state["filter_multiselect_channels"] = all_channels
            st.session_state["filter_min_satisfaction_score"] = 3

        st.button(
            "↺ Clear filters and show all data",
            key="reset_all_filters_button",
            use_container_width=True,
            on_click=reset_filters,
            help="Clear the active restrictions and restore the complete dataset.",
        )

        date_selection = st.date_input(
            "Date Range",
            value=(min_calendar, max_calendar),
            min_value=min_calendar,
            max_value=max_calendar,
            key="filter_date_range",
            help="Select training start and end dates.",
        )

        if isinstance(date_selection, (tuple, list)) and len(date_selection) == 2:
            start_date, end_date = date_selection
        else:
            start_date, end_date = min_calendar, max_calendar

        st.divider()

        selected_tracks = st.multiselect(
            "Training Track",
            options=all_tracks,
            default=all_tracks,
            key="filter_multiselect_tracks",
        )

        selected_cities = st.multiselect(
            "Target City",
            options=all_cities,
            default=all_cities,
            key="filter_multiselect_cities",
        )

        selected_channels = st.multiselect(
            "Delivery Channel",
            options=all_channels,
            default=all_channels,
            key="filter_multiselect_channels",
        )

        st.divider()

        min_satisfaction = st.slider(
            "Minimum Satisfaction Score",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
            format="%d ★",
            key="filter_min_satisfaction_score",
            help="Filter sessions by minimum whole-star satisfaction rating.",
        )

        st.divider()
        st.header("⚙️ Metric Configuration")

        try:
            st.segmented_control(
                "Primary Chart Metric",
                options=_METRICS,
                key="overview_bar_metric",
            )
        except AttributeError:
            st.selectbox(
                "Primary Chart Metric",
                options=_METRICS,
                key="overview_bar_metric",
            )

        st.caption("Filters apply concurrently across all dashboard tabs.")

    if not selected_tracks or not selected_cities or not selected_channels:
        fdf = df.iloc[0:0].copy()
    else:
        mask = (
            (df["date"] >= pd.Timestamp(start_date))
            & (df["date"] <= pd.Timestamp(end_date))
            & df["track"].isin(selected_tracks)
            & df["city"].isin(selected_cities)
            & df["channel"].isin(selected_channels)
            & (df["satisfaction_score"] >= min_satisfaction)
        )
        fdf = df.loc[mask].copy()

    current_sig = (
        str(start_date), str(end_date), tuple(selected_tracks),
        tuple(selected_cities), tuple(selected_channels), min_satisfaction,
    )
    if st.session_state["last_filter_sig"] is not None and current_sig != st.session_state["last_filter_sig"]:
        st.toast(f"Active filter updated: {len(fdf)} sessions matched.", icon="⚡")
    st.session_state["last_filter_sig"] = current_sig
    st.sidebar.caption(f"**{len(fdf):,}** matching sessions")

    return fdf


# =============================================================================
# SECTION 6: EXECUTIVE KPI STRIP
# =============================================================================
def _safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) > 0 else 0.0


def render_executive_kpis(fdf: pd.DataFrame, df: pd.DataFrame) -> None:
    total_participants = int(fdf["participants"].sum())
    avg_completion = _safe_mean(fdf["completion_rate"])
    total_profit = int(fdf["profit"].sum())
    total_revenue = int(fdf["revenue"].sum())
    overall_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0.0

    global_margin = (df["profit"].sum() / df["revenue"].sum() * 100) if df["revenue"].sum() > 0 else 0.0

    delta_participants = _safe_mean(fdf["participants"]) - _safe_mean(df["participants"])
    delta_completion = (avg_completion - _safe_mean(df["completion_rate"])) * 100
    delta_profit = _safe_mean(fdf["profit"]) - _safe_mean(df["profit"])
    delta_margin = overall_margin - global_margin

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1.container(border=True):
        st.metric(
            label="Total Enrollment",
            value=f"{total_participants:,}",
            delta=f"{delta_participants:+.1f} / session vs avg",
        )
    with kpi2.container(border=True):
        st.metric(
            label="Avg Completion Rate",
            value=f"{avg_completion * 100:.1f}%",
            delta=f"{delta_completion:+.1f}% pts vs avg",
        )
    with kpi3.container(border=True):
        st.metric(
            label="Net Profit Generated",
            value=f"EGP {total_profit:,}",
            delta=f"EGP {delta_profit:+,.0f} / session vs avg",
        )
    with kpi4.container(border=True):
        st.metric(
            label="Operating Margin",
            value=f"{overall_margin:.1f}%",
            delta=f"{delta_margin:+.1f}% pts vs avg",
        )


# =============================================================================
# SECTION 7: TAB RENDERERS
# =============================================================================

# -----------------------------------------------------------------------------
# TAB 1: OVERVIEW & EXECUTIVE METRICS
# -----------------------------------------------------------------------------
def render_tab_overview(fdf: pd.DataFrame, df: pd.DataFrame) -> None:
    st.subheader("Executive Operations Summary")
    render_executive_kpis(fdf, df)

    st.markdown("---")

    active_metric = st.session_state.get("overview_bar_metric", "profit")
    if active_metric is None or active_metric not in _METRICS:
        active_metric = "profit"

    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        with st.container(border=True):
            st.subheader(f"Total {active_metric.capitalize()} by City")
            city_metric_data = fdf.groupby("city", as_index=False)[active_metric].sum()
            city_metric_data = city_metric_data.sort_values(by=active_metric, ascending=False)
            st.bar_chart(city_metric_data, x="city", y=active_metric, use_container_width=True)

    with row1_col2:
        with st.container(border=True):
            st.subheader("Participant Distribution by Track")
            track_dist = fdf.groupby("track", as_index=False)["participants"].sum()
            donut_fig = px.pie(
                track_dist,
                names="track",
                values="participants",
                hole=0.52,
                color_discrete_sequence=["#00F5FF", "#00FFA3", "#FF007A"],
                template="plotly_dark",
            )
            donut_fig.update_layout(
                margin=dict(t=20, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            )
            donut_fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(donut_fig, use_container_width=True, key="plotly_track_donut")

    st.markdown("---")
    st.subheader("Operational Benchmarks & Channel Dynamics")

    snap1, snap2, snap3 = st.columns(3)
    with snap1.container(border=True):
        top_city = fdf.groupby("city")["participants"].sum().idxmax()
        st.metric("Top City by Headcount", top_city)
    with snap2.container(border=True):
        top_track = fdf.groupby("track")["profit"].sum().idxmax()
        st.metric("Most Profitable Track", top_track)
    with snap3.container(border=True):
        avg_rating = _safe_mean(fdf["satisfaction_score"])
        st.metric("Cohort Satisfaction Average", f"{avg_rating:.2f} / 5.00 ★")

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        with st.container(border=True):
            st.subheader("Participants by Delivery Channel")
            channel_dist = fdf.groupby("channel", as_index=False)["participants"].sum()
            st.bar_chart(channel_dist, x="channel", y="participants", use_container_width=True)

    with row2_col2:
        with st.container(border=True):
            st.subheader("Financial Comparison: Revenue vs Cost by City")
            city_fin = fdf.groupby("city", as_index=False)[["revenue", "cost"]].sum()
            melted_fin = pd.melt(
                city_fin,
                id_vars="city",
                value_vars=["revenue", "cost"],
                var_name="Account",
                value_name="Amount_EGP",
            )
            fin_bar_fig = px.bar(
                melted_fin,
                x="city",
                y="Amount_EGP",
                color="Account",
                barmode="group",
                color_discrete_map={"revenue": "#00FFA3", "cost": "#FF007A"},
                template="plotly_dark",
            )
            fin_bar_fig.update_layout(
                margin=dict(t=20, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            )
            st.plotly_chart(fin_bar_fig, use_container_width=True, key="plotly_city_financials")


# -----------------------------------------------------------------------------
# TAB 2: TEMPORAL TRENDS
# -----------------------------------------------------------------------------
def render_tab_trends(fdf: pd.DataFrame) -> None:
    st.subheader("Enrollment Trajectory & Pipeline Performance")

    month_index = pd.period_range("2026-09", "2026-12", freq="M").astype(str)
    monthly_intake = (
        fdf.groupby("month")["participants"]
        .sum()
        .sort_index()
        .reindex(month_index, fill_value=0)
        .rename("Enrolled Participants")
    )
    with st.container(border=True):
        st.markdown("#### Monthly Participant Intake")
        st.line_chart(monthly_intake, use_container_width=True)

    st.markdown("---")

    st.markdown("#### Track Intake Across Months (Multi-Series Trend)")
    monthly_track_pivot = (
        fdf.pivot_table(
            index="month",
            columns="track",
            values="participants",
            aggfunc="sum",
            fill_value=0,
        )
        .sort_index()
        .reindex(index=month_index, fill_value=0)
        .reindex(columns=_TRACKS, fill_value=0)
    )
    with st.container(border=True):
        if int(monthly_track_pivot.to_numpy().sum()) == 0:
            st.info("No monthly track activity is available for these filters.")
        else:
            st.line_chart(monthly_track_pivot, use_container_width=True)

    st.markdown("---")

    st.markdown("#### Cumulative Intake by Track")
    daily_index = pd.date_range("2026-09-01", "2026-12-31", freq="D")
    cumulative_participants = (
        fdf.pivot_table(
            index="date",
            columns="track",
            values="participants",
            aggfunc="sum",
        )
        .fillna(0)
        .sort_index()
        .reindex(daily_index, fill_value=0)
        .cumsum()
        .reindex(columns=_TRACKS, fill_value=0)
    )
    with st.container(border=True):
        st.area_chart(cumulative_participants, use_container_width=True)

    st.markdown("---")

    st.markdown("#### Declarative Altair Grouped Chart: Enrolled vs Completed by Track")
    track_performance = (
        fdf.groupby("track")[["participants", "completed"]]
        .sum()
        .reindex(_TRACKS, fill_value=0)
        .rename_axis("track")
        .reset_index()
    )
    melted_performance = pd.melt(
        track_performance,
        id_vars=["track"],
        value_vars=["participants", "completed"],
        var_name="Metric",
        value_name="Count",
    )

    hover_sel = alt.selection_point(on="mouseover", empty="all")

    altair_grouped = (
        alt.Chart(melted_performance)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .add_params(hover_sel)
        .encode(
            x=alt.X("track:N", title="Training Track"),
            xOffset=alt.XOffset("Metric:N"),
            y=alt.Y("Count:Q", title="Student Headcount", axis=alt.Axis(format=",.0f")),
            color=alt.Color(
                "Metric:N",
                scale=alt.Scale(domain=["participants", "completed"], range=["#00B8FF", "#00FFA3"]),
                title="Status",
            ),
            opacity=alt.condition(hover_sel, alt.value(1.0), alt.value(0.40)),
            tooltip=[
                alt.Tooltip("track:N", title="Track"),
                alt.Tooltip("Metric:N", title="Category"),
                alt.Tooltip("Count:Q", title="Students", format=","),
            ],
        )
        .properties(height=380)
    )
    with st.container(border=True):
        st.altair_chart(altair_grouped, use_container_width=True, key="altair_enrollment_completion")


# -----------------------------------------------------------------------------
# TAB 3: GEOGRAPHIC FOOTPRINT
# -----------------------------------------------------------------------------
def render_tab_geography(fdf: pd.DataFrame) -> None:
    st.subheader("Geographic Distribution Across Egyptian Governorates")

    geo_summary = (
        fdf.groupby(["city", "governorate", "latitude", "longitude"], as_index=False)
        .agg(
            sessions=("date", "count"),
            participants=("participants", "sum"),
            completed=("completed", "sum"),
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
            avg_satisfaction=("satisfaction_score", "mean"),
        )
        .rename(columns={"latitude": "lat", "longitude": "lon"})
    )
    safe_p = geo_summary["participants"].where(geo_summary["participants"] > 0, 1)
    geo_summary["completion_pct"] = np.where(
        geo_summary["participants"] > 0,
        (geo_summary["completed"] / safe_p * 100).round(1),
        0.0,
    )
    geo_summary["avg_satisfaction"] = geo_summary["avg_satisfaction"].round(2)
    with st.container(border=True):
        st.subheader("Training Activity Map")
        if FOLIUM_MAP_AVAILABLE:
            try:
                st.caption("Zoom in for more geographic detail; marker size remains readable.")
                city_map = folium.Map(
                    location=[28.5, 31.0],
                    zoom_start=6,
                    tiles="OpenStreetMap",
                    control_scale=True,
                )
                max_participants = max(int(geo_summary["participants"].max()), 1)
                for _, row in geo_summary.iterrows():
                    marker_radius = 7 + 11 * int(row["participants"]) / max_participants
                    folium.CircleMarker(
                        location=[float(row["lat"]), float(row["lon"])],
                        radius=marker_radius,
                        color="#287D78",
                        weight=2,
                        fill=True,
                        fill_color="#48A999",
                        fill_opacity=0.55,
                        tooltip=f"{row['city']} · {int(row['participants']):,} participants",
                        popup=(
                            f"{row['city']} ({row['governorate']}) | "
                            f"Sessions: {int(row['sessions'])} | "
                            f"Completion: {row['completion_pct']:.1f}%"
                        ),
                    ).add_to(city_map)
                st_folium(
                    city_map,
                    height=480,
                    use_container_width=True,
                    returned_objects=[],
                    key="training_activity_folium_map",
                )
            except Exception:
                st.warning("The interactive map could not be displayed. Showing the native map instead.")
                geo_summary["map_radius"] = geo_summary["participants"].clip(lower=1) * 150
                st.map(
                    geo_summary,
                    latitude="lat",
                    longitude="lon",
                    size="map_radius",
                    use_container_width=True,
                )
        else:
            st.info("Interactive map packages are unavailable. Showing the native map instead.")
            geo_summary["map_radius"] = geo_summary["participants"].clip(lower=1) * 150
            st.map(
                geo_summary,
                latitude="lat",
                longitude="lon",
                size="map_radius",
                use_container_width=True,
            )

    with st.container(border=True):
        st.subheader("Regional Performance Ledger")
        st.dataframe(
            geo_summary[[
                "city", "governorate", "sessions", "participants",
                "completion_pct", "profit", "avg_satisfaction"
            ]].sort_values("participants", ascending=False).reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
            key="geo_city_ledger_table",
            column_config={
                "city": st.column_config.TextColumn("City"),
                "governorate": st.column_config.TextColumn("Governorate"),
                "sessions": st.column_config.NumberColumn("Cohorts"),
                "participants": st.column_config.NumberColumn("Total Students", format="%d"),
                "completion_pct": st.column_config.NumberColumn("Completion %", format="%.1f%%"),
                "profit": st.column_config.NumberColumn("Net Profit", format="EGP %d"),
                "avg_satisfaction": st.column_config.NumberColumn("Satisfaction Rating", format="%.2f ★"),
            },
        )


# -----------------------------------------------------------------------------
# TAB 4: DATA EXPLORER & INSPECTION (Required Inspection Section)
# -----------------------------------------------------------------------------
def _toggle_viewer(label: str, state_key: str, btn_key: str) -> bool:
    if state_key not in st.session_state:
        st.session_state[state_key] = False

    def on_click_flip() -> None:
        st.session_state[state_key] = not st.session_state[state_key]

    is_active = st.session_state[state_key]
    btn_label = f"Hide {label}" if is_active else f"Show {label}"
    st.button(btn_label, key=btn_key, use_container_width=True, on_click=on_click_flip)
    return is_active


def render_tab_explorer(fdf: pd.DataFrame) -> None:
    st.subheader("Dataset Structure & Interactive Inspection")

    total_rows, total_cols = fdf.shape
    dim_col1, dim_col2, dim_col3 = st.columns(3)

    with dim_col1.container(border=True):
        st.metric("Filtered Records (Rows)", f"{total_rows:,}")
    with dim_col2.container(border=True):
        st.metric("Total Schema Dimensions (Cols)", f"{total_cols:,}")
    with dim_col3.container(border=True):
        st.metric("Memory Footprint", f"{fdf.memory_usage(deep=True).sum() / 1024:.1f} KB")

    st.markdown("---")
    st.markdown("#### Inspection Panels")

    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4, gap="small")
    with col_btn1:
        view_head = _toggle_viewer("Preview (head)", "exp_head_vis", "exp_btn_head")
    with col_btn2:
        view_desc = _toggle_viewer("Summary (describe)", "exp_desc_vis", "exp_btn_desc")
    with col_btn3:
        view_full = _toggle_viewer("Full Table", "exp_full_vis", "exp_btn_full")
    with col_btn4:
        view_edit = _toggle_viewer("Editable Sandbox", "exp_edit_vis", "exp_btn_edit")

    if view_head:
        with st.container(border=True):
            st.markdown("##### Preview — First 5 Rows (`df.head()`)")
            st.dataframe(fdf.head(5), use_container_width=True, hide_index=True, key="tbl_head_preview")

    if view_desc:
        with st.container(border=True):
            st.markdown("##### Descriptive Statistics (`df.describe()`)")
            st.dataframe(
                fdf.describe(include="all").astype(str),
                use_container_width=True,
                key="tbl_describe_summary",
            )

    if view_full:
        with st.container(border=True):
            st.markdown("##### Complete Filtered Dataset (`st.dataframe`)")
            st.dataframe(fdf, use_container_width=True, hide_index=True, key="tbl_full_dataset")

    if view_edit:
        with st.container(border=True):
            st.markdown("##### Interactive Editable Sandbox (`st.data_editor`)")
            st.caption("Modifications here are local to the current session.")
            st.data_editor(
                fdf.copy(),
                num_rows="fixed",
                use_container_width=True,
                hide_index=True,
                key="tbl_editable_sandbox",
            )

    st.markdown("---")
    st.markdown("#### Export Filtered Data")
    csv_payload = fdf.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"⬇️ Download Filtered Data as CSV ({len(fdf)} Rows)",
        data=csv_payload,
        file_name="egypt_training_filtered_export.csv",
        mime="text/csv",
        use_container_width=True,
        key="btn_download_filtered_csv",
    )


# -----------------------------------------------------------------------------
# TAB 5: AI INTELLIGENCE ASSISTANT (Seamless AI & Local Records Engine)
# -----------------------------------------------------------------------------
def _resolve_gemini_key() -> str:
    """
    Safely resolves the Gemini API key from environment variable or Streamlit secrets.
    Prevents StreamlitSecretNotFoundError by strictly using safe try-except blocks.
    """
    env_key = _GEMINI_API_KEY.strip()
    if env_key:
        return env_key

    # Safe secrets inspection without throwing exception if file does not exist
    try:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass

    return ""


def render_tab_assistant(fdf: pd.DataFrame) -> None:
    st.subheader("🤖 Operations Intelligence Assistant")
    st.caption("Ask questions about real-time cohort KPIs, tracks, locations, delivery models, or certifications.")

    # 1. Live Aggregated Metric Snapshot
    total_enrollment = int(fdf["participants"].sum())
    total_completers = int(fdf["completed"].sum())
    avg_comp_pct = (total_completers / total_enrollment * 100) if total_enrollment > 0 else 0.0
    total_profit_egp = int(fdf["profit"].sum())
    total_revenue_egp = int(fdf["revenue"].sum())
    top_city = fdf.groupby("city")["participants"].sum().idxmax() if not fdf.empty else "N/A"
    top_track = fdf.groupby("track")["profit"].sum().idxmax() if not fdf.empty else "N/A"

    # 2. Deterministic Local Knowledge Base (Guaranteed Fallback)
    live_summary = (
        f"📊 **Current Filtered Operations Snapshot:**\n\n"
        f"- **Total Active Enrollment:** {total_enrollment:,} students across {len(fdf)} sessions.\n"
        f"- **Overall Completion Rate:** {avg_comp_pct:.1f}% ({total_completers:,} graduates).\n"
        f"- **Financial Overview:** EGP {total_revenue_egp:,} gross revenue with EGP {total_profit_egp:,} net profit.\n"
        f"- **Leading Highlights:** Top city by intake is **{top_city}**, and the most profitable track is **{top_track}**."
    )

    greeting_reply = (
        "Hello! 👋 I am delighted to support your training program operations.\n\n"
        "Here are some strategic areas you can ask me about:\n"
        "- 📊 **Live Performance:** Type 'summary' or 'metrics' for real-time filtered KPIs.\n"
        "- 💡 **Track Details:** Inquire about AI, Data Science, or Coding curriculum goals.\n"
        "- 📍 **Geographic Footprint:** Ask about training coverage in Cairo, Giza, Alexandria, Aswan, etc.\n"
        "- 🎓 **Certificates:** Review graduation requirements and attendance rules.\n"
        "- 💰 **Financial Health:** Inquire about cost drivers, margins, and operational profitability."
    )

    knowledge_repo: dict[str, str] = {
        "greeting": greeting_reply,
        "summary": live_summary,
        "tracks": (
            "💡 **Curriculum Track Architecture:**\n\n"
            "1. **Artificial Intelligence (AI):** Covers machine learning, neural networks, computer vision, and NLP frameworks.\n"
            "2. **Data Science & Analytics (Data):** Emphasizes exploratory data analysis, statistical modeling, SQL, and business intelligence dashboards.\n"
            "3. **Software Engineering (Coding):** Focuses on core programming algorithms, object-oriented design, web backend services, and clean architecture."
        ),
        "geography": (
            "📍 **Geographic Presence:**\n\n"
            "The program spans eight strategic metropolitan hubs across Egypt:\n"
            "- **Greater Cairo:** Cairo and Giza hubs.\n"
            "- **Delta & Mediterranean:** Alexandria, Tanta (Gharbia), and Mansoura (Dakahlia).\n"
            "- **Upper Egypt:** Assiut, Luxor, and Aswan."
        ),
        "channels": (
            "🌐 **Delivery Channel Breakdown:**\n\n"
            "- **Online:** Fully virtual lectures and labs delivering wide geographic accessibility.\n"
            "- **Offline:** In-person immersive workshops hosted at designated regional university technology labs.\n"
            "- **Hybrid:** Blended learning offering digital theory combined with weekend practical capstones."
        ),
        "certificates": (
            "🎓 **Certificate Eligibility Criteria:**\n\n"
            "- Maintain an aggregate attendance record of at least 80% across all scheduled sessions.\n"
            "- Successfully deliver and present the end-of-track capstone project.\n"
            "- Verified digital credentials are issued jointly by the training program administration."
        ),
        "financials": (
            "💰 **Operating Economics & Margins:**\n\n"
            "Each training session operates with strict cost tracking (instructor fees, infrastructure, licensing) "
            "against allocated institutional budgets. Net profit represents gross revenue minus direct operational expenses."
        ),
    }

    keyword_lookup: dict[str, tuple[str, ...]] = {
        "greeting": ("hello", "hi", "hey", "greetings", "good morning", "good afternoon"),
        "summary": ("summary", "stats", "metrics", "kpi", "numbers", "participants", "enrollment", "profit"),
        "tracks": ("track", "ai", "data", "coding", "curriculum", "syllabus", "content"),
        "geography": ("city", "cities", "governorate", "location", "locations", "map", "cairo", "aswan"),
        "channels": ("channel", "channels", "online", "offline", "hybrid", "delivery", "mode"),
        "certificates": ("certificate", "certification", "completion", "pass", "attendance", "graduation"),
        "financials": ("cost", "revenue", "profit", "margin", "money", "fee", "budget"),
    }

    fallback_response = (
        "Thank you for your question! 😊 I don't have specific details on that topic in my local records.\n\n"
        "Feel free to ask about:\n"
        "- Real-time filtered KPI statistics ('summary').\n"
        "- Training tracks ('AI', 'Data', 'Coding').\n"
        "- Regional locations and governorate coverage.\n"
        "- Delivery models (Online, Offline, Hybrid) and certification criteria."
    )

    def generate_local_response(prompt: str) -> str:
        clean_query = prompt.casefold().strip()
        for topic in ("greeting", "summary", "tracks", "geography", "channels", "certificates", "financials"):
            if any(term in clean_query for term in keyword_lookup[topic]):
                return knowledge_repo[topic]
        return fallback_response

    # 3. Seamless Multi-Tier Engine
    def query_assistant(prompt: str) -> str:
        api_key = _resolve_gemini_key()

        if api_key and GENAI_AVAILABLE:
            data_columns = [
                "date", "city", "governorate", "track", "channel",
                "participants", "completed", "satisfaction_score", "cost",
                "revenue", "profit", "profit_margin_pct", "completion_rate",
            ]
            available_columns = [column for column in data_columns if column in fdf.columns]
            filtered_records = fdf[available_columns].to_json(
                orient="records", date_format="iso"
            )
            system_prompt = (
                "You are the Operations Intelligence Assistant for the Egypt Training Program Dashboard. "
                "Answer accurately and concisely using only the filtered session records and context below. "
                "Do not invent values or claim information that is not present. If the records do not answer "
                "the question, say so. Respond in the same language as the user's question.\n"
                f"- Filtered Sessions Count: {len(fdf)}\n"
                f"- Active Enrollment: {total_enrollment:,} students\n"
                f"- Completion Rate: {avg_comp_pct:.1f}%\n"
                f"- Net Profit: EGP {total_profit_egp:,} (Revenue: EGP {total_revenue_egp:,})\n"
                f"- Top City by Enrollment: {top_city}\n"
                f"- Top Profitable Track: {top_track}\n"
                f"- Active Tracks: {', '.join(fdf['track'].unique())}\n"
                f"- Active Cities: {', '.join(fdf['city'].unique())}\n"
                f"- Active Channels: {', '.join(fdf['channel'].unique())}\n"
                f"Filtered session records in JSON: {filtered_records}\n"
                "Use the records to calculate comparisons or answer questions about individual sessions. "
                "If the query is conversational or general, be welcoming and provide clear bullet points."
            )

            candidate_models = [
                "gemini-3.5-flash",
                "gemini-3.8-flash",
                "gemini-3.5-flash-lite",
                "gemini-2.5-flash-lite",
                "gemini-flash-latest",
            ]

            try:
                client = genai.Client(api_key=api_key)
                for model_candidate in candidate_models:
                    try:
                        resp = client.models.generate_content(
                            model=model_candidate,
                            contents=f"System Context:\n{system_prompt}\n\nUser Question:\n{prompt}",
                        )
                        if resp and resp.text:
                            return resp.text
                    except Exception:
                        continue
            except Exception:
                pass

        # Silent, clean fallback to local knowledge engine
        return generate_local_response(prompt)

    # 4. Render Pure Chat Interface
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Ask about training metrics, tracks, locations, or finances...", key="input_assistant_chat")
    if user_prompt:
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing operational data..."):
                reply = query_assistant(user_prompt)
            st.markdown(reply)

        st.session_state["chat_messages"].append({"role": "user", "content": user_prompt})
        st.session_state["chat_messages"].append({"role": "assistant", "content": reply})


# =============================================================================
# SECTION 8: APPLICATION ENTRY POINT
# =============================================================================
@st.cache_data
def build_dashboard_illustration() -> bytes:
    width, height = 960, 340
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)

    for y in range(height):
        blend = y / max(height - 1, 1)
        top = (224, 241, 238)
        bottom = (249, 228, 204)
        color = tuple(int(top[i] * (1 - blend) + bottom[i] * blend) for i in range(3))
        draw.line((0, y, width, y), fill=color)

    draw.ellipse((720, 34, 840, 154), fill=(255, 255, 255))
    draw.polygon([(390, 300), (560, 92), (730, 300)], fill=(188, 142, 95))
    draw.polygon([(560, 92), (730, 300), (625, 300)], fill=(215, 174, 124))
    draw.polygon([(590, 300), (735, 142), (880, 300)], fill=(157, 119, 88))
    draw.polygon([(735, 142), (880, 300), (790, 300)], fill=(190, 151, 111))
    draw.rectangle((0, 300, width, height), fill=(54, 142, 137))
    draw.line((0, 317, width, 317), fill=(111, 190, 179), width=5)
    draw.line((0, 334, width, 334), fill=(37, 117, 119), width=4)

    font = ImageFont.load_default()
    draw.text((48, 72), "EGYPT  /  TRAINING  /  INSIGHTS", fill=(38, 67, 68), font=font)
    draw.text((48, 112), "Learning across every region", fill=(38, 67, 68), font=font)
    draw.text((48, 146), "People  |  Progress  |  Performance", fill=(50, 107, 105), font=font)

    image_buffer = BytesIO()
    image.save(image_buffer, format="PNG")
    return image_buffer.getvalue()


def main() -> None:
    with st.container(border=True):
        title_col, illustration_col = st.columns([1.35, 1])
        with title_col:
            st.title("📈 Egypt Training Program Operations Dashboard")
            st.caption("Samsung Innovation Campus — Unit 41-42: Data Visualization & AI Dashboard")
            st.markdown(
                "Monitor training delivery, completion, governorate coverage, and financial performance."
            )
        with illustration_col:
            if PIL_AVAILABLE:
                st.image(build_dashboard_illustration(), width=520)
            else:
                st.markdown("## 🇪🇬 🏙️ 📊 🎓")
                st.caption("Training insights across Egypt")

    full_df = load_dataset()
    filtered_df = render_sidebar(full_df)

    if filtered_df.empty:
        st.warning(
            "⚠️ No training sessions match the currently selected filter combination. "
            "Use **Reset all filters** in the sidebar or broaden your selections to display analytics."
        )
        st.stop()

    if _safe_mean(filtered_df["completion_rate"]) >= 0.90 and st.session_state["last_balloons_sig"] != st.session_state["last_filter_sig"]:
        st.balloons()
        st.session_state["last_balloons_sig"] = st.session_state["last_filter_sig"]

    st.caption(
        f"Displaying **{len(filtered_df):,}** of **{len(full_df):,}** training sessions · "
        f"**{filtered_df['city'].nunique()}** active cities · "
        f"**{filtered_df['governorate'].nunique()}** governorates represented"
    )
    st.divider()

    tab_overview, tab_trends, tab_geo, tab_explorer, tab_ai = st.tabs([
        "📊 Overview",
        "📈 Trends",
        "🗺️ Geography",
        "🔎 Data Explorer",
        "💬 Operations Assistant",
    ])

    with tab_overview:
        render_tab_overview(filtered_df, full_df)

    with tab_trends:
        render_tab_trends(filtered_df)

    with tab_geo:
        render_tab_geography(filtered_df)

    with tab_explorer:
        render_tab_explorer(filtered_df)

    with tab_ai:
        render_tab_assistant(filtered_df)


if __name__ == "__main__":
    main()