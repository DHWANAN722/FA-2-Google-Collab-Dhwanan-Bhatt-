# =============================================================================
# FA-2: ATM Intelligence Demand Forecasting with Data Mining
# FinTrust Bank Ltd. — Cyberpunk Dashboard Edition
# =============================================================================
# HOW TO RUN:
#   1. pip install -r requirements.txt
#   2. Place atm_cash_management_dataset.csv in the same directory as app.py.
#   3. Run:  streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinTrust ATM Intelligence",
    page_icon="🏧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CYBERPUNK COLOUR CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
NEON_CYAN    = "#00F0FF"
NEON_MAGENTA = "#FF00E5"
NEON_GREEN   = "#39FF14"
NEON_YELLOW  = "#FFE500"
NEON_ORANGE  = "#FF6B00"
NEON_PINK    = "#FF2D7B"
DARK_BG      = "#0A0A1A"
CARD_BG      = "#12122A"
GRID_COLOR   = "#1A1A3A"
TEXT_COLOR    = "#E0E0FF"

NEON_PALETTE = [
    NEON_CYAN, NEON_MAGENTA, NEON_GREEN, NEON_YELLOW,
    NEON_ORANGE, NEON_PINK, "#7B61FF", "#00FF88", "#FF4444", "#44BBFF",
]

# ─────────────────────────────────────────────────────────────────────────────
# INJECT CYBERPUNK CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');

:root {
    --neon-cyan: #00F0FF;
    --neon-magenta: #FF00E5;
    --neon-green: #39FF14;
    --neon-yellow: #FFE500;
    --dark-bg: #0A0A1A;
    --card-bg: #12122A;
    --text: #E0E0FF;
}

/* ── Dark background everywhere ── */
.stApp, .main, [data-testid="stAppViewContainer"],
[data-testid="stHeader"], header {
    background: linear-gradient(170deg, #0A0A1A 0%, #0D0D2B 40%, #0A0A1A 100%) !important;
    color: var(--text) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #08081A 0%, #10102A 100%) !important;
    border-right: 1px solid rgba(0,240,255,0.25) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stRadio label span {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 1.5px !important;
}

/* ── Headings ── */
h1, h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    text-transform: uppercase !important;
    letter-spacing: 3px !important;
}
h1 {
    color: var(--neon-cyan) !important;
    text-shadow: 0 0 20px rgba(0,240,255,0.6), 0 0 60px rgba(0,240,255,0.2) !important;
    font-size: 1.6rem !important;
    border-bottom: 2px solid var(--neon-magenta);
    padding-bottom: 12px;
}
h2 {
    color: var(--neon-magenta) !important;
    text-shadow: 0 0 15px rgba(255,0,229,0.5) !important;
    font-size: 1.15rem !important;
}
h3 {
    color: var(--neon-green) !important;
    text-shadow: 0 0 10px rgba(57,255,20,0.4) !important;
    font-size: 0.95rem !important;
}

/* ── Body text ── */
p, li, span, div, td, th, label {
    font-family: 'Rajdhani', sans-serif !important;
    color: var(--text) !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #12122A 0%, #1A1A3E 100%) !important;
    border: 1px solid rgba(0,240,255,0.25) !important;
    border-radius: 10px !important;
    padding: 16px !important;
    box-shadow: 0 0 20px rgba(0,240,255,0.08), inset 0 0 20px rgba(0,240,255,0.04) !important;
}
[data-testid="stMetric"] label {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.6rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--neon-cyan) !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--neon-green) !important;
    text-shadow: 0 0 10px rgba(57,255,20,0.5) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: transparent !important;
    border-bottom: 2px solid #1A1A3A !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.58rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: #6666AA !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 4px 4px 0 0 !important;
    padding: 8px 14px !important;
}
.stTabs [aria-selected="true"] {
    color: var(--neon-cyan) !important;
    border-color: var(--neon-cyan) !important;
    background: rgba(0,240,255,0.07) !important;
    text-shadow: 0 0 12px rgba(0,240,255,0.6) !important;
}

/* ── Selectbox / Multiselect ── */
[data-baseweb="select"] > div,
[data-baseweb="popover"] > div {
    background: #12122A !important;
    border-color: rgba(0,240,255,0.25) !important;
    color: var(--text) !important;
}

/* ── Expander ── */
details {
    background: var(--card-bg) !important;
    border: 1px solid rgba(255,229,0,0.15) !important;
    border-radius: 8px !important;
}
details summary span {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.72rem !important;
    letter-spacing: 1px !important;
    color: var(--neon-yellow) !important;
}

/* ── Alert boxes ── */
.stAlert {
    background: var(--card-bg) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(0,240,255,0.15) !important;
    border-radius: 8px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0A0A1A; }
::-webkit-scrollbar-thumb { background: rgba(0,240,255,0.3); border-radius: 3px; }

/* ── Animated neon scanline at top ── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00F0FF, #FF00E5, #39FF14, transparent);
    z-index: 9999;
    animation: pulse 3s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 0.4; }
    50% { opacity: 1; }
}

/* ── Observation callout ── */
.obs-box {
    background: linear-gradient(135deg, rgba(0,240,255,0.06) 0%, rgba(255,0,229,0.04) 100%);
    border-left: 3px solid var(--neon-cyan);
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 10px 0 18px 0;
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.92rem;
    color: #C0C0E0;
    line-height: 1.5;
}
.obs-box strong { color: var(--neon-cyan); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — PLOTLY CYBERPUNK LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=DARK_BG,
    font=dict(family="Rajdhani, sans-serif", color=TEXT_COLOR, size=13),
    title_font=dict(family="Orbitron, sans-serif", size=15, color=NEON_CYAN),
    xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
               title_font=dict(color=NEON_CYAN, size=12)),
    yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
               title_font=dict(color=NEON_CYAN, size=12)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_COLOR, size=11)),
    margin=dict(l=50, r=20, t=55, b=45),
    hoverlabel=dict(bgcolor=CARD_BG, font_size=13, font_family="Rajdhani"),
)


def neon(fig):
    """Apply cyberpunk layout to any Plotly figure."""
    fig.update_layout(**PLOTLY_BASE)
    return fig


def observation(text):
    """Render a styled observation callout."""
    st.markdown(
        f'<div class="obs-box">📌 <strong>Observation:</strong> {text}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load the ATM cash management dataset."""
    df = pd.read_csv("atm_cash_management_dataset.csv", parse_dates=["Date"])
    return df


df = load_data()


# ─────────────────────────────────────────────────────────────────────────────
# PRECOMPUTE: CLUSTERING (Stage 4) & ANOMALIES (Stage 5) — cached
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def compute_clusters(df):
    """
    Stage 4 — K-Means clustering.
    Features: Total_Withdrawals, Total_Deposits, Location_Type (encoded),
              Nearby_Competitor_ATMs.
    Best K chosen via highest silhouette score in range 2-10.
    """
    le = LabelEncoder()
    df = df.copy()
    df["Location_Type_Enc"] = le.fit_transform(df["Location_Type"])

    features = [
        "Total_Withdrawals", "Total_Deposits",
        "Location_Type_Enc", "Nearby_Competitor_ATMs",
    ]
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Evaluate K = 2..10
    K_range = range(2, 11)
    inertias, sil_scores = [], []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    best_k = list(K_range)[int(np.argmax(sil_scores))]

    # Final model with best K
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["Cluster"] = km_final.fit_predict(X_scaled)

    # Label clusters by ascending mean withdrawal
    cluster_means = df.groupby("Cluster")["Total_Withdrawals"].mean().sort_values()
    tags = [
        "Very Low", "Low", "Medium-Low", "Medium",
        "Medium-High", "High", "Very High", "Extreme", "Ultra", "Hyper",
    ]
    label_map = {
        cid: f"{tags[min(i, len(tags)-1)]} Demand"
        for i, (cid, _) in enumerate(cluster_means.items())
    }
    df["Cluster_Label"] = df["Cluster"].map(label_map)

    return df, best_k, inertias, sil_scores, list(K_range)


@st.cache_data
def compute_anomalies(df):
    """
    Stage 5 — Anomaly detection with three methods:
      • Z-Score (threshold 2.5)
      • IQR (1.5× fence)
      • Isolation Forest (5 % contamination)
    """
    df = df.copy()

    # Z-Score
    df["Z_Score"] = np.abs(stats.zscore(df["Total_Withdrawals"]))
    df["Anomaly_ZScore"] = df["Z_Score"] > 2.5

    # IQR
    Q1 = df["Total_Withdrawals"].quantile(0.25)
    Q3 = df["Total_Withdrawals"].quantile(0.75)
    IQR = Q3 - Q1
    df["Anomaly_IQR"] = (
        (df["Total_Withdrawals"] < Q1 - 1.5 * IQR)
        | (df["Total_Withdrawals"] > Q3 + 1.5 * IQR)
    )

    # Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    df["Anomaly_IF"] = iso.fit_predict(df[["Total_Withdrawals"]]) == -1

    return df


# Run cached preprocessing
df, best_k, inertias, sil_scores, K_range = compute_clusters(df)
df = compute_anomalies(df)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:10px 0 20px 0;'>
        <span style='font-family:Orbitron; font-size:1.3rem; color:#00F0FF;
        text-shadow: 0 0 25px rgba(0,240,255,0.7);'>⬡ FINTRUST</span><br>
        <span style='font-family:Share Tech Mono; font-size:0.65rem; color:#FF00E5;
        letter-spacing:4px;'>ATM INTELLIGENCE v2.0</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "◈ NAVIGATE",
        [
            "🔍 Overview",
            "📊 Stage 3 · EDA",
            "🧩 Stage 4 · Clustering",
            "⚡ Stage 5 · Anomaly Detection",
            "🎮 Stage 6 · Interactive Planner",
        ],
    )

    st.markdown("---")
    st.markdown(f"""
    <div style='font-family:Share Tech Mono; font-size:0.68rem; color:#8888AA;
    padding:10px; border:1px solid #1A1A3A; border-radius:6px;
    background:rgba(0,240,255,0.03);'>
    📁 RECORDS &nbsp;{len(df):,}<br>
    📅 RANGE &nbsp;&nbsp;{df["Date"].min().strftime("%Y-%m-%d")} →
    {df["Date"].max().strftime("%Y-%m-%d")}<br>
    🏧 ATMs &nbsp;&nbsp;&nbsp;&nbsp;{df["ATM_ID"].nunique()}<br>
    🧩 CLUSTERS &nbsp;{best_k}
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE : OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "🔍 Overview":
    st.markdown("# 🏧 ATM Intelligence Demand Forecasting")
    st.caption(
        "FinTrust Bank Ltd. — FA-2: Building Actionable Insights & Interactive Dashboard"
    )
    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("ATM Count", df["ATM_ID"].nunique())
    c3.metric("Avg Withdrawal", f"₹{df['Total_Withdrawals'].mean():,.0f}")
    c4.metric("Avg Deposit", f"₹{df['Total_Deposits'].mean():,.0f}")
    c5.metric("Holiday Records", f"{df['Holiday_Flag'].sum():,}")

    st.markdown("---")
    st.markdown("### 📋 Dataset Preview")
    st.dataframe(df.head(25), use_container_width=True, height=420)

    st.markdown("### 📈 Summary Statistics")
    st.dataframe(df.describe().round(2), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE : STAGE 3 — EDA
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Stage 3 · EDA":
    st.markdown("# Stage 3 — Exploratory Data Analysis")
    st.caption("Uncovering patterns, trends, and relationships in ATM transaction data.")
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Distributions", "📦 Outliers", "📈 Time Trends",
        "🎄 Holiday Impact", "🌤️ External Factors", "🔗 Relationships",
    ])

    # ── 3.1  Distribution Analysis ────────────────────────────────────────
    with tab1:
        st.markdown("## 3.1 Distribution of Withdrawals & Deposits")
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Histogram(
                x=df["Total_Withdrawals"], nbinsx=35,
                marker_color=NEON_CYAN, marker_line_width=0, opacity=0.85,
                name="Withdrawals",
            ))
            fig.update_layout(title="Total Withdrawals",
                              xaxis_title="Amount (₹)", yaxis_title="Frequency")
            st.plotly_chart(neon(fig), use_container_width=True)
        with c2:
            fig = go.Figure(go.Histogram(
                x=df["Total_Deposits"], nbinsx=35,
                marker_color=NEON_MAGENTA, marker_line_width=0, opacity=0.85,
                name="Deposits",
            ))
            fig.update_layout(title="Total Deposits",
                              xaxis_title="Amount (₹)", yaxis_title="Frequency")
            st.plotly_chart(neon(fig), use_container_width=True)
        observation(
            "Withdrawals show a roughly normal distribution centred around ₹50K. "
            "Deposits are lower and right-skewed — ATMs dispense far more than they receive."
        )

    # ── 3.2  Box Plots ───────────────────────────────────────────────────
    with tab2:
        st.markdown("## 3.2 Box Plots — Outlier Detection")
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Box(
                y=df["Total_Withdrawals"], marker_color=NEON_CYAN,
                line_color=NEON_CYAN, fillcolor="rgba(0,240,255,0.12)",
                name="Withdrawals",
            ))
            fig.update_layout(title="Withdrawals Box Plot")
            st.plotly_chart(neon(fig), use_container_width=True)
        with c2:
            fig = go.Figure(go.Box(
                y=df["Total_Deposits"], marker_color=NEON_MAGENTA,
                line_color=NEON_MAGENTA, fillcolor="rgba(255,0,229,0.12)",
                name="Deposits",
            ))
            fig.update_layout(title="Deposits Box Plot")
            st.plotly_chart(neon(fig), use_container_width=True)
        observation(
            "A handful of high-value outliers are visible in withdrawals — likely holiday "
            "or event-driven spikes. Deposits have a tighter IQR with fewer extremes."
        )

    # ── 3.3  Time-Based Trends ────────────────────────────────────────────
    with tab3:
        st.markdown("## 3.3 Time-Based Trends")

        # 3.3a — Line chart
        daily = df.groupby("Date")["Total_Withdrawals"].sum().reset_index()
        fig = go.Figure(go.Scatter(
            x=daily["Date"], y=daily["Total_Withdrawals"],
            mode="lines", line=dict(color=NEON_CYAN, width=1.2),
            fill="tozeroy", fillcolor="rgba(0,240,255,0.06)",
        ))
        fig.update_layout(title="3.3a  Total Withdrawals Over Time",
                          xaxis_title="Date", yaxis_title="Total Withdrawals (₹)")
        st.plotly_chart(neon(fig), use_container_width=True)
        observation(
            "Withdrawals stay broadly stable with periodic spikes, suggesting recurring "
            "high-demand events like paydays or public holidays."
        )

        c1, c2 = st.columns(2)
        # 3.3b — Day of Week
        with c1:
            day_order = [
                "Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday",
            ]
            day_avg = (
                df.groupby("Day_of_Week")["Total_Withdrawals"]
                .mean().reindex(day_order).reset_index()
            )
            fig = go.Figure(go.Bar(
                x=day_avg["Day_of_Week"], y=day_avg["Total_Withdrawals"],
                marker_color=NEON_CYAN, marker_line_width=0, opacity=0.9,
            ))
            fig.update_layout(title="3.3b  Avg Withdrawals by Day",
                              xaxis_title="Day", yaxis_title="Avg Withdrawals (₹)")
            st.plotly_chart(neon(fig), use_container_width=True)

        # 3.3c — Time of Day
        with c2:
            time_avg = (
                df.groupby("Time_of_Day")["Total_Withdrawals"]
                .mean().reset_index()
            )
            fig = go.Figure(go.Bar(
                x=time_avg["Time_of_Day"], y=time_avg["Total_Withdrawals"],
                marker_color=NEON_GREEN, marker_line_width=0, opacity=0.9,
            ))
            fig.update_layout(title="3.3c  Avg Withdrawals by Time of Day",
                              xaxis_title="Time of Day",
                              yaxis_title="Avg Withdrawals (₹)")
            st.plotly_chart(neon(fig), use_container_width=True)

        observation(
            "Withdrawals are fairly consistent across weekdays. "
            "Morning and Evening periods tend to show the highest activity."
        )

    # ── 3.4  Holiday & Event Impact ───────────────────────────────────────
    with tab4:
        st.markdown("## 3.4 Holiday & Special Event Impact")
        c1, c2 = st.columns(2)
        with c1:
            h_avg = df.groupby("Holiday_Flag")["Total_Withdrawals"].mean().reset_index()
            h_avg["Label"] = h_avg["Holiday_Flag"].map(
                {0: "No Holiday", 1: "Holiday"}
            )
            fig = go.Figure(go.Bar(
                x=h_avg["Label"], y=h_avg["Total_Withdrawals"],
                marker_color=[NEON_CYAN, NEON_ORANGE], marker_line_width=0,
            ))
            fig.update_layout(title="By Holiday Flag",
                              yaxis_title="Avg Withdrawals (₹)")
            st.plotly_chart(neon(fig), use_container_width=True)
        with c2:
            e_avg = (
                df.groupby("Special_Event_Flag")["Total_Withdrawals"]
                .mean().reset_index()
            )
            e_avg["Label"] = e_avg["Special_Event_Flag"].map(
                {0: "No Event", 1: "Special Event"}
            )
            fig = go.Figure(go.Bar(
                x=e_avg["Label"], y=e_avg["Total_Withdrawals"],
                marker_color=[NEON_CYAN, NEON_GREEN], marker_line_width=0,
            ))
            fig.update_layout(title="By Special Event Flag",
                              yaxis_title="Avg Withdrawals (₹)")
            st.plotly_chart(neon(fig), use_container_width=True)
        observation(
            "ATMs see noticeably higher withdrawals on holidays and special event days. "
            "External calendar events are key demand drivers for replenishment scheduling."
        )

    # ── 3.5  External Factors ─────────────────────────────────────────────
    with tab5:
        st.markdown("## 3.5 External Factors")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(
                df, x="Weather_Condition", y="Total_Withdrawals",
                color="Weather_Condition",
                color_discrete_sequence=[
                    NEON_CYAN, NEON_MAGENTA, NEON_GREEN, NEON_YELLOW
                ],
            )
            fig.update_layout(title="Withdrawals by Weather", showlegend=False,
                              xaxis_title="Weather",
                              yaxis_title="Total Withdrawals (₹)")
            st.plotly_chart(neon(fig), use_container_width=True)
        with c2:
            fig = px.box(
                df, x="Nearby_Competitor_ATMs", y="Total_Withdrawals",
                color="Nearby_Competitor_ATMs",
                color_discrete_sequence=NEON_PALETTE,
            )
            fig.update_layout(title="Withdrawals vs Competitor ATMs",
                              showlegend=False,
                              xaxis_title="Nearby Competitors",
                              yaxis_title="Total Withdrawals (₹)")
            st.plotly_chart(neon(fig), use_container_width=True)
        observation(
            "Weather conditions show a mild impact — rainy/snowy days may slightly "
            "reduce foot traffic. ATMs with fewer nearby competitors attract more volume."
        )

    # ── 3.6  Relationship Analysis ────────────────────────────────────────
    with tab6:
        st.markdown("## 3.6 Relationship Analysis")

        # Scatter
        fig = go.Figure(go.Scattergl(
            x=df["Previous_Day_Cash_Level"],
            y=df["Cash_Demand_Next_Day"],
            mode="markers",
            marker=dict(color=NEON_CYAN, size=3, opacity=0.35),
        ))
        fig.update_layout(
            title="Previous Day Cash Level vs Next Day Demand",
            xaxis_title="Previous Day Cash Level (₹)",
            yaxis_title="Cash Demand Next Day (₹)",
        )
        st.plotly_chart(neon(fig), use_container_width=True)
        observation(
            "No strong linear relationship — next-day demand is driven more by external "
            "factors (day of week, holidays, events) than current stock levels."
        )

        # Correlation heatmap
        st.markdown("### Correlation Heatmap")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        exclude = {
            "Cluster", "Location_Type_Enc", "Z_Score",
            "Anomaly_ZScore", "Anomaly_IQR", "Anomaly_IF",
        }
        numeric_cols = [c for c in numeric_cols if c not in exclude]
        corr = df[numeric_cols].corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale=[
                [0, "#0A0A1A"], [0.5, "#FF00E5"], [1, "#00F0FF"]
            ],
            zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=10, color=TEXT_COLOR),
        ))
        fig.update_layout(
            title="Correlation Heatmap — Numeric Features",
            height=550, margin=dict(l=120, b=120),
        )
        st.plotly_chart(neon(fig), use_container_width=True)
        observation(
            "Total_Withdrawals has a positive correlation with Cash_Demand_Next_Day. "
            "Holiday and Special Event flags show mild positive correlations with "
            "withdrawal volume."
        )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE : STAGE 4 — CLUSTERING
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🧩 Stage 4 · Clustering":
    st.markdown("# Stage 4 — Clustering Analysis of ATMs")
    st.caption(
        "Grouping ATMs by demand behaviour using K-Means for smarter cash management."
    )
    st.markdown("---")

    # ── 4.1 / 4.2  Elbow + Silhouette ────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Scatter(
            x=list(K_range), y=inertias, mode="lines+markers",
            line=dict(color=NEON_CYAN, width=2),
            marker=dict(
                size=8, color=NEON_CYAN,
                line=dict(width=1, color=DARK_BG),
            ),
        ))
        fig.update_layout(title="4.1  Elbow Method",
                          xaxis_title="Number of Clusters (K)",
                          yaxis_title="Inertia")
        st.plotly_chart(neon(fig), use_container_width=True)

    with c2:
        colors = [
            NEON_GREEN if k == best_k else NEON_MAGENTA for k in K_range
        ]
        fig = go.Figure(go.Bar(
            x=list(K_range), y=sil_scores,
            marker_color=colors, marker_line_width=0,
        ))
        fig.update_layout(
            title=f"4.2  Silhouette Scores  (Best K = {best_k})",
            xaxis_title="K", yaxis_title="Silhouette Score",
        )
        st.plotly_chart(neon(fig), use_container_width=True)

    observation(
        f"The elbow curve shows diminishing returns beyond a certain K. "
        f"The best silhouette score selects <strong>K = {best_k}</strong>, "
        f"giving the tightest, most separated clusters."
    )

    st.markdown("---")

    # ── 4.3  Cluster scatter ──────────────────────────────────────────────
    st.markdown("## 4.3 ATM Clusters: Withdrawals vs Deposits")
    fig = px.scatter(
        df, x="Total_Withdrawals", y="Total_Deposits",
        color="Cluster_Label", opacity=0.5,
        color_discrete_sequence=NEON_PALETTE,
    )
    fig.update_traces(marker_size=4)
    fig.update_layout(
        height=520,
        xaxis_title="Total Withdrawals (₹)",
        yaxis_title="Total Deposits (₹)",
    )
    st.plotly_chart(neon(fig), use_container_width=True)
    observation(
        "Clusters form distinct demand bands. High-demand ATMs (urban, event-prone) "
        "should be prioritised for frequent replenishment, while low-demand ones can "
        "follow relaxed schedules."
    )

    # ── 4.4  Cluster counts + summary ─────────────────────────────────────
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.markdown("### 4.4 Records per Cluster")
        counts = df["Cluster_Label"].value_counts().reset_index()
        counts.columns = ["Cluster", "Count"]
        fig = go.Figure(go.Bar(
            x=counts["Cluster"], y=counts["Count"],
            marker_color=NEON_PALETTE[: len(counts)],
            marker_line_width=0,
        ))
        fig.update_layout(xaxis_title="Cluster", yaxis_title="Count",
                          height=380)
        st.plotly_chart(neon(fig), use_container_width=True)

    with c2:
        st.markdown("### Cluster Summary (Averages)")
        summary = (
            df.groupby("Cluster_Label")[
                ["Total_Withdrawals", "Total_Deposits",
                 "Nearby_Competitor_ATMs"]
            ].mean().round(0).reset_index()
        )
        summary.columns = [
            "Cluster", "Avg Withdrawals", "Avg Deposits", "Avg Competitors"
        ]
        st.dataframe(summary, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE : STAGE 5 — ANOMALY DETECTION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Stage 5 · Anomaly Detection":
    st.markdown("# Stage 5 — Anomaly Detection")
    st.caption(
        "Identifying unusual withdrawal spikes during holidays, events, and beyond."
    )
    st.markdown("---")

    # ── Metric summary ────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Z-Score Anomalies (|z| > 2.5)", int(df["Anomaly_ZScore"].sum()))
    c2.metric("IQR Anomalies", int(df["Anomaly_IQR"].sum()))
    c3.metric("Isolation Forest Anomalies", int(df["Anomaly_IF"].sum()))

    st.markdown("---")

    # ── 5.1  Holiday vs Normal distribution ───────────────────────────────
    st.markdown("## 5.1 Withdrawal Distribution: Holiday vs Normal")
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df[df["Holiday_Flag"] == 0]["Total_Withdrawals"],
        nbinsx=35, marker_color=NEON_CYAN, opacity=0.6, name="Normal Days",
    ))
    fig.add_trace(go.Histogram(
        x=df[df["Holiday_Flag"] == 1]["Total_Withdrawals"],
        nbinsx=35, marker_color=NEON_ORANGE, opacity=0.75, name="Holiday Days",
    ))
    fig.update_layout(
        barmode="overlay",
        xaxis_title="Total Withdrawals (₹)", yaxis_title="Frequency",
        height=420,
    )
    st.plotly_chart(neon(fig), use_container_width=True)
    observation(
        "Holiday days show a distribution shifted towards higher withdrawals, "
        "confirming elevated demand during holidays."
    )

    st.markdown("---")

    # ── 5.2  Anomalies over time ──────────────────────────────────────────
    st.markdown("## 5.2 Anomalies Over Time — Isolation Forest")
    normal = df[~df["Anomaly_IF"]]
    anomaly = df[df["Anomaly_IF"]]
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=normal["Date"], y=normal["Total_Withdrawals"],
        mode="markers",
        marker=dict(color=NEON_CYAN, size=2.5, opacity=0.3),
        name="Normal",
    ))
    fig.add_trace(go.Scattergl(
        x=anomaly["Date"], y=anomaly["Total_Withdrawals"],
        mode="markers",
        marker=dict(
            color=NEON_ORANGE, size=7, opacity=0.9,
            line=dict(width=1, color=NEON_YELLOW),
        ),
        name="Anomaly",
    ))
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Total Withdrawals (₹)", height=450,
    )
    st.plotly_chart(neon(fig), use_container_width=True)
    observation(
        "Anomalies (orange) coincide with extreme withdrawal periods — strong "
        "candidates for event-driven demand spikes needing proactive cash loading."
    )

    st.markdown("---")

    # ── 5.3  Anomalies by Holiday flag ────────────────────────────────────
    st.markdown("## 5.3 Anomalies: Holiday vs Normal Days")
    anom_h = (
        df[df["Anomaly_IF"]]["Holiday_Flag"]
        .value_counts().reindex([0, 1], fill_value=0)
    )
    anom_h.index = ["Normal Days", "Holiday Days"]
    fig = go.Figure(go.Bar(
        x=anom_h.index, y=anom_h.values,
        marker_color=[NEON_CYAN, NEON_ORANGE], marker_line_width=0,
    ))
    fig.update_layout(yaxis_title="Anomaly Count", height=380)
    st.plotly_chart(neon(fig), use_container_width=True)
    observation(
        "A disproportionate share of anomalies falls on holiday days, confirming "
        "that holidays are a key trigger for unusual withdrawal behaviour."
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE : STAGE 6 — INTERACTIVE PLANNER
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🎮 Stage 6 · Interactive Planner":
    st.markdown("# Stage 6 — Interactive ATM Demand Planner")
    st.caption(
        "Filter by Day, Time, and Location to explore demand insights on the fly."
    )
    st.markdown("---")

    # ── Filter controls ───────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        day_opts = ["All"] + sorted(df["Day_of_Week"].unique().tolist())
        sel_day = st.selectbox("🗓️ Day of Week", day_opts)
    with fc2:
        time_opts = ["All"] + sorted(df["Time_of_Day"].unique().tolist())
        sel_time = st.selectbox("🕐 Time of Day", time_opts)
    with fc3:
        loc_opts = ["All"] + sorted(df["Location_Type"].unique().tolist())
        sel_loc = st.selectbox("📍 Location Type", loc_opts)

    # Apply filters
    filtered = df.copy()
    if sel_day != "All":
        filtered = filtered[filtered["Day_of_Week"] == sel_day]
    if sel_time != "All":
        filtered = filtered[filtered["Time_of_Day"] == sel_time]
    if sel_loc != "All":
        filtered = filtered[filtered["Location_Type"] == sel_loc]

    st.markdown("---")

    if filtered.empty:
        st.warning("No records match these filters. Try a different combination.")
    else:
        # ── KPI row ──────────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Matched Records", f"{len(filtered):,}")
        k2.metric("Avg Withdrawal", f"₹{filtered['Total_Withdrawals'].mean():,.0f}")
        k3.metric("Avg Deposit", f"₹{filtered['Total_Deposits'].mean():,.0f}")
        n_anom = int(filtered["Anomaly_IF"].sum())
        k4.metric("Anomalies", f"{n_anom}  ({n_anom/len(filtered)*100:.1f}%)")

        st.markdown("---")

        # ── Dashboard row: 3 charts ──────────────────────────────────────
        d1, d2, d3 = st.columns(3)

        with d1:
            fig = go.Figure(go.Histogram(
                x=filtered["Total_Withdrawals"], nbinsx=25,
                marker_color=NEON_CYAN, marker_line_width=0, opacity=0.85,
            ))
            fig.update_layout(
                title="Withdrawal Distribution",
                xaxis_title="₹", yaxis_title="Freq", height=370,
            )
            st.plotly_chart(neon(fig), use_container_width=True)

        with d2:
            cl = filtered["Cluster_Label"].value_counts().reset_index()
            cl.columns = ["Cluster", "Count"]
            fig = go.Figure(go.Bar(
                x=cl["Cluster"], y=cl["Count"],
                marker_color=NEON_PALETTE[: len(cl)], marker_line_width=0,
            ))
            fig.update_layout(
                title="Cluster Breakdown",
                xaxis_title="Cluster", yaxis_title="Count", height=370,
            )
            fig.update_xaxes(tickangle=30)
            st.plotly_chart(neon(fig), use_container_width=True)

        with d3:
            fn = filtered[~filtered["Anomaly_IF"]]
            fa = filtered[filtered["Anomaly_IF"]]
            fig = go.Figure()
            fig.add_trace(go.Scattergl(
                x=fn["Previous_Day_Cash_Level"],
                y=fn["Total_Withdrawals"],
                mode="markers",
                marker=dict(color=NEON_CYAN, size=3, opacity=0.35),
                name="Normal",
            ))
            fig.add_trace(go.Scattergl(
                x=fa["Previous_Day_Cash_Level"],
                y=fa["Total_Withdrawals"],
                mode="markers",
                marker=dict(
                    color=NEON_ORANGE, size=8, opacity=0.9,
                    line=dict(width=1, color=NEON_YELLOW),
                ),
                name="Anomaly",
            ))
            fig.update_layout(
                title="Anomalies in Selection",
                xaxis_title="Prev Day Cash (₹)",
                yaxis_title="Withdrawals (₹)", height=370,
            )
            st.plotly_chart(neon(fig), use_container_width=True)

        # ── Expandable details ────────────────────────────────────────────
        with st.expander("📋 DETAILED SUMMARY STATISTICS"):
            st.dataframe(
                filtered[
                    ["Total_Withdrawals", "Total_Deposits", "Cash_Demand_Next_Day"]
                ].describe().round(0),
                use_container_width=True,
            )

        with st.expander("🧩 CLUSTER DISTRIBUTION TABLE"):
            st.dataframe(
                filtered.groupby("Cluster_Label")[
                    ["Total_Withdrawals", "Total_Deposits",
                     "Nearby_Competitor_ATMs"]
                ].mean().round(0),
                use_container_width=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; padding:8px 0; font-family:Share Tech Mono;
font-size:0.65rem; color:#555577; letter-spacing:2px;'>
⬡ FINTRUST ATM INTELLIGENCE v2.0 &nbsp;|&nbsp; FA-2 DATA MINING PROJECT
&nbsp;|&nbsp; BUILT WITH STREAMLIT + PLOTLY
</div>
""", unsafe_allow_html=True)
