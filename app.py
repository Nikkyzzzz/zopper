# ======================== IMPORTS & SETUP ========================
# This section imports all necessary libraries for data processing, visualization, and Streamlit app creation

import re
import base64
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import warnings

# Suppress deprecation warnings to keep console output clean
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Attempt to import plotly for interactive visualizations; gracefully handle if not installed
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

# Configure Python environment to suppress warnings
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

# Import machine learning models for clustering and forecasting
from sklearn.cluster import KMeans  # Used for store segmentation
from sklearn.linear_model import LinearRegression  # Used for trend analysis and January forecast


# ======================== PAGE CONFIGURATION & STYLING ========================
# Configure Streamlit page layout, title, and initial appearance

st.set_page_config(
    page_title="Jumbo & Company — Device Insurance Analytics",
    page_icon="📊",
    layout="wide",  # Use wide layout to maximize screen space
    initial_sidebar_state="collapsed",  # Start with sidebar hidden
)

CSS = """
<style>
/* Custom CSS styling to create a professional, modern UI with consistent colors and layout */
/* --- Color Variables & Global Styles --- */
:root{
    --bg:#ffffff;
    --card:#f8f9fa;
    --card2:#f0f2f5;
    --text:#1a1a1a;
    --muted:#4a5568;
    --accent:#0056b3;
    --accent-alt:#204492;
    --accent-cool:#1db5e9;
    --good:#0f9d58;
    --bad:#d64545;
    --warn:#f26f2c;
    --border:rgba(0,0,0,0.14);
    --shadow: 0 12px 32px rgba(0,0,0,0.12);
  --radius:18px;
}
.main > div {padding-top: 1.8rem; background: var(--bg);} 
.block-container{max-width: 1350px; padding-top: 1.8rem; background: var(--bg);} 
[data-testid="stSidebar"] {background: linear-gradient(180deg, #f8f9fa 0%, #eef1f6 100%); border-right: 1px solid var(--border);} 
h1, h2, h3, h4, h5, h6, p, span, div {color: var(--text) !important;}
.small-muted{color: var(--muted) !important; font-size: 0.9rem;}
.badge{
    display:inline-block; padding: 0.15rem 0.4rem; border-radius: 999px;
    background: linear-gradient(120deg, rgba(0,86,179,0.12), rgba(29,181,233,0.12));
    border: 1px solid rgba(0,86,179,0.20);
    color: var(--text); font-size: 0.7rem; margin-right: 0.25rem;
}
.kpi{
    background: linear-gradient(145deg, rgba(255,255,255,0.85), rgba(240,242,245,0.9));
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.6rem 0.7rem;
    box-shadow: var(--shadow);
}
.kpi .label{color: var(--muted) !important; font-size: 0.75rem;}
.kpi .value{font-size: 1.2rem; font-weight: 700; color: var(--accent) !important;}
.kpi .delta{margin-top: 0.1rem; font-size: 0.75rem; color: var(--muted) !important;}
.panel{
    background: linear-gradient(145deg, rgba(248,249,250,0.95), rgba(240,242,245,0.95));
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.1rem;
    box-shadow: var(--shadow);
}
.hr{height:1px;background:var(--border);margin:0.9rem 0;}
.stButton > button{
    border-radius: 14px;
    border: 1px solid rgba(0,86,179,0.35);
    background: rgba(0,86,179,0.12);
    color: var(--text);
    padding: 0.5rem 0.9rem;
}
.stButton > button:hover{border-color: rgba(0,86,179,0.65); background: rgba(0,86,179,0.18);}
a{color: var(--accent) !important;}

.brand-row{
    display:flex; align-items:center; gap:0.5rem; margin:0.4rem 0 0.6rem 0;
    padding:0.3rem 0.5rem; border:1px solid var(--border); border-radius:10px;
    background: linear-gradient(120deg, rgba(0,86,179,0.08), rgba(29,181,233,0.08));
}
.brand-row img{height:28px;}
.brand-name{font-weight:700; color: var(--accent-alt); font-size:0.9rem;}
.brand-tag{color: var(--muted); font-size:0.8rem;}

/* Enhanced styles for better readability */
.stDataFrame {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

.stDataFrame table {
    background: var(--card2) !important;
}

.stDataFrame th {
    background: rgba(0,86,179,0.08) !important;
    font-weight: 600 !important;
    border-bottom: 1px solid var(--border) !important;
}

.stDataFrame tr:hover {
    background: rgba(0,86,179,0.06) !important;
}

/* Better form controls */
.stSelectbox, .stMultiselect, .stSlider {
    background: var(--card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* Better expander */
.streamlit-expanderHeader {
    background: var(--card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* Breadcrumb styling */
.breadcrumb {
    background: rgba(0,86,179,0.06); 
    padding: 0.5rem 1rem; 
    border-radius: 10px; 
    margin-bottom: 1.5rem; 
    border: 1px solid var(--border);
}

/* Section header */
.section-header {
    margin-bottom: 1.5rem;
}

.section-header h3 {
    color: var(--text); 
    margin-bottom: 0.25rem;
}

/* Hide Streamlit warning messages */
.stAlert, [data-testid="stNotification"], .element-container:has(.stAlert) {
    display: none !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ======================== LOGO SETUP ========================
# Load and encode the company logo as base64 for display in the sidebar

LOGO_PATH = Path(__file__).parent / "data" / "zopper_logo.svg"
LOGO_B64 = base64.b64encode(LOGO_PATH.read_bytes()).decode() if LOGO_PATH.exists() else None


# ======================== HELPER CONSTANTS & UTILITIES ========================
# Define constants for month ordering and mapping for time-series analysis

MONTHS_ORDER = ["Aug", "Sep", "Oct", "Nov", "Dec"]  # Months covered in the dataset
MONTH_TO_IDX = {m: i + 1 for i, m in enumerate(MONTHS_ORDER)}  # Convert month names to numeric indices (Aug=1, Dec=5)
IDX_TO_MONTH = {v: k for k, v in MONTH_TO_IDX.items()}  # Reverse mapping for converting indices back to month names

def _coerce_pct(x):
    """Convert percentage input to decimal format (0.23 instead of 23%)"""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = s.replace("%", "")  # Remove percent sign if present
    try:
        v = float(s)  # Convert to float
    except Exception:
        return np.nan  # Return NaN if conversion fails
    # Normalize: if value > 1.5, it's likely in percentage form (23 -> 0.23)
    if v > 1.5:
        v = v / 100.0
    return v

def load_data(file) -> pd.DataFrame:
    """Load Excel file and validate required columns and data format"""
    df = pd.read_excel(file)
    df = df.copy()
    # Standardize column names by stripping whitespace
    df.columns = [str(c).strip() for c in df.columns]
    
    # Validate required columns
    if "Branch" not in df.columns:
        raise ValueError("Missing column: Branch")
    
    # Handle flexible store column naming (Store_Name or Store)
    store_col = "Store_Name" if "Store_Name" in df.columns else ("Store" if "Store" in df.columns else None)
    if store_col is None:
        raise ValueError("Missing store column (Store_Name/Store).")
    df = df.rename(columns={store_col: "Store"})
    
    # Extract only month columns that are present in the dataset
    month_cols = [m for m in MONTHS_ORDER if m in df.columns]
    if not month_cols:
        raise ValueError("No month columns found (Aug..Dec).")
    
    # Convert all percentage values to decimal format
    for m in month_cols:
        df[m] = df[m].apply(_coerce_pct)

    # Keep only relevant columns and remove rows with missing Branch or Store
    df = df[["Branch", "Store"] + month_cols].dropna(subset=["Branch", "Store"])
    # Clean up text fields
    df["Branch"] = df["Branch"].astype(str).str.strip()
    df["Store"] = df["Store"].astype(str).str.strip()

    return df, month_cols

def to_long(df_wide: pd.DataFrame, month_cols):
    """Transform wide data (Branch, Store, Aug, Sep, ...) to long format for analysis"""
    # Melt converts columns to rows: each month value becomes a separate row
    long = df_wide.melt(id_vars=["Branch", "Store"], value_vars=month_cols,
                        var_name="Month", value_name="AttachPct")
    long["Month"] = long["Month"].astype(str).str.strip()
    # Add numeric time index for trend analysis (Aug=1, Sep=2, ... Dec=5)
    long["t"] = long["Month"].map(MONTH_TO_IDX)
    # Remove any rows with missing values and sort by branch, store, and time
    long = long.dropna(subset=["AttachPct", "t"]).sort_values(["Branch", "Store", "t"])
    return long

def add_store_metrics(long: pd.DataFrame):
    """Compute key performance metrics for each store: avg, min, max, trend, volatility"""
    g = long.groupby(["Branch", "Store"])
    
    # Calculate basic statistics for each store across all months
    metrics = g["AttachPct"].agg(avg="mean", min="min", max="max", std="std").reset_index()
    metrics["std"] = metrics["std"].fillna(0)  # Handle NaN for stores with single data point
    
    # Calculate trend (slope) using linear regression: AttachPct ~ time (t)
    slopes = []
    last_vals = []  # Track the most recent attach% value
    for (b, s), d in g:
        X = d[["t"]].values  # Time indices: Aug=1, Dec=5
        y = d["AttachPct"].values  # Attach % values
        if len(d) >= 2:
            # Fit linear regression to calculate trend slope
            lr = LinearRegression().fit(X, y)
            slope = float(lr.coef_[0])  # Positive slope = improving, negative = declining
        else:
            slope = 0.0  # No trend if only 1 data point
        slopes.append(((b, s), slope))
        # Get the most recent (last) attach% value for this store
        last_vals.append(((b, s), float(d.sort_values("t")["AttachPct"].iloc[-1])))
    
    # Combine all metrics into single dataframe
    slope_df = pd.DataFrame([{"Branch": k[0], "Store": k[1], "slope": v} for k, v in slopes])
    last_df  = pd.DataFrame([{"Branch": k[0], "Store": k[1], "last_attach": v} for k, v in last_vals])
    out = metrics.merge(slope_df, on=["Branch","Store"], how="left").merge(last_df, on=["Branch","Store"], how="left")
    out["volatility"] = out["std"]  # Volatility = standard deviation of attach %
    return out

def store_segmentation(store_metrics: pd.DataFrame, n_clusters=4):
    """Group stores into segments based on average performance, volatility, and trend"""
    # Select the 3 features for clustering
    X = store_metrics[["avg", "volatility", "slope"]].copy()
    
    # Normalize features using robust scaling (resistant to outliers)
    # This ensures each feature contributes equally to clustering
    X = (X - X.median()) / (X.quantile(0.75) - X.quantile(0.25) + 1e-9)
    
    # Apply K-Means clustering to group stores
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(X.values)
    
    seg = store_metrics.copy()
    seg["segment_id"] = labels  # Numeric cluster assignment (0-3)

    # Assign meaningful names based on cluster centroids in original space
    cent = seg.groupby("segment_id")[["avg","volatility","slope"]].mean()
    names = {}
    for sid, row in cent.iterrows():
        # Champions: High performance AND improving trend
        if row["avg"] >= cent["avg"].quantile(0.75) and row["slope"] >= 0:
            names[sid] = "Champions (High & improving)"
        # At-risk: High performance BUT declining trend
        elif row["avg"] >= cent["avg"].median() and row["slope"] < 0:
            names[sid] = "At-risk (High but falling)"
        # Risers: Low performance BUT improving trend
        elif row["avg"] < cent["avg"].median() and row["slope"] > 0:
            names[sid] = "Risers (Low but improving)"
        # Long tail: Low performance AND flat/declining trend
        else:
            names[sid] = "Long tail (Low & flat)"
    
    seg["segment"] = seg["segment_id"].map(names).fillna("Segment")
    return seg

def predict_jan(store_metrics: pd.DataFrame, long: pd.DataFrame):
    """
    Forecast January attach% (t=6) per store using ensemble approach:
    1. Store-level linear trend extrapolation (Aug-Dec data, predict t=6)
    2. Branch-level trend as regularization for stores with high volatility
    3. Confidence score based on data stability and volume
    """
    g_store = long.groupby(["Branch", "Store"])
    g_branch = long.groupby(["Branch"])

    # Step 1: Fit branch-level linear trend model (AttachPct ~ time)
    # This serves as a fallback/regularization for individual stores
    branch_pred = {}
    for b, d in g_branch:
        X = d[["t"]].values  # Time indices
        y = d["AttachPct"].values  # Attach % values for entire branch
        if len(d) >= 2:
            # Fit linear regression at branch level
            lr = LinearRegression().fit(X, y)
            # Predict January (t=6)
            pred = float(lr.predict(np.array([[6]]) )[0])
        else:
            # Fallback: use average if insufficient data
            pred = float(np.nanmean(y)) if len(d) else 0.0
        branch_pred[b] = pred

    # Step 2: Fit store-level predictions and combine with branch predictions
    rows = []
    for (b, s), d in g_store:
        X = d[["t"]].values  # Time indices for this store
        y = d["AttachPct"].values  # Attach % values for this store
        if len(d) >= 2:
            # Fit linear regression at store level
            lr = LinearRegression().fit(X, y)
            # Predict January (t=6)
            store_jan = float(lr.predict(np.array([[6]]) )[0])
        else:
            # Fallback: use last available value if insufficient data
            store_jan = float(y[-1]) if len(d) else 0.0

        # Get branch-level prediction for this store
        br_jan = float(branch_pred.get(b, 0.0))
        
        # Regularization: Weight branch prediction more heavily for volatile stores
        # High volatility stores get pulled toward branch average for stability
        vol = float(store_metrics.loc[(store_metrics.Branch==b)&(store_metrics.Store==s), "volatility"].iloc[0])
        w_branch = np.clip(vol / 0.12, 0.15, 0.65)  # Branch weight: 15-65% based on volatility
        w_store = 1.0 - w_branch  # Store weight: 35-85%
        
        # Handle edge cases with NaN values
        if np.isnan(store_jan):
            store_jan = br_jan if not np.isnan(br_jan) else 0.0
        if np.isnan(br_jan):
            br_jan = store_jan
            
        # Blend store and branch predictions based on regularization weights
        pred = w_store * store_jan + w_branch * br_jan

        # Ensure prediction is within plausible bounds (0% to 100%)
        pred = float(np.clip(pred, 0.0, 1.0))

        # Calculate confidence score: combination of stability and data volume
        # Lower volatility = higher confidence; more data points = higher confidence
        n = len(d)  # Number of data points for this store
        conf = float(np.clip(1.0 - (vol / 0.18), 0.0, 1.0)) * (0.7 + 0.3 * min(1.0, n/5))
        
        # Store the forecast results with supporting metrics
        rows.append({
            "Branch": b,
            "Store": s,
            "Jan_Pred_AttachPct": pred,  # Predicted January attach %
            "Model_Confidence": conf,  # Confidence score (0-1)
            "Store_Trend_Slope": float(store_metrics.loc[(store_metrics.Branch==b)&(store_metrics.Store==s), "slope"].iloc[0]),  # Monthly trend
            "Volatility": vol,  # Historical volatility
            "Last_Month_AttachPct": float(d.sort_values("t")["AttachPct"].iloc[-1]),  # December attach %
        })
    pred_df = pd.DataFrame(rows)
    return pred_df

def fmt_pct(x):
    """Convert decimal number (0.23) to percentage string (23.0%)"""
    if pd.isna(x):
        return "—"  # Display dash for missing values
    return f"{x*100:.1f}%"  # Format to 1 decimal place

def kpi(label, value, delta=None, help_text=None):
    """Display a styled KPI card with metric name, value, and optional change indicator"""
    # Add help icon with tooltip if help text provided
    help_icon = f"<span title='{help_text}' style='margin-left: 0.3rem; color: var(--muted); cursor: help;'>ⓘ</span>" if help_text else ""
    # Add delta (change) row if provided
    d_html = f"<div class='delta'>{delta}</div>" if delta is not None else ""
    st.markdown(
        f"""
        <div class="kpi">
          <div class="label">{label}{help_icon}</div>
          <div class="value">{value}</div>
          {d_html}
        </div>
        """,
        unsafe_allow_html=True
    )

def section_header(title, description=None, icon=""):
    """Create a formatted section heading for better visual hierarchy"""
    html = f"""
    <div class='section-header'>
        <h3 style='color: var(--text); margin-bottom: 0.25rem;'>
            {title}
        </h3>
        {f"<p class='small-muted' style='margin-top: 0;'>{description}</p>" if description else ""}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def tooltip(text, icon="ⓘ"):
    """Generate an HTML tooltip icon with help text on hover"""
    return f'<span title="{text}" style="color: var(--muted); cursor: help; margin-left: 0.3rem;">{icon}</span>'

def styled_dataframe(df, height=300, download_filename=None):
    """Render dataframe with custom styling, tooltips, and consistent formatting"""
    # Display the dataframe
    result = st.dataframe(
            df,
            width='stretch',  # Use full available width
            hide_index=True,  # Don't show row numbers
            height=height,  # Set display height
            column_config={
                col: st.column_config.Column(
                    help=f"Column: {col}"  # Add column name as tooltip
                ) for col in df.columns
            }
        )
    
    # Add download button below the table if filename is provided
    if download_filename:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=download_filename,
            mime="text/csv",
            type="secondary"
        )
    
    return result


# ======================== SIDEBAR CONFIGURATION ========================
# Set up navigation controls and data upload options in the sidebar

if LOGO_B64:
    # Display company logo at top of sidebar
    st.sidebar.image(f"data:image/svg+xml;base64,{LOGO_B64}", width='stretch')
    st.sidebar.markdown("<div class='hr'></div>", unsafe_allow_html=True)

st.sidebar.markdown("## Controls")
st.sidebar.markdown("<div class='small-muted'>Upload the given sheet or use the bundled sample (converted from .xls).</div>", unsafe_allow_html=True)

# Set default path for sample data file
default_path = Path(__file__).parent / "data" / "Jumbo_Attach_Sample.xlsx"
# Toggle to use sample data or upload custom file
use_sample = st.sidebar.toggle("Use sample file", value=True)

# File uploader widget (only shown if not using sample)
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"]) if not use_sample else None

st.sidebar.markdown("<div class='hr'></div>", unsafe_allow_html=True)
st.sidebar.markdown("### Navigation")
# Radio button for page selection - controls which dashboard page to display
page = st.sidebar.radio(
    "Page Selection",
    ["Executive Summary", 
     "Branch & Month Insights", 
     "Store Deep Dive", 
     "Store Segments", 
     "Forecast: January", 
     "Download Pack"],
    index=0,  # Default to Executive Summary
    label_visibility="collapsed"
)

st.sidebar.markdown("<div class='hr'></div>", unsafe_allow_html=True)
st.sidebar.markdown("### Data Controls")
st.sidebar.markdown("<div class='small-muted'>Upload your data file or use the provided sample dataset.</div>", unsafe_allow_html=True)

st.sidebar.markdown("<div class='hr'></div>", unsafe_allow_html=True)
st.sidebar.markdown("### What this app produces")
st.sidebar.markdown(
    """
- **Leaderboard** of top/bottom stores & branches
- **Month-over-month** movement and volatility
- **Smart store** categorization (segments)
- **Explainable Jan forecast** per store (with confidence)
- **One-click downloads** (clean data + forecast)
""",
)


# ======================== DATA LOADING & CACHING ========================
# Streamlit caching layer - avoids reprocessing data when user interacts with widgets
# This makes the app responsive and efficient

@st.cache_data(show_spinner=False)
def cached_load(file_path_or_bytes):
    return load_data(file_path_or_bytes)

@st.cache_data(show_spinner=False)
def cached_long(df_wide, month_cols):
    return to_long(df_wide, month_cols)

@st.cache_data(show_spinner=False)
def cached_metrics(long):
    return add_store_metrics(long)

@st.cache_data(show_spinner=False)
def cached_segments(metrics):
    return store_segmentation(metrics, n_clusters=4)

@st.cache_data(show_spinner=False)
def cached_pred(metrics, long):
    return predict_jan(metrics, long)

# Helper function to load data from either sample file or user upload
def get_df():
    """Load data from sample file or user-uploaded file"""
    if use_sample:
        # Load sample data if using default
        if not default_path.exists():
            raise FileNotFoundError("Sample file missing: app/data/Jumbo_Attach_Sample.xlsx")
        return cached_load(default_path)
    if uploaded is None:
        # No file selected and not using sample
        return None
    return cached_load(uploaded)

def load_with_progress():
    """Load data with user-friendly progress feedback"""
    with st.spinner("Loading and processing data..."):
        loaded = get_df()
        if loaded is None:
            # Prompt user to upload or enable sample
            st.info("Upload an .xlsx file to begin, or toggle **Use sample file**.")
            st.stop()  # Stop execution until data is provided
        return loaded

# ======================== DATA PROCESSING PIPELINE ========================
# Execute the complete data transformation pipeline with caching at each step
# This creates a dependency chain: load → long format → metrics → segments → forecast

df_wide, month_cols = load_with_progress()  # Load and validate raw data
long = cached_long(df_wide, month_cols)  # Transform to time-series format
store_metrics = cached_metrics(long)  # Calculate performance metrics
segments = cached_segments(store_metrics)  # Group stores into segments
# Generate forecasts and merge with segment assignments
pred = cached_pred(store_metrics, long).merge(segments[["Branch","Store","segment"]], on=["Branch","Store"], how="left")


# ======================== PAGE HEADER ========================
# Display main title and subtitle

st.markdown("### 📊 Jumbo & Company — Device Insurance Attach% Analytics")
st.markdown(
    "<p class='small-muted' style='margin-top: -0.5rem; margin-bottom: 0.8rem; font-size: 0.8rem;'>Interactive dashboard for store performance analysis, segmentation, and forecasting</p>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style='display: flex; flex-wrap: wrap; gap: 0.3rem; margin-bottom: 0.8rem;'>
        <span class='badge'>📈 Executive Insights</span>
        <span class='badge'>🏪 Branch & Store Analysis</span>
        <span class='badge'>🎯 Store Segmentation</span>
        <span class='badge'>🔮 January Forecast</span>
        <span class='badge'>📥 Export Ready</span>
    </div>
    """,
    unsafe_allow_html=True
)

# ======================== GLOBAL KEY PERFORMANCE INDICATORS ========================
# Display top-level metrics that summarize the entire dataset
# These KPIs appear on every page for quick overview

c1, c2, c3, c4 = st.columns(4)

# Calculate key metrics
overall_avg = float(long["AttachPct"].mean())  # Average attach % across all data
mo_last = long[long["Month"] == month_cols[-1]]["AttachPct"].mean()  # December average
mo_first = long[long["Month"] == month_cols[0]]["AttachPct"].mean()  # August average
delta = mo_last - mo_first  # Month-over-month change

with c1:
    kpi("Overall Avg Attach%", 
        fmt_pct(overall_avg), 
        f"Aug→Dec change: {fmt_pct(delta)}",
        "Average attach rate across all stores and months")
with c2:
    kpi("Stores Covered", 
        f"{long['Store'].nunique():,}",
        help_text="Total number of unique stores in dataset")
with c3:
    kpi("Branches Covered", 
        f"{long['Branch'].nunique():,}",
        help_text="Total number of unique branches")
with c4:
    best_store = store_metrics.sort_values("avg", ascending=False).iloc[0]
    kpi("Best Performing Store", 
        f"{best_store['Store']}", 
        f"{best_store['Branch']} • Avg {fmt_pct(best_store['avg'])}",
        "Store with highest average attach rate")

st.markdown("<div class='hr' style='margin: 0.5rem 0;'></div>", unsafe_allow_html=True)


# ======================== PAGE ROUTING ========================
# Display different content based on user's page selection
# Each page section contains specific analysis and visualizations

if page == "Executive Summary":
    # ======================== EXECUTIVE SUMMARY PAGE ========================
    # High-level overview page showing key insights and performance rankings
    st.markdown(
        """
        <div class='breadcrumb'>
            <span class='small-muted'>Navigation:</span> 
            <strong>Executive Summary</strong>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    left, right = st.columns([1.2, 1.0], gap="large")

    with left:
        section_header("What's Happening", "Key insights and branch performance at a glance")
        st.markdown(
            """
            This dashboard is built to help a business reader quickly answer:
            - **Which branches and stores** are driving attach%?
            - **Where are we losing** attach% (declining trends / volatile stores)?
            - **Which store cohorts** need different actions?
            - **What is the likely attach%** for **January** store-by-store?
            """
        )

        # Branch leaderboard
        branch_tbl = long.groupby("Branch")["AttachPct"].mean().reset_index().sort_values("AttachPct", ascending=False)
        branch_tbl["AttachPct"] = branch_tbl["AttachPct"].apply(fmt_pct)
        branch_tbl.columns = ["Branch", "Avg Attach%"]
        section_header("Branch Leaderboard", "Top performing branches by average attach rate")
        styled_dataframe(branch_tbl, height=250, download_filename="branch_leaderboard.csv")

    with right:
        section_header("Quick Wins & Watchouts", "Stores showing significant improvement or decline")
        
        # Top risers and fallers by slope
        tmp = store_metrics.copy()
        tmp["avg_fmt"] = tmp["avg"].apply(fmt_pct)
        tmp["slope_fmt"] = tmp["slope"].apply(lambda x: f"{x*100:+.2f} pp / month")
        risers = tmp.sort_values("slope", ascending=False).head(7)[["Branch","Store","avg_fmt","slope_fmt"]]
        risers.columns = ["Branch", "Store", "Avg Attach%", "Monthly Trend"]
        
        fallers = tmp.sort_values("slope", ascending=True).head(7)[["Branch","Store","avg_fmt","slope_fmt"]]
        fallers.columns = ["Branch", "Store", "Avg Attach%", "Monthly Trend"]

        section_header("Risers (Improving Stores)")
        styled_dataframe(risers, height=200, download_filename="improving_stores.csv")
        
        section_header("Watchouts (Declining Stores)")
        styled_dataframe(fallers, height=200, download_filename="declining_stores.csv")

    section_header("Month-over-month Distribution", "Box plot showing attach rate distribution by month")
    if px:
        fig = px.box(long, x="Month", y="AttachPct", points="all")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(
            margin=dict(l=10,r=10,t=40,b=10), 
            height=420,
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font=dict(color='#1a1a1a')
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Plotly not available in this environment. Install plotly for interactive charts.")

elif page == "Branch & Month Insights":
    # ======================== BRANCH & MONTH INSIGHTS PAGE ========================
    # Analyze monthly trends and performance patterns across branches
    st.markdown(
        """
        <div class='breadcrumb'>
            <span class='small-muted'>Navigation:</span> 
            <a href='#' onclick='window.location.reload()'>Home</a> 
            <span class='small-muted'>→</span> 
            <strong>Branch & Month Insights</strong>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    section_header("Branch Performance by Month", "Analyze monthly trends and performance across branches")
    
    with st.expander("🔍 Filter Options", expanded=False):
        bcol1, bcol2, bcol3 = st.columns([1,1,1])
        branches = sorted(long["Branch"].unique().tolist())
        sel_branches = bcol1.multiselect("Filter branches", branches, default=branches[:3])
        sel_months = bcol2.multiselect("Filter months", month_cols, default=month_cols)
        view = bcol3.selectbox("View", ["Trend line", "Heatmap table"], index=0)

    sub = long[long["Branch"].isin(sel_branches) & long["Month"].isin(sel_months)].copy()

    if view == "Trend line":
        if px:
            br_m = sub.groupby(["Branch","Month"])["AttachPct"].mean().reset_index()
            br_m["Month"] = pd.Categorical(br_m["Month"], categories=MONTHS_ORDER, ordered=True)
            fig = px.line(br_m.sort_values("Month"), x="Month", y="AttachPct", color="Branch", markers=True)
            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(
                height=460, 
                margin=dict(l=10,r=10,t=40,b=10),
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font=dict(color='#1a1a1a')
            )
            st.plotly_chart(fig, width='stretch')
        else:
            styled_dataframe(sub.head(30))
    else:
        pivot = sub.pivot_table(index="Branch", columns="Month", values="AttachPct", aggfunc="mean")
        pivot = pivot.reindex(columns=MONTHS_ORDER).round(4)
        # Remove any completely empty rows
        pivot = pivot.dropna(how='all')
        
        # Display the heatmap table
        st.dataframe(pivot.style.format("{:.1%}"), width='stretch', height=400)
        
        # Add download button below the table
        csv_data = pivot.to_csv().encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="branch_month_heatmap.csv",
            mime="text/csv",
            type="secondary"
        )

    section_header("Branch Health Analysis", "Actionable insights by branch performance", "🏥")
    br = long.groupby("Branch")["AttachPct"].agg(avg="mean", std="std").reset_index()
    br["std"] = br["std"].fillna(0)
    br["Health"] = np.where(br["avg"] >= br["avg"].quantile(0.75), "Strong 🟢",
                     np.where(br["avg"] >= br["avg"].median(), "Stable 🟡", "Needs Focus 🔴"))
    br = br.sort_values(["Health","avg"], ascending=[True, False])
    show = br.copy()
    show["Avg Attach%"] = show["avg"].apply(fmt_pct)
    show["Volatility"] = show["std"].apply(lambda x: f"{x*100:.1f} pp")
    show["Health"] = show["Health"]
    styled_dataframe(show[["Branch", "Avg Attach%", "Volatility", "Health"]], height=300, download_filename="branch_health_analysis.csv")

elif page == "Store Deep Dive":
    # ======================== STORE DEEP DIVE PAGE ========================
    # Drill down into individual store performance with trend analysis and comparisons
    st.markdown(
        """
        <div class='breadcrumb'>
            <span class='small-muted'>Navigation:</span> 
            <a href='#' onclick='window.location.reload()'>Home</a> 
            <span class='small-muted'>→</span> 
            <strong>Store Deep Dive</strong>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    section_header("Store Explorer", "Drill down into individual store performance")
    
    with st.expander("Filter & View Options", expanded=True):
        cA, cB, cC = st.columns([1.1, 1.2, 1.0])
        branches = sorted(long["Branch"].unique().tolist())
        sel_branch = cA.selectbox("Branch", ["All"] + branches, index=0)

        stores = sorted(long[long["Branch"].eq(sel_branch)]["Store"].unique().tolist()) if sel_branch != "All" else sorted(long["Store"].unique().tolist())
        sel_store = cB.selectbox("Store", stores, index=0)

        metric_view = cC.selectbox("Focus View", ["Trend", "Compare to branch", "Volatility"], index=0)

    sub = long[(long["Store"] == sel_store)].copy()
    store_row = store_metrics[store_metrics["Store"] == sel_store].iloc[0]
    
    section_header(f"{sel_store}", f"Branch: {store_row['Branch']}")
    
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Avg Attach%", fmt_pct(store_row["avg"]), help="Average attach rate across all months")
    s2.metric("Min→Max Range", f"{fmt_pct(store_row['min'])} → {fmt_pct(store_row['max'])}", help="Performance range across months")
    s3.metric("Monthly Trend", f"{store_row['slope']*100:+.2f} pp / month", help="Positive = improving, Negative = declining")
    s4.metric("Volatility", f"{store_row['volatility']*100:.1f} pp", help="Standard deviation of attach rates")

    if px:
        if metric_view == "Trend":
            sub["Month"] = pd.Categorical(sub["Month"], categories=MONTHS_ORDER, ordered=True)
            fig = px.line(sub.sort_values("Month"), x="Month", y="AttachPct", markers=True)
            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(
                height=420, 
                margin=dict(l=10,r=10,t=40,b=10),
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font=dict(color='#1a1a1a')
            )
            st.plotly_chart(fig, width='stretch')

        elif metric_view == "Compare to branch":
            b = store_row["Branch"]
            br_series = long[long["Branch"] == b].groupby("Month")["AttachPct"].mean().reset_index()
            br_series["Series"] = f"{b} average"
            st_series = sub.groupby("Month")["AttachPct"].mean().reset_index()
            st_series["Series"] = sel_store
            both = pd.concat([br_series, st_series], ignore_index=True)
            both["Month"] = pd.Categorical(both["Month"], categories=MONTHS_ORDER, ordered=True)
            fig = px.line(both.sort_values("Month"), x="Month", y="AttachPct", color="Series", markers=True)
            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(
                height=420, 
                margin=dict(l=10,r=10,t=40,b=10),
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font=dict(color='#1a1a1a')
            )
            st.plotly_chart(fig, width='stretch')

        else:
            # volatility view: bar of month-to-month deltas
            sub = sub.sort_values("t")
            sub["MoM Change"] = sub["AttachPct"].diff()
            fig = px.bar(sub, x="Month", y="MoM Change")
            fig.update_yaxes(tickformat="+.0%")
            fig.update_layout(
                height=420, 
                margin=dict(l=10,r=10,t=40,b=10),
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font=dict(color='#1a1a1a')
            )
            st.plotly_chart(fig, width='stretch')
    else:
        styled_dataframe(sub)

    section_header("Store Leaderboard", "Rank stores by different performance metrics")
    
    with st.expander("Ranking Options", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            topn = st.slider("Show top/bottom N stores", 5, 30, 12)
        with col2:
            by = st.selectbox("Rank by", ["Average Attach%", "Trend", "Volatility"], 
                             index=0, format_func=lambda x: {
                                 "Average Attach%": "avg",
                                 "Trend": "slope", 
                                 "Volatility": "volatility"
                             }[x])
    
    # Map display names to column names
    by_col = {"Average Attach%": "avg", "Trend": "slope", "Volatility": "volatility"}[by]
    
    tbl = store_metrics.merge(segments[["Branch","Store","segment"]], on=["Branch","Store"], how="left")
    tbl = tbl.sort_values(by_col, ascending=False)

    show = tbl[["Branch","Store","segment","avg","slope","volatility","last_attach"]].copy()
    show["Avg Attach%"] = show["avg"].apply(fmt_pct)
    show["Last Month"] = show["last_attach"].apply(fmt_pct)
    show["Monthly Trend"] = show["slope"].apply(lambda x: f"{x*100:+.2f} pp/m")
    show["Volatility"] = show["volatility"].apply(lambda x: f"{x*100:.1f} pp")
    
    styled_dataframe(show[["Branch", "Store", "segment", "Avg Attach%", "Monthly Trend", "Volatility", "Last Month"]].head(topn), height=350, download_filename="store_leaderboard.csv")

elif page == "Store Segments":
    # ======================== STORE SEGMENTS PAGE ========================
    # View store categorization based on performance metrics for targeted strategies
    st.markdown(
        """
        <div class='breadcrumb'>
            <span class='small-muted'>Navigation:</span> 
            <a href='#' onclick='window.location.reload()'>Home</a> 
            <span class='small-muted'>→</span> 
            <strong>Store Segments</strong>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    section_header("Store Categorization (Segments)", "Group stores by performance characteristics for targeted actions")
    st.markdown(
        """
        Stores are grouped using **K-Means clustering** on three explainable features:
        1. **Average attach%** - Overall performance level
        2. **Volatility** - Consistency of performance
        3. **Trend slope** - Direction of performance
        
        This turns hundreds of stores into 3–5 action-oriented cohorts.
        """
    )

    seg_counts = segments["segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Number of Stores"]
    if px:
        fig = px.bar(seg_counts, x="Segment", y="Number of Stores")
        fig.update_layout(
            height=360, 
            margin=dict(l=10,r=10,t=40,b=10),
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font=dict(color='#1a1a1a')
        )
        st.plotly_chart(fig, width='stretch')
    
    section_header("Segment Distribution", "Number of stores in each segment")
    styled_dataframe(seg_counts, height=150, download_filename="segment_distribution.csv")

    section_header("Segment Playbook (Recommended Actions)", "What to do for each store segment")
    st.markdown(
        """
- **Champions (High & improving):** Replicate scripts, incentives, and promoter staff. *Best practice sharing opportunities.*
          
- **At-risk (High but falling):** Investigate churn in sales staff, stock-outs, counter practices. *Refresh pitch and retrain staff.*
          
- **Risers (Low but improving):** Double down on training and nudges. *Best ROI cohort for incremental investment.*
          
- **Long tail (Low & flat):** Consider targeted interventions (bundles), or reduce effort and focus on big wins. *Evaluate store viability.*
        """
    )

    section_header("Segment Details", "View all stores with their segment assignments")
    
    seg = segments.copy()
    seg["Avg Attach%"] = seg["avg"].apply(fmt_pct)
    seg["Trend"] = seg["slope"].apply(lambda x: f"{x*100:+.2f} pp/m")
    seg["Volatility"] = seg["volatility"].apply(lambda x: f"{x*100:.1f} pp")
    seg["Last Month"] = seg["last_attach"].apply(fmt_pct)
    
    styled_dataframe(seg[["Branch","Store","segment","Avg Attach%","Trend","Volatility","Last Month"]], height=400, download_filename="store_segments_details.csv")

elif page == "Forecast: January":
    # ======================== FORECAST: JANUARY PAGE ========================
    # Predict store-level attach% for January with confidence scoring
    st.markdown(
        """
        <div class='breadcrumb'>
            <span class='small-muted'>Navigation:</span> 
            <a href='#' onclick='window.location.reload()'>Home</a> 
            <span class='small-muted'>→</span> 
            <strong>Forecast: January</strong>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    section_header("January Attach% Prediction (Store-level)", "Forecast using trend analysis and branch regularization")
    st.markdown(
        """
        **Forecast methodology:**
        - Uses **trend extrapolation** per store (Aug-Dec → Jan)
        - **Regularized by branch trend** for stability
        - **Confidence score** based on volatility and data points
        - **Simple and explainable** for business discussions
        """
    )

    with st.expander("Filter & Sort Options", expanded=False):
        f1, f2, f3 = st.columns([1.1,1.1,1.0])
        branches = sorted(pred["Branch"].unique().tolist())
        sel_br = f1.selectbox("Branch filter", ["All"] + branches, index=0)
        segs = sorted(pred["segment"].dropna().unique().tolist())
        sel_seg = f2.selectbox("Segment filter", ["All"] + segs, index=0)
        sort_by_display = f3.selectbox("Sort by", 
                                     ["Jan Prediction", "Model Confidence", "Store Trend", "Volatility"], 
                                     index=0)
        
        # Map display names to column names
        sort_by_map = {
            "Jan Prediction": "Jan_Pred_AttachPct",
            "Model Confidence": "Model_Confidence",
            "Store Trend": "Store_Trend_Slope",
            "Volatility": "Volatility"
        }
        sort_by = sort_by_map[sort_by_display]

    sub = pred.copy()
    if sel_br != "All":
        sub = sub[sub["Branch"] == sel_br]
    if sel_seg != "All":
        sub = sub[sub["segment"] == sel_seg]

    sub = sub.sort_values(sort_by, ascending=False)

    view = sub[["Branch","Store","segment","Jan_Pred_AttachPct","Model_Confidence","Last_Month_AttachPct","Store_Trend_Slope","Volatility"]].copy()
    view["Jan Prediction"] = view["Jan_Pred_AttachPct"].apply(fmt_pct)
    view["Last Month"] = view["Last_Month_AttachPct"].apply(fmt_pct)
    view["Confidence"] = view["Model_Confidence"].apply(lambda x: f"{x*100:.0f}%")
    view["Trend"] = view["Store_Trend_Slope"].apply(lambda x: f"{x*100:+.2f} pp/m")
    view["Volatility"] = view["Volatility"].apply(lambda x: f"{x*100:.1f} pp")
    
    styled_dataframe(view[["Branch","Store","segment","Jan Prediction","Confidence","Last Month","Trend","Volatility"]], height=400, download_filename="january_forecast.csv")

    if px:
        section_header("Top Predictions Visualization", "Bar chart of top predicted stores")
        topk = st.slider("Number of stores to plot", 10, 50, 20, key="topk_slider")
        plot_df = sub.head(topk).copy()
        plot_df["Jan_Pred_AttachPct_pct"] = plot_df["Jan_Pred_AttachPct"] * 100
        fig = px.bar(plot_df, x="Store", y="Jan_Pred_AttachPct_pct", color="Branch")
        fig.update_layout(
            height=460, 
            margin=dict(l=10,r=10,t=40,b=10),
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font=dict(color='#1a1a1a')
        )
        fig.update_yaxes(title="Jan predicted attach%")
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, width='stretch')

elif page == "Download Pack":
    # ======================== DOWNLOAD PACK PAGE ========================
    # Allow users to export processed data in various formats
    
    # Add breadcrumb navigation
    st.markdown(
        """
        <div class='breadcrumb'>
            <span class='small-muted'>Navigation:</span> 
            <a href='#' onclick='window.location.reload()'>Home</a> 
            <span class='small-muted'>→</span> 
            <strong>Download Pack</strong>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    section_header("Data Export", "Download cleaned data, metrics, segments, and forecasts")
    st.markdown("<div class='small-muted'>Export the analysis results for reporting or further analysis.</div>", unsafe_allow_html=True)

    # Prepare data for export - clean and format datasets
    long_export = long.copy()
    long_export["AttachPct"] = long_export["AttachPct"].round(6)  # Round to 6 decimals for export

    # Merge segments with forecast predictions for comprehensive export
    metrics_export = segments.merge(pred[["Branch","Store","Jan_Pred_AttachPct","Model_Confidence"]], on=["Branch","Store"], how="left")
    metrics_export = metrics_export.rename(columns={"Jan_Pred_AttachPct":"Jan_Pred"})

    # Helper function to convert dataframe to CSV bytes for download
    def to_bytes(df):
        """Convert dataframe to CSV format bytes"""
        return df.to_csv(index=False).encode("utf-8")

    # ======================== INDIVIDUAL FILE DOWNLOADS ========================
    # Provide option to download individual CSV files
    section_header("Individual CSV Downloads", "Download specific datasets as CSV files")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            "⬇️ Long Format Data", 
            data=to_bytes(long_export), 
            file_name="attach_long_format.csv", 
            mime="text/csv",
            help="Month-by-month attach rates for all stores"
        )
    
    with col2:
        st.download_button(
            "⬇️ Store Metrics & Segments", 
            data=to_bytes(metrics_export), 
            file_name="store_segments_metrics.csv", 
            mime="text/csv",
            help="Store-level metrics with segment assignments"
        )
    
    with col3:
        st.download_button(
            "⬇️ January Forecast", 
            data=to_bytes(pred), 
            file_name="january_forecast.csv", 
            mime="text/csv",
            help="January predictions with confidence scores"
        )

    # ======================== EXCEL WORKBOOK EXPORT ========================
    # Provide comprehensive export with all analysis results in single Excel file
    section_header("Complete Excel Workbook", "Single file with all analysis results")
    st.markdown("<div class='small-muted'>Comprehensive Excel file with multiple sheets for easy sharing.</div>", unsafe_allow_html=True)
    
    try:
        import io
        # Create Excel workbook in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            # Export each dataset to separate sheet with descriptive names
            df_wide.to_excel(writer, index=False, sheet_name="01_Raw_Data")  # Original data
            long_export.to_excel(writer, index=False, sheet_name="02_Long_Format")  # Time-series format
            store_metrics.to_excel(writer, index=False, sheet_name="03_Store_Metrics")  # Calculated metrics
            segments.to_excel(writer, index=False, sheet_name="04_Segments")  # Segment assignments
            pred.to_excel(writer, index=False, sheet_name="05_January_Forecast")  # Forecast predictions
        
        st.download_button(
            "⬇️ Download Complete Excel Pack", 
            data=output.getvalue(), 
            file_name="attach_insights_pack.xlsx",
            help="All data in a single Excel file with multiple sheets"
        )
    except Exception as e:
        st.warning("Excel export requires openpyxl. Install with: `pip install openpyxl`")
        st.info("Using sample data? CSV downloads are still available.")