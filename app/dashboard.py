import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
from prophet import Prophet
import os

st.set_page_config(page_title="RetailPulse AI", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ── Load Data ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "superstore.csv")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, encoding="latin1")
    df.columns = df.columns.str.strip().str.rstrip(";")
    df[df.columns[-1]] = df[df.columns[-1]].astype(str).str.rstrip(";").astype(float)
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    return df

df = load_data()

# ── Sidebar Filters ────────────────────────────────────────────────────────────
st.sidebar.header("🔎 Filters")

all_regions = sorted(df["Region"].dropna().unique().tolist())
selected_regions = st.sidebar.multiselect("Region", all_regions, default=all_regions)

all_categories = sorted(df["Category"].dropna().unique().tolist())
selected_categories = st.sidebar.multiselect("Category", all_categories, default=all_categories)

min_date = df["Order Date"].min().date()
max_date = df["Order Date"].max().date()
date_range = st.sidebar.date_input(
    "Order Date Range", value=(min_date, max_date),
    min_value=min_date, max_value=max_date
)

# Apply filters
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = df[
        (df["Region"].isin(selected_regions)) &
        (df["Category"].isin(selected_categories)) &
        (df["Order Date"].dt.date >= start_date) &
        (df["Order Date"].dt.date <= end_date)
    ]
else:
    filtered_df = df[
        (df["Region"].isin(selected_regions)) &
        (df["Category"].isin(selected_categories))
    ]

if filtered_df.empty:
    st.warning("⚠️ No data matches your filters. Please adjust the sidebar.")
    st.stop()

# ── Title ──────────────────────────────────────────────────────────────────────
st.title("🛒 RetailPulse AI Dashboard")

# ── KPI Metrics (always visible) ──────────────────────────────────────────────
total_sales  = round(filtered_df["Sales"].sum(), 2)
total_profit = round(filtered_df["Profit"].sum(), 2)
total_orders = filtered_df["Order ID"].nunique()

col1, col2, col3 = st.columns(3)
col1.metric("Total Sales",  f"${total_sales:,.2f}")
col2.metric("Total Profit", f"${total_profit:,.2f}")
col3.metric("Total Orders", total_orders)

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔮 Forecasting", "🚨 Anomalies"])

# ── Tab 1: Overview ────────────────────────────────────────────────────────────
with tab1:
    # Monthly Sales Trend
    monthly_sales = (
        filtered_df.groupby(filtered_df["Order Date"].dt.to_period("M"))["Sales"]
        .sum().reset_index()
    )
    monthly_sales["Order Date"] = monthly_sales["Order Date"].astype(str)

    fig1 = px.line(
        monthly_sales, x="Order Date", y="Sales",
        title="📈 Monthly Sales Trend", markers=True
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Category Pie + Region Bar side by side
    col_a, col_b = st.columns(2)

    category_sales = filtered_df.groupby("Category")["Sales"].sum().reset_index()
    fig2 = px.pie(category_sales, names="Category", values="Sales", title="Sales by Category")
    col_a.plotly_chart(fig2, use_container_width=True)

    region_profit = filtered_df.groupby("Region")["Profit"].sum().reset_index()
    fig3 = px.bar(
        region_profit, x="Region", y="Profit",
        title="Profit by Region", color="Profit",
        color_continuous_scale="RdYlGn"
    )
    col_b.plotly_chart(fig3, use_container_width=True)

    # Executive Summary
    st.subheader("📋 Executive Summary")
    sales_growth = monthly_sales["Sales"].pct_change().mean() * 100
    region_label = ", ".join(selected_regions) if selected_regions else "All Regions"
    best_category = filtered_df.groupby("Category")["Sales"].sum().idxmax()
    worst_region  = filtered_df.groupby("Region")["Profit"].sum().idxmin()

    st.write(f"""
RetailPulse AI detected strong sales activity in the **{region_label}** region(s).

Average monthly sales growth: **{sales_growth:.2f}%**.

Best performing category: **{best_category}** | Weakest profitability region: **{worst_region}**
    """)

    # Download button
    st.divider()
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Filtered Data",
        data=csv,
        file_name="filtered_sales_data.csv",
        mime="text/csv"
    )

# ── Tab 2: Forecasting ─────────────────────────────────────────────────────────
with tab2:
    st.subheader("🔮 Sales Forecast (Next 6 Months)")

    forecast_data = (
        filtered_df.groupby(filtered_df["Order Date"].dt.to_period("M"))["Sales"]
        .sum().reset_index()
    )
    forecast_data["Order Date"] = forecast_data["Order Date"].astype(str)
    forecast_data.columns = ["ds", "y"]
    forecast_data["ds"] = pd.to_datetime(forecast_data["ds"])

    if len(forecast_data) >= 2:
        prophet_model = Prophet()
        prophet_model.fit(forecast_data)

        future = prophet_model.make_future_dataframe(periods=6, freq="ME")
        forecast = prophet_model.predict(future)

        fig_forecast = px.line(
            forecast, x="ds", y="yhat",
            title="Predicted Monthly Sales",
            labels={"ds": "Date", "yhat": "Forecasted Sales"}
        )
        fig_forecast.add_scatter(
            x=forecast_data["ds"], y=forecast_data["y"],
            mode="markers", name="Actual Sales",
            marker=dict(color="orange", size=6)
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
    else:
        st.warning("Not enough data to generate a forecast. Try selecting more regions or a wider date range.")

# ── Tab 3: Anomalies ───────────────────────────────────────────────────────────
with tab3:
    st.subheader("🚨 Anomaly Detection")

    anomaly_df = filtered_df[["Sales", "Profit"]].copy()
    iso_model = IsolationForest(contamination=0.02, random_state=42)
    filtered_df = filtered_df.copy()
    filtered_df["Anomaly"] = iso_model.fit_predict(anomaly_df)
    anomaly_count = (filtered_df["Anomaly"] == -1).sum()

    fig4 = px.scatter(
        filtered_df, x="Sales", y="Profit",
        color=filtered_df["Anomaly"].astype(str),
        title="Anomaly Detection (Red = Anomaly)",
        color_discrete_map={"1": "steelblue", "-1": "red"},
        hover_data=["Product Name", "Category", "Region"]
    )
    st.plotly_chart(fig4, use_container_width=True)

    # AI Insights
    st.subheader("💡 AI-Generated Insights")
    st.write(f"🚨 Anomalies detected: **{anomaly_count}** transactions")
    st.write(f"⚠️ Weakest profitability region: **{filtered_df.groupby('Region')['Profit'].sum().idxmin()}**")
    st.write(f"📈 Best performing category: **{filtered_df.groupby('Category')['Sales'].sum().idxmax()}**")

    # Show anomaly rows
    with st.expander("🔍 View Anomalous Transactions"):
        st.dataframe(
            filtered_df[filtered_df["Anomaly"] == -1]
            [["Order ID", "Product Name", "Category", "Region", "Sales", "Profit"]]
            .sort_values("Profit")
            .reset_index(drop=True)
        )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("RetailPulse AI • Built with Streamlit, Plotly, Scikit-learn & Prophet")