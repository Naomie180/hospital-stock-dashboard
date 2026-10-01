import streamlit as st
import pandas as pd
import plotly.express as px
import io

st.set_page_config(
    page_title="Hospital Stock Movement Dashboard",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Hospital Stock Movement Dashboard")
st.markdown("**Main Store Analysis | June – September 2026**")

# ========== LOAD DATA ==========
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    df["prev_stock_qty"] = pd.to_numeric(df["prev_stock_qty"], errors="coerce").fillna(0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    return df

uploaded_file = st.sidebar.file_uploader("Upload Stock Movement Excel", type=["xlsx"])

if uploaded_file is None:
    st.info("👈 Please upload the Excel file from the sidebar to continue.")
    st.stop()

df = load_data(uploaded_file)

# ========== SIDEBAR FILTER ==========
min_date = df["date"].min().date()
max_date = df["date"].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start, end = date_range
    df_filtered = df[(df["date"].dt.date >= start) & (df["date"].dt.date <= end)].copy()
else:
    df_filtered = df.copy()

st.sidebar.success(f"Showing **{len(df_filtered):,}** records")

# ========== TABS ==========
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🚀 Fast Moving",
    "🏬 By Department",
    "📅 Over Time",
    "📦 Still in Main Store",
    "🛒 Purchases",
    "📊 Insights"
])

# ----- TAB 1: Fast Moving -----
with tab1:
    st.header("1. Fastest Moving Products")
    
    fast = (
        df_filtered[df_filtered["quantity"] < 0]
        .groupby(["item_code", "item_name"])
        .agg(
            total_qty_out=("quantity", lambda x: -x.sum()),
            times_moved=("quantity", "count")
        )
        .reset_index()
        .sort_values("total_qty_out", ascending=False)
    )
    
    top_n = st.slider("Show Top N products", 10, 50, 20)
    top_fast = fast.head(top_n)
    
    fig = px.bar(
        top_fast,
        x="total_qty_out",
        y="item_name",
        orientation="h",
        title=f"Top {top_n} Fastest Moving Items",
        color="total_qty_out",
        color_continuous_scale="Blues"
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(top_fast, use_container_width=True, hide_index=True)

# ----- TAB 2: By Department -----
with tab2:
    st.header("2. Department Usage")
    
    dept = (
        df_filtered[
            (df_filtered["purpose"] == "Transfer") &
            (df_filtered["destination_location"].notna())
        ]
        .groupby("destination_location")
        .agg(
            total_qty=("quantity", lambda x: -x.sum()),
            transfers=("quantity", "count")
        )
        .reset_index()
        .sort_values("total_qty", ascending=False)
    )
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(dept, values="total_qty", names="destination_location",
                     title="Share by Department", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.bar(dept, x="destination_location", y="total_qty",
                      title="Quantity by Department", color="total_qty")
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)
    
    st.dataframe(dept, use_container_width=True, hide_index=True)

# ----- TAB 3: Over Time -----
with tab3:
    st.header("3. Consumption Over Time")
    
    monthly = (
        df_filtered[df_filtered["quantity"] < 0]
        .assign(month=lambda x: x["date"].dt.to_period("M").astype(str))
        .groupby("month")
        .agg(total_issued=("quantity", lambda x: -x.sum()))
        .reset_index()
    )
    
    fig = px.line(monthly, x="month", y="total_issued", markers=True,
                  title="Total Quantity Issued per Month")
    st.plotly_chart(fig, use_container_width=True)

# ----- TAB 4: Still in Main Store -----
with tab4:
    st.header("4. Items Still Remaining in Main Store")
    
    current_stock = (
        df_filtered
        .sort_values(["item_code", "date", "reference"])
        .groupby(["item_code", "item_name"])
        .agg(
            last_prev=("prev_stock_qty", "last"),
            last_qty=("quantity", "last"),
            last_price=("price", "last"),
            last_date=("date", "last")
        )
        .reset_index()
    )
    current_stock["ending_stock"] = current_stock["last_prev"] + current_stock["last_qty"]
    
    still_in = current_stock[current_stock["ending_stock"] > 0].sort_values("ending_stock", ascending=False)
    zero_stock = current_stock[current_stock["ending_stock"] == 0]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Still in Main Store", f"{len(still_in):,}")
    col2.metric("Zero Stock", f"{len(zero_stock):,}")
    col3.metric("Total Unique Items", f"{len(current_stock):,}")
    
    st.dataframe(
        still_in[["item_code", "item_name", "ending_stock", "last_price", "last_date"]],
        use_container_width=True,
        hide_index=True,
        height=500
    )
    
    csv = still_in.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Still in Main Store (CSV)",
        data=csv,
        file_name="items_still_in_main_store.csv",
        mime="text/csv"
    )

# ----- TAB 5: Purchases -----
with tab5:
    st.header("5. Purchases Summary")
    
    purchases = df_filtered[
        (df_filtered["Stock_Entry"] == "Purchase") &
        (df_filtered["quantity"] > 0)
    ]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Purchase Transactions", f"{len(purchases):,}")
    col2.metric("Total Units Purchased", f"{purchases['quantity'].sum():,.0f}")
    col3.metric("Total Purchase Value", f"{purchases['amount'].sum():,.0f}")
    
    top_purch = (
        purchases
        .groupby(["item_code", "item_name"])
        .agg(qty=("quantity", "sum"), value=("amount", "sum"))
        .reset_index()
        .sort_values("qty", ascending=False)
        .head(20)
    )
    st.dataframe(top_purch, use_container_width=True, hide_index=True)

# ----- TAB 6: Insights -----
with tab6:
    st.header("6. Insights & Recommendations")
    
    st.subheader("✅ Improvements Visible")
    st.markdown("""
    - Regular transfers from Main Store to departments
    - Opening balances and purchases are recorded
    - Stock adjustments are being performed
    - Clear audit trail with reference numbers
    """)
    
    st.subheader("⚠️ Risks Identified")
    st.markdown("""
    1. Some destination locations are staff names instead of proper stores
    2. Zero-stock items still appear in transfer requests
    3. High-value items need closer monitoring
    """)
    
    st.subheader("📌 Recommendations")
    st.markdown("""
    - Set min/max stock levels for top 50 fast movers
    - Enforce proper store names in destination field
    - Weekly review of zero-stock items
    - Monthly physical counts for high-value items
    """)

st.markdown("---")
st.caption("Hospital Stock Movement Dashboard | Built with Streamlit")