import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Future Stock Movement Predictor",
    page_icon="📦",
    layout="wide"
)

st.title("Future Stock Movement Predictor")
st.caption("Main Store → departments: products, quantities, and how often they will move")

# ----- Sidebar -----
st.sidebar.header("Settings")
horizon = st.sidebar.slider("How far ahead (days)", min_value=7, max_value=90, value=30, step=7)
min_days = st.sidebar.slider("Min days of history per product–department", 3, 20, 5)

uploaded = st.sidebar.file_uploader("Upload stock movement Excel", type=["xlsx"])

if uploaded is None:
    st.info("Upload a Main Store stock-movement Excel file in the sidebar to run future predictions.")
    st.stop()

@st.cache_data(show_spinner="Loading data...")
def load_df(file):
    df = pd.read_excel(file)
    df.columns = [str(c).strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    df["item_code"] = df["item_code"].astype(str).str.strip()
    df["item_name"] = df.get("item_name", df["item_code"]).astype(str).str.strip()
    if "destination_location" in df.columns:
        df["destination_location"] = (
            df["destination_location"].astype(str).str.strip()
            .replace({"nan": "Unknown", "": "Unknown"})
        )
    else:
        df["destination_location"] = "Unknown"
    return df

df = load_df(uploaded)
st.success(f"Loaded {len(df):,} rows  |  {df['date'].min().date()} → {df['date'].max().date()}")

# ----- Transfers out -----
transfers = df[df["quantity"] < 0].copy()
entry_col = None
for c in ["Stock_Entry", "purpose", "description"]:
    if c in transfers.columns:
        entry_col = c
        break
if entry_col is not None:
    mask = transfers[entry_col].astype(str).str.contains("Transfer", case=False, na=False)
    if mask.sum() > 0:
        transfers = transfers[mask].copy()

transfers["qty_out"] = -transfers["quantity"]

if len(transfers) < 50:
    st.error("Not enough transfer-out rows to build a forecast. Check quantity sign and Transfer labels.")
    st.stop()

# ----- Daily aggregates -----
daily = (
    transfers
    .groupby(["date", "item_code", "item_name", "destination_location"], as_index=False)
    .agg(qty=("qty_out", "sum"), times=("qty_out", "count"))
)

pair_n = daily.groupby(["item_code", "destination_location"]).size().reset_index(name="n_days")
active = pair_n[pair_n["n_days"] >= min_days][["item_code", "destination_location"]]
daily_f = daily.merge(active, on=["item_code", "destination_location"])

if len(daily_f) < 30:
    st.warning("Very few active pairs. Try lowering 'Min days of history' in the sidebar.")
    st.stop()

# ----- Features -----
daily_f = daily_f.sort_values(["item_code", "destination_location", "date"]).reset_index(drop=True)

def add_features(g):
    g = g.copy()
    g["qty_lag1"] = g["qty"].shift(1)
    g["qty_lag7"] = g["qty"].shift(7)
    g["qty_roll7"] = g["qty"].shift(1).rolling(7, min_periods=1).mean()
    g["times_lag1"] = g["times"].shift(1)
    g["times_roll7"] = g["times"].shift(1).rolling(7, min_periods=1).mean()
    g["dow"] = g["date"].dt.dayofweek
    g["month"] = g["date"].dt.month
    return g

feat = (
    daily_f.groupby(["item_code", "destination_location"], group_keys=False)
    .apply(add_features)
    .dropna(subset=["qty_lag1"])
)

feature_cols = ["qty_lag1", "qty_lag7", "qty_roll7", "times_lag1", "times_roll7", "dow", "month"]

# ----- Train (lighter for cloud speed) -----
@st.cache_resource(show_spinner="Training future prediction models...")
def train_models(feat_hash, X, y_qty, y_times):
    m_qty = RandomForestRegressor(n_estimators=60, max_depth=10, random_state=42, n_jobs=-1)
    m_times = RandomForestRegressor(n_estimators=40, max_depth=8, random_state=42, n_jobs=-1)
    m_qty.fit(X, y_qty)
    m_times.fit(X, y_times)
    return m_qty, m_times

X_all = feat[feature_cols]
model_qty, model_times = train_models(
    hash(tuple(feat["date"].astype(str).head(5))),
    X_all, feat["qty"], feat["times"]
)

# ----- Forecast -----
@st.cache_data(show_spinner="Computing future predictions...")
def run_forecast(_feat, _daily_f, horizon, feature_cols):
    last_rows = (
        _feat.sort_values("date")
        .groupby(["item_code", "destination_location"], as_index=False)
        .tail(1)
        .copy()
    )
    name_map = (
        _daily_f[["item_code", "item_name"]]
        .drop_duplicates("item_code")
        .set_index("item_code")["item_name"]
        .to_dict()
    )
    last_rows["item_name"] = last_rows["item_code"].map(name_map)
    last_rows["item_name"] = last_rows["item_name"].fillna(last_rows["item_code"])

    # Cap pairs for speed on free Streamlit Cloud
    last_rows = last_rows.head(400)

    forecasts = []
    for _, row in last_rows.iterrows():
        state = {
            "qty_lag1": float(row["qty"]),
            "qty_lag7": float(row["qty_lag7"]) if pd.notna(row.get("qty_lag7")) else float(row["qty"]),
            "qty_roll7": float(row["qty_roll7"]) if pd.notna(row.get("qty_roll7")) else float(row["qty"]),
            "times_lag1": float(row["times"]),
            "times_roll7": float(row["times_roll7"]) if pd.notna(row.get("times_roll7")) else float(row["times"]),
            "dow": 0,
            "month": 0,
        }
        total_qty, total_times = 0.0, 0.0
        recent_qty = [float(row["qty"])] * 7

        for d in range(1, horizon + 1):
            cur = row["date"] + pd.Timedelta(days=d)
            state["dow"] = cur.dayofweek
            state["month"] = cur.month
            X = pd.DataFrame([state])[feature_cols]
            q = max(0.0, float(model_qty.predict(X)[0]))
            t = max(0.0, float(model_times.predict(X)[0]))
            total_qty += q
            total_times += t
            recent_qty = (recent_qty + [q])[-7:]
            state["qty_lag1"] = q
            state["qty_lag7"] = recent_qty[0]
            state["qty_roll7"] = float(np.mean(recent_qty))
            state["times_lag1"] = t
            state["times_roll7"] = 0.7 * state["times_roll7"] + 0.3 * t

        forecasts.append({
            "item_code": row["item_code"],
            "item_name": row["item_name"],
            "department": row["destination_location"],
            "future_qty": round(total_qty, 1),
            "future_times": round(total_times, 1),
        })

    return pd.DataFrame(forecasts).sort_values("future_qty", ascending=False)

fc = run_forecast(feat, daily_f, horizon, feature_cols)

# ----- Display -----
st.subheader("Future predicted movements")
st.dataframe(fc.head(50), use_container_width=True, hide_index=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("By department")
    by_dept = (
        fc.groupby("department", as_index=False)
        .agg(future_qty=("future_qty", "sum"),
             future_times=("future_times", "sum"),
             n_products=("item_code", "nunique"))
        .sort_values("future_qty", ascending=False)
    )
    st.dataframe(by_dept, use_container_width=True, hide_index=True)

with col2:
    st.subheader("By product")
    by_item = (
        fc.groupby(["item_code", "item_name"], as_index=False)
        .agg(future_qty=("future_qty", "sum"),
             future_times=("future_times", "sum"),
             n_departments=("department", "nunique"))
        .sort_values("future_qty", ascending=False)
    )
    st.dataframe(by_item.head(30), use_container_width=True, hide_index=True)

st.download_button(
    "Download full future predictions (Excel)",
    data=fc.to_csv(index=False).encode("utf-8"),
    file_name="future_stock_movements.csv",
    mime="text/csv"
)
