import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
import io

# ----------------------------
# Original function (unchanged)
# ----------------------------
def clean_and_analyze_sales_data(input_file: str, output_file: str = "Tclean.csv"):
    na_tokens = ["", " ", "  ", "n/a", "na", "none", "null", "nan", "-", "--", "?", "missing", "unknown"]
    df_raw = pd.read_csv(input_file, dtype=str, keep_default_na=False)

    def clean_str_cell(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        if s.lower() in na_tokens:
            return np.nan
        return s

    df = df_raw.applymap(clean_str_cell)

    def normalize_col(col: str) -> str:
        col = col.strip().lower()
        col = re.sub(r"[\s_/]+", " ", col)
        col = col.replace("$", "").replace("#", "")
        return col

    df.columns = [normalize_col(c) for c in df.columns]

    col_variants = {
        "item": ["item", "product", "menu item", "item name"],
        "quantity": ["quantity", "qty", "count", "no of items", "number of items"],
        "price per unit": ["price per unit", "unit price", "price", "per unit price", "unit cost"],
        "total spent": ["total spent", "total", "amount", "line total", "revenue", "sale amount"],
        "payment method": ["payment method", "payment", "pay method", "payment type", "method"],
    }

    canonical_map = {}
    for canon, variants in col_variants.items():
        for v in variants:
            if v in df.columns:
                canonical_map[v] = canon
                break

    renamed_cols = {src: dst for src, dst in canonical_map.items()}
    df = df.rename(columns=renamed_cols)

    for col in ["item", "quantity", "price per unit", "total spent", "payment method"]:
        if col not in df.columns:
            df[col] = np.nan

    def to_numeric_series(s: pd.Series) -> pd.Series:
        if s.dtype == object:
            cleaned = s.astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
            cleaned = cleaned.str.replace(r"(?<=\d)\.(?=.*\.)", "", regex=True)
        else:
            cleaned = s
        return pd.to_numeric(cleaned, errors="coerce")

    df["quantity"] = to_numeric_series(df["quantity"])
    df["price per unit"] = to_numeric_series(df["price per unit"])
    df["total spent"] = to_numeric_series(df["total spent"])

    calc_total = df["quantity"] * df["price per unit"]
    tolerance = 0.01
    needs_fix = df["total spent"].isna() | (np.abs(df["total spent"] - calc_total) > tolerance)
    df.loc[needs_fix, "total spent"] = calc_total.loc[needs_fix]

    for col in df.columns:
        if col not in ["quantity", "price per unit", "total spent"]:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            if col in ["item", "payment method"]:
                df[col] = df[col].apply(lambda x: x.title() if isinstance(x, str) else x)

    Tclean = df[df["total spent"].notna()].copy()
    Tclean.to_csv(output_file, index=False)

    summary_stats = Tclean["total spent"].agg(
        Count="count",
        Mean="mean",
        Std="std",
        Min="min",
        Median="median",
        Max="max",
        Sum="sum",
    ).to_frame().T.round(2)

    item_counts = Tclean.dropna(subset=["item"]).groupby("item").size().sort_values(ascending=False)
    qty_per_item = Tclean.groupby("item")["quantity"].sum().sort_values(ascending=False)

    most_frequent_item = item_counts.index[0] if len(item_counts) else None
    greatest_quantity_item = qty_per_item.index[0] if len(qty_per_item) else None

    pay_counts = Tclean.dropna(subset=["payment method"]).groupby("payment method").size().sort_values(ascending=False)
    most_preferred_payment = pay_counts.index[0] if len(pay_counts) else None

    st.subheader("✅ Mostly Sold Item")
    st.write(f"By # of transactions: **{most_frequent_item}**")
    st.write(f"By total quantity: **{greatest_quantity_item}**")

    st.subheader("💳 Most Preferred Payment Method")
    st.write(f"**{most_preferred_payment}**")

    st.success(f"Cleaned table saved to: `{output_file}`")

    st.subheader("📈 Visualizations")

    total_spent_per_item = Tclean.groupby("item")["total spent"].sum().sort_values(ascending=False)
    fig1, ax1 = plt.subplots()
    total_spent_per_item.plot(kind="bar", ax=ax1)
    ax1.set_title("Total Spent (Revenue) per Item")
    ax1.set_xlabel("Item")
    ax1.set_ylabel("Total Spent")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    item_counts.plot(kind="bar", ax=ax2)
    ax2.set_title("Number of Transactions per Item")
    ax2.set_xlabel("Item")
    ax2.set_ylabel("Transactions")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig2)

    if len(pay_counts) > 0:
        fig3, ax3 = plt.subplots()
        ax3.pie(pay_counts.values, labels=pay_counts.index, autopct="%1.1f%%", startangle=90)
        ax3.set_title("Payment Methods (Share of Transactions)")
        ax3.axis("equal")
        st.pyplot(fig3)

    fig4, ax4 = plt.subplots()
    Tclean["total spent"].plot(kind="hist", bins=20, ax=ax4)
    ax4.set_title("Histogram of Total Spent per Transaction")
    ax4.set_xlabel("Total Spent")
    ax4.set_ylabel("Frequency")
    st.pyplot(fig4)

    return Tclean, summary_stats

# ----------------------------
# Streamlit wrapper
# ----------------------------
st.title("☕ Cafe Sales Data Cleaning & Analysis")

github_url = st.text_input(
    "Enter the raw GitHub CSV URL:",
    "https://raw.githubusercontent.com/USERNAME/REPO/main/dirty_cafe_sales-1.csv"
)

if st.button("Run Analysis"):
    try:
        st.info("Loading data from GitHub...")
        df = pd.read_csv(github_url, dtype=str, keep_default_na=False)
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())

        with st.spinner("Cleaning and analyzing..."):
            cleaned, summary = clean_and_analyze_sales_data(github_url)

        st.subheader("Summary Statistics")
        st.dataframe(summary)

        st.download_button(
            "💾 Download Cleaned Data as CSV",
            data=open("Tclean.csv", "rb"),
            file_name="Tclean.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info("Enter your GitHub CSV URL above and click **Run Analysis** to begin.")
