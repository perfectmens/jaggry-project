import streamlit as st
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# 🧠 PostgreSQL Connection Setup
# ==============================
@st.cache_resource
def init_connection():
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="postgres",   # change if your DB name differs
            user="postgres",       # your username
            password="your_password",  # your password
            port="5432"
        )
        return conn
    except Exception as e:
        st.error(f"Database connection failed ❌: {e}")
        return None


# ==============================
# 📊 Load data from PostgreSQL
# ==============================
def load_batches(conn, start_batch=1, end_batch=20):
    query = f"""
    SELECT batch_id, start_time, end_time, next_batch_start
    FROM batch_data
    WHERE batch_id BETWEEN {start_batch} AND {end_batch}
      AND next_batch_start IS NOT NULL
    ORDER BY batch_id;
    """
    df = pd.read_sql(query, conn)
    return df


# ==========================================
# ⏱️ Calculate delays and classify categories
# ==========================================
def classify_batches(df):
    df["delay_in_seconds"] = (df["next_batch_start"] - df["end_time"]).dt.total_seconds()

    def label_delay(x):
        if x <= 10:
            return "Perfect"
        elif 10 < x <= 60:
            return "Normal"
        elif 60 < x <= 900:
            return "Cleaning Delay"
        else:
            return "Failure"

    df["status"] = df["delay_in_seconds"].apply(label_delay)
    return df


# =======================================
# 📈 Visualizations based on user queries
# =======================================
def show_visualizations(df):
    st.subheader("📊 Batch Delay Distribution")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x="batch_id", y="delay_in_seconds", hue="status", data=df, ax=ax)
    ax.set_title("Delay between Batches (seconds)")
    ax.set_xlabel("Batch ID")
    ax.set_ylabel("Delay (s)")
    st.pyplot(fig)

    st.subheader("🧮 Average Delay per Category")
    avg_delay = df.groupby("status")["delay_in_seconds"].mean().reset_index()

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.barplot(x="status", y="delay_in_seconds", data=avg_delay, ax=ax2)
    for i, row in avg_delay.iterrows():
        ax2.text(i, row["delay_in_seconds"] + 5, f"{row['delay_in_seconds']:.1f}s", ha='center')
    ax2.set_ylabel("Average Delay (s)")
    st.pyplot(fig2)

    st.subheader("🧭 Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(5, 3))
    sns.heatmap(df[["batch_id", "delay_in_seconds"]].corr(), annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)


# ======================
# 🚀 Streamlit Interface
# ======================
st.title("🏭 Jaggery Batch Assistant")
st.caption("Analyze batch timings, visualize delays, and identify performance gaps.")

conn = init_connection()
if conn:
    st.sidebar.header("Query Options")

    start_batch = st.sidebar.number_input("Start Batch ID", min_value=1, value=1)
    end_batch = st.sidebar.number_input("End Batch ID", min_value=start_batch, value=20)

    if st.sidebar.button("🔍 Analyze"):
        with st.spinner("Fetching data from PostgreSQL..."):
            df = load_batches(conn, start_batch, end_batch)
            if df.empty:
                st.warning("No data found for the given batch range.")
            else:
                df = classify_batches(df)
                st.success("✅ Data successfully analyzed!")

                st.dataframe(df)

                show_visualizations(df)

                # Summary
                st.subheader("📘 Summary")
                summary = df["status"].value_counts().reset_index()
                summary.columns = ["Status", "Count"]
                st.table(summary)

else:
    st.stop()

