import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="CVRP 3-Algorithm Comparison",
    page_icon="🚚",
    layout="wide",
)


st.markdown(
    """
    <style>
    .main { background-color: #f7f9fc; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .title-box {
        background: linear-gradient(135deg, #1e3a8a, #2563eb);
        padding: 26px 30px;
        border-radius: 22px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 10px 28px rgba(37, 99, 235, 0.2);
    }
    .title-box h1 { margin: 0; font-size: 34px; font-weight: 800; }
    .title-box p { margin-top: 10px; opacity: 0.92; font-size: 16px; }
    .metric-card {
        background: white;
        padding: 22px 24px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
    }
    .metric-label { color: #64748b; font-size: 14px; font-weight: 700; }
    .metric-value { color: #0f172a; font-size: 26px; font-weight: 800; margin-top: 8px; }
    .note-box {
        background: #fff7ed;
        color: #7c2d12;
        padding: 16px 20px;
        border: 1px solid #fed7aa;
        border-radius: 16px;
        margin-bottom: 20px;
        line-height: 1.6;
    }
    .section-title {
        font-size: 24px;
        font-weight: 800;
        color: #0f172a;
        margin-top: 16px;
        margin-bottom: 14px;
    }
    div[data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 12px;
        padding: 10px 18px;
        border: 1px solid #e5e7eb;
        font-weight: 700;
    }
    .stTabs [aria-selected="true"] { background-color: #2563eb !important; color: white !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


RESULTS_SUMMARY = Path("results/results_summary.csv")
RESULTS_AVG = Path("results/results_avg.csv")
PROJECT_WIDE = Path("results/project_wide.csv")
WINNING_RATE = Path("results/winning_rate.csv")

missing_files = [
    str(p) for p in [RESULTS_SUMMARY, RESULTS_AVG, PROJECT_WIDE, WINNING_RATE]
    if not p.exists()
]

if missing_files:
    st.error(
        "Thiếu file kết quả. Hãy chạy: python -m src.experiments.run_compare\n\n"
        f"Thiếu: {missing_files}"
    )
    st.stop()


df = pd.read_csv(RESULTS_SUMMARY)
avg_df = pd.read_csv(RESULTS_AVG)
wide_df = pd.read_csv(PROJECT_WIDE)
winning_df = pd.read_csv(WINNING_RATE)


st.markdown(
    """
    <div class="title-box">
        <h1>🚚 So sánh 3 thuật toán CVRP</h1>
        <p>Clarke-Wright, Paper RL-Attention và Proposed AGIH-2opt trên cùng bộ dữ liệu CVRP</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="note-box">
        <b>Hướng đúng của sản phẩm:</b>
        Clarke-Wright là thuật toán cũ. Paper RL-Attention là route sinh từ thuật toán bài báo sau khi chạy trên Colab.
        Proposed AGIH-2opt lấy route của Paper làm nghiệm khởi tạo và cải thiện bằng 2-opt/relocate.
        Dashboard này không dùng OR-Tools trong bảng so sánh chính.
    </div>
    """,
    unsafe_allow_html=True,
)


st.sidebar.title("⚙️ Bộ lọc")

method_order = [
    "Clarke-Wright",
    "Paper-RL-Attention",
    "Proposed-AGIH-2opt",
]

available_methods = [m for m in method_order if m in df["Method"].unique()]

selected_methods = st.sidebar.multiselect(
    "Chọn thuật toán hiển thị",
    available_methods,
    default=available_methods,
)

instances = sorted(df["Instance"].unique())
selected_instance = st.sidebar.selectbox("Chọn instance để xem route", instances)

filtered_df = df[df["Method"].isin(selected_methods)].copy()
filtered_avg = avg_df[avg_df["Method"].isin(selected_methods)].copy()


best_row = filtered_avg.sort_values("Distance").iloc[0]
fastest_row = filtered_avg.sort_values("Runtime").iloc[0]
proposed_avg = filtered_avg[filtered_avg["Method"] == "Proposed-AGIH-2opt"]

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Số instance test</div>
            <div class="metric-value">{df["Instance"].nunique()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Distance tốt nhất</div>
            <div class="metric-value">{best_row["Method"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Runtime nhanh nhất</div>
            <div class="metric-value">{fastest_row["Method"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    if not proposed_avg.empty:
        value = proposed_avg.iloc[0]["Distance"]
    else:
        value = 0.0

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Proposed avg distance</div>
            <div class="metric-value">{value:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Bảng wide giống ACO/GA",
    "📈 Trung bình 3 thuật toán",
    "🏆 Winning rate",
    "🗺️ Bản đồ lộ trình",
])


with tab1:
    st.markdown(
        '<div class="section-title">Bảng kết quả dạng wide</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        wide_df,
        use_container_width=True,
        hide_index=True,
    )


with tab2:
    st.markdown(
        '<div class="section-title">Kết quả trung bình 3 thuật toán</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        filtered_avg.style.format({
            "Distance": "{:.4f}",
            "Vehicles": "{:.2f}",
            "Runtime": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        '<div class="section-title">Chi tiết từng instance</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        filtered_df[[
            "Instance",
            "Method",
            "Type",
            "Distance",
            "Vehicles",
            "Runtime",
        ]].style.format({
            "Distance": "{:.4f}",
            "Runtime": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )


with tab3:
    st.markdown(
        '<div class="section-title">Winning rate giữa 3 thuật toán</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        winning_df.style.format({
            "A_better_than_B_%": "{:.2f}",
            "Tie_%": "{:.2f}",
            "Avg_Improvement_%": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )


with tab4:
    st.markdown(
        '<div class="section-title">Trực quan hóa lộ trình xe</div>',
        unsafe_allow_html=True,
    )

    method_options = [
        m for m in selected_methods
        if m in df[df["Instance"] == selected_instance]["Method"].unique()
    ]

    selected_method = st.selectbox("Chọn thuật toán", method_options)

    row = df[
        (df["Instance"] == selected_instance)
        & (df["Method"] == selected_method)
    ].iloc[0]

    coords = json.loads(row["Coords"])
    demands = json.loads(row["Demands"])
    routes = json.loads(row["Routes"])

    i1, i2, i3, i4 = st.columns(4)

    with i1:
        st.metric("Instance", selected_instance)

    with i2:
        st.metric("Method", selected_method)

    with i3:
        st.metric("Distance", f"{row['Distance']:.4f}")

    with i4:
        st.metric("Vehicles", int(row["Vehicles"]))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[c[0] for c in coords[1:]],
        y=[c[1] for c in coords[1:]],
        mode="markers+text",
        text=[str(i) for i in range(1, len(coords))],
        textposition="top center",
        marker=dict(size=10),
        name="Customers",
    ))

    fig.add_trace(go.Scatter(
        x=[coords[0][0]],
        y=[coords[0][1]],
        mode="markers+text",
        text=["Depot 0"],
        textposition="top center",
        marker=dict(size=18, symbol="square"),
        name="Depot",
    ))

    for idx, route in enumerate(routes, start=1):
        fig.add_trace(go.Scatter(
            x=[coords[node][0] for node in route],
            y=[coords[node][1] for node in route],
            mode="lines+markers",
            name=f"Route {idx}",
            line=dict(width=3),
        ))

    fig.update_layout(
        title=f"{selected_method} - {selected_instance}",
        height=680,
        xaxis_title="X coordinate",
        yaxis_title="Y coordinate",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#e5e7eb")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#e5e7eb")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="section-title">Chi tiết route</div>',
        unsafe_allow_html=True,
    )

    for idx, route in enumerate(routes, start=1):
        load = sum(demands[node] for node in route if node != 0)
        route_str = " → ".join(map(str, route))
        st.write(f"**Route {idx}:** {route_str} | Load = {load}")