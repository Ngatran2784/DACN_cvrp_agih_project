import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


st.set_page_config(
    page_title="CVRP Algorithm Dashboard",
    page_icon="🚚",
    layout="wide",
)


# =========================
# CSS TRANG TRÍ
# =========================
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .title-box {
        background: linear-gradient(135deg, #1e3a8a, #2563eb);
        padding: 28px 32px;
        border-radius: 22px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 28px rgba(37, 99, 235, 0.22);
    }

    .title-box h1 {
        margin: 0;
        font-size: 34px;
        font-weight: 800;
    }

    .title-box p {
        margin-top: 10px;
        font-size: 16px;
        opacity: 0.92;
    }

    .metric-card {
        background: white;
        padding: 22px 24px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
    }

    .metric-label {
        color: #64748b;
        font-size: 14px;
        font-weight: 600;
    }

    .metric-value {
        color: #0f172a;
        font-size: 28px;
        font-weight: 800;
        margin-top: 8px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 800;
        color: #0f172a;
        margin-top: 12px;
        margin-bottom: 14px;
    }

    .route-box {
        background: white;
        padding: 18px 22px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        margin-bottom: 12px;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.05);
    }

    .route-title {
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 6px;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 12px;
        padding: 10px 18px;
        border: 1px solid #e5e7eb;
        font-weight: 700;
    }

    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# LOAD DATA
# =========================
summary_path = Path("results/results_summary.csv")
avg_path = Path("results/results_avg.csv")

if not summary_path.exists() or not avg_path.exists():
    st.error("Chưa có file kết quả. Hãy chạy: python -m src.experiments.run_compare")
    st.stop()

df = pd.read_csv(summary_path)
avg_df = pd.read_csv(avg_path)
win_path = Path("results/winning_rate.csv")
winning_df = pd.read_csv(win_path) if win_path.exists() else pd.DataFrame()

required_cols = [
    "Instance", "Method", "Distance", "Vehicles", "Runtime",
    "Gap_vs_OR_Tools_%", "Coords", "Demands", "Routes"
]

missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    st.error(f"File results_summary.csv thiếu cột: {missing_cols}")
    st.stop()


# =========================
# HEADER
# =========================
st.markdown(
    """
    <div class="title-box">
        <h1>🚚 CVRP Algorithm Comparison Dashboard</h1>
        <p>So sánh thuật toán cũ, thuật toán bài báo và phương pháp đề xuất cho bài toán CVRP</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Bộ lọc")

all_methods = sorted(df["Method"].unique())
selected_methods = st.sidebar.multiselect(
    "Chọn thuật toán",
    all_methods,
    default=all_methods,
)

all_instances = sorted(df["Instance"].unique())
selected_instance = st.sidebar.selectbox(
    "Chọn instance để xem route",
    all_instances,
)

filtered_df = df[df["Method"].isin(selected_methods)].copy()
filtered_avg = avg_df[avg_df["Method"].isin(selected_methods)].copy()

st.sidebar.markdown("---")
st.sidebar.info(
    "Dashboard đọc dữ liệu từ `results/results_summary.csv` và `results/results_avg.csv`."
)


# =========================
# METRIC CARDS
# =========================
best_row = filtered_avg.sort_values("Distance").iloc[0]
fastest_row = filtered_avg.sort_values("Runtime").iloc[0]
min_gap_row = filtered_avg.sort_values("Gap_vs_OR_Tools_%").iloc[0]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Số instance test</div>
            <div class="metric-value">{df['Instance'].nunique()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Distance tốt nhất</div>
            <div class="metric-value">{best_row['Method']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Runtime nhanh nhất</div>
            <div class="metric-value">{fastest_row['Method']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Gap thấp nhất</div>
            <div class="metric-value">{min_gap_row['Gap_vs_OR_Tools_%']:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Bảng so sánh",
    "📈 Biểu đồ",
    "🗺️ Bản đồ lộ trình",
    "🏆 Winning rate",
])


# =========================
# TAB 1: BẢNG
# =========================
with tab1:
    st.markdown('<div class="section-title">Kết quả trung bình theo thuật toán</div>', unsafe_allow_html=True)

    show_avg = filtered_avg.copy()
    show_avg = show_avg.rename(columns={
        "Distance": "Avg Distance",
        "Vehicles": "Avg Vehicles",
        "Runtime": "Avg Runtime",
        "Gap_vs_OR_Tools_%": "Avg Gap vs OR-Tools (%)",
    })

    st.dataframe(
        show_avg.style.format({
            "Avg Distance": "{:.4f}",
            "Avg Vehicles": "{:.2f}",
            "Avg Runtime": "{:.4f}",
            "Avg Gap vs OR-Tools (%)": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown('<div class="section-title">Kết quả chi tiết từng instance</div>', unsafe_allow_html=True)

    detail_cols = [
        "Instance", "Method", "Distance", "Vehicles",
        "Runtime", "Gap_vs_OR_Tools_%"
    ]

    st.dataframe(
        filtered_df[detail_cols].style.format({
            "Distance": "{:.4f}",
            "Runtime": "{:.4f}",
            "Gap_vs_OR_Tools_%": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )


# =========================
# TAB 2: BIỂU ĐỒ
# =========================
with tab2:
    st.markdown('<div class="section-title">Biểu đồ so sánh trung bình</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        fig_distance = px.bar(
            filtered_avg,
            x="Method",
            y="Distance",
            text="Distance",
            title="Average Distance theo thuật toán",
        )
        fig_distance.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_distance.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig_distance, use_container_width=True)

    with c2:
        fig_runtime = px.bar(
            filtered_avg,
            x="Method",
            y="Runtime",
            text="Runtime",
            title="Average Runtime theo thuật toán",
        )
        fig_runtime.update_traces(texttemplate="%{text:.4f}s", textposition="outside")
        fig_runtime.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig_runtime, use_container_width=True)

    st.markdown('<div class="section-title">Gap vs OR-Tools</div>', unsafe_allow_html=True)

    fig_gap = px.bar(
        filtered_avg,
        x="Method",
        y="Gap_vs_OR_Tools_%",
        text="Gap_vs_OR_Tools_%",
        title="Average Gap so với OR-Tools (%)",
    )
    fig_gap.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig_gap.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig_gap, use_container_width=True)


# =========================
# TAB 3: ROUTE MAP
# =========================
with tab3:
    st.markdown('<div class="section-title">Trực quan hóa lộ trình xe</div>', unsafe_allow_html=True)

    method_options = df[df["Instance"] == selected_instance]["Method"].unique()
    selected_method = st.selectbox("Chọn thuật toán", method_options)

    row = df[
        (df["Instance"] == selected_instance) &
        (df["Method"] == selected_method)
    ].iloc[0]

    coords = json.loads(row["Coords"])
    demands = json.loads(row["Demands"])
    routes = json.loads(row["Routes"])

    info1, info2, info3, info4 = st.columns(4)

    with info1:
        st.metric("Instance", selected_instance)

    with info2:
        st.metric("Distance", f"{row['Distance']:.4f}")

    with info3:
        st.metric("Vehicles", int(row["Vehicles"]))

    with info4:
        st.metric("Runtime", f"{row['Runtime']:.4f}s")

    fig = go.Figure()

    # Customers
    fig.add_trace(go.Scatter(
        x=[c[0] for c in coords[1:]],
        y=[c[1] for c in coords[1:]],
        mode="markers+text",
        text=[str(i) for i in range(1, len(coords))],
        textposition="top center",
        marker=dict(size=10),
        name="Customers",
    ))

    # Depot
    fig.add_trace(go.Scatter(
        x=[coords[0][0]],
        y=[coords[0][1]],
        mode="markers+text",
        text=["Depot 0"],
        textposition="top center",
        marker=dict(size=18, symbol="square"),
        name="Depot",
    ))

    # Routes
    for idx, route in enumerate(routes, start=1):
        route_x = [coords[node][0] for node in route]
        route_y = [coords[node][1] for node in route]

        fig.add_trace(go.Scatter(
            x=route_x,
            y=route_y,
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
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#e5e7eb")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#e5e7eb")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Chi tiết route</div>', unsafe_allow_html=True)

    for idx, route in enumerate(routes, start=1):
        load = sum(demands[node] for node in route if node != 0)
        route_str = " → ".join(map(str, route))

        st.markdown(
            f"""
            <div class="route-box">
                <div class="route-title">Route {idx}</div>
                <div><b>Lộ trình:</b> {route_str}</div>
                <div><b>Tải trọng:</b> {load}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab4:
    st.markdown(
        '<div class="section-title">Winning rate giữa các thuật toán</div>',
        unsafe_allow_html=True,
    )

    if winning_df.empty:
        st.warning("Chưa có file winning_rate.csv. Hãy chạy lại: python -m src.experiments.run_compare")
    else:
        st.dataframe(
            winning_df.style.format({
                "A_better_than_B_%": "{:.2f}",
                "Tie_%": "{:.2f}",
                "Avg_Improvement_%": "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        proposed_rows = winning_df[
            winning_df["Method_A"] == "Proposed-AGIH-2opt"
        ]

        if not proposed_rows.empty:
            st.markdown(
                '<div class="section-title">Tỷ lệ Proposed-AGIH-2opt thắng các thuật toán khác</div>',
                unsafe_allow_html=True,
            )

            fig_win = px.bar(
                proposed_rows,
                x="Method_B",
                y="A_better_than_B_%",
                text="A_better_than_B_%",
                title="Proposed-AGIH-2opt better than other methods (%)",
            )

            fig_win.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside",
            )

            fig_win.update_layout(
                height=420,
                margin=dict(l=20, r=20, t=60, b=20),
                yaxis_title="Winning rate (%)",
                xaxis_title="Compared method",
            )

            st.plotly_chart(fig_win, use_container_width=True)