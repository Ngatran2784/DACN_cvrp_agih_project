import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="CVRP 3-Algorithm Comparison", page_icon="🚚", layout="wide")

st.markdown(
    """
    <style>
    .main { background-color: #f6f8fc; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .title-box {
        background: linear-gradient(135deg, #1e3a8a, #2563eb);
        padding: 28px 32px;
        border-radius: 22px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 10px 28px rgba(37, 99, 235, 0.22);
    }
    .title-box h1 { margin: 0; font-size: 34px; font-weight: 850; }
    .title-box p { margin-top: 10px; font-size: 16px; opacity: 0.94; }
    .note-box {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #9a3412;
        padding: 16px 20px;
        border-radius: 16px;
        margin-bottom: 20px;
        line-height: 1.6;
    }
    .metric-card {
        background: white;
        padding: 22px 24px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
    }
    .metric-label { color: #64748b; font-size: 14px; font-weight: 700; }
    .metric-value { color: #0f172a; font-size: 26px; font-weight: 850; margin-top: 8px; }
    .section-title { font-size: 24px; font-weight: 850; color: #0f172a; margin-top: 20px; margin-bottom: 16px; }
    .route-box {
        background: white;
        padding: 16px 20px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        margin-bottom: 10px;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.05);
    }
    .route-title { font-weight: 850; color: #1e3a8a; margin-bottom: 6px; }
    div[data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

paths = {
    "summary": Path("results/results_summary.csv"),
    "avg": Path("results/results_avg.csv"),
    "wide": Path("results/project_wide.csv"),
    "win": Path("results/winning_rate.csv"),
    "paper": Path("results/paper_reported_results.csv"),
    "three": Path("results/three_algorithm_summary.csv"),
}

missing = [str(p) for p in paths.values() if not p.exists()]
if missing:
    st.error("Thiếu file kết quả. Hãy chạy: python -m src.experiments.run_compare")
    st.write("Các file đang thiếu:", missing)
    st.stop()

df = pd.read_csv(paths["summary"])
avg_df = pd.read_csv(paths["avg"])
wide_df = pd.read_csv(paths["wide"])
winning_df = pd.read_csv(paths["win"])
paper_df = pd.read_csv(paths["paper"])
three_df = pd.read_csv(paths["three"])

st.markdown(
    """
    <div class="title-box">
        <h1>🚚 CVRP 3-Algorithm Comparison Dashboard</h1>
        <p>So sánh đúng 3 nhóm: thuật toán cũ, thuật toán bài báo và thuật toán đề xuất</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="note-box">
        <b>Lưu ý:</b> Dashboard chỉ hiển thị 3 thuật toán chính. Paper RL-BS(10) dùng số liệu công bố trong bài báo gốc. 
        Clarke-Wright và Proposed-AGIH-2opt là hai thuật toán chạy trực tiếp trong project nên có bảng instance và bản đồ route.
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar: no OR-Tools shown
st.sidebar.title("⚙️ Bộ lọc")
display_methods = ["Clarke-Wright", "Proposed-AGIH-2opt"]
selected_methods = st.sidebar.multiselect(
    "Chọn thuật toán để xem route/project",
    display_methods,
    default=display_methods,
)
instance_list = sorted(df["INSTANCE"].unique())
selected_instance = st.sidebar.selectbox("Chọn instance để xem route", instance_list)

route_df = df[df["METHOD"].isin(selected_methods)].copy()
chart_avg = avg_df[avg_df["METHOD"].isin(display_methods)].copy()

# Metrics
proposed_avg = avg_df[avg_df["METHOD"] == "Proposed-AGIH-2opt"].iloc[0]
best_project = chart_avg.sort_values("DISTANCE").iloc[0]
fastest_project = chart_avg.sort_values("RUNTIME_S").iloc[0]
prop_win_cw_row = winning_df[(winning_df["METHOD_A"] == "Proposed-AGIH-2opt") & (winning_df["METHOD_B"] == "Clarke-Wright")]
prop_win_cw = float(prop_win_cw_row.iloc[0]["A_BETTER_THAN_B_%"]) if not prop_win_cw_row.empty else 0.0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"<div class='metric-card'><div class='metric-label'>Số instance test</div><div class='metric-value'>{df['INSTANCE'].nunique()}</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><div class='metric-label'>Distance tốt nhất trong project</div><div class='metric-value'>{best_project['METHOD']}</div></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><div class='metric-label'>Runtime nhanh nhất trong project</div><div class='metric-value'>{fastest_project['METHOD']}</div></div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='metric-card'><div class='metric-label'>Proposed thắng Clarke-Wright</div><div class='metric-value'>{prop_win_cw:.2f}%</div></div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Bảng số liệu 3 thuật toán",
    "📊 Bảng wide project",
    "📄 Số liệu bài báo",
    "🏆 Winning rate",
    "🗺️ Bản đồ lộ trình",
])

with tab1:
    st.markdown('<div class="section-title">Bảng so sánh 3 thuật toán chính</div>', unsafe_allow_html=True)
    st.dataframe(
        three_df.style.format({
            "AVG_DISTANCE_CVRP20": "{:.4f}",
            "AVG_VEHICLES_CVRP20": "{:.2f}",
            "AVG_RUNTIME_S_CVRP20": "{:.4f}",
            "PROJECT_WIN_VS_CLARKE_WRIGHT_%": "{:.2f}",
            "PAPER_WIN_VS_OLD_VRP50_%": "{:.2f}",
            "PAPER_WIN_VS_OLD_VRP100_%": "{:.2f}",
        }, na_rep="—"),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Paper RL-BS(10) chỉ có số liệu report từ bài báo, không có route từng instance. Vì vậy bảng không gán distance project cho paper.")

    st.markdown('<div class="section-title">Biểu đồ project: Average Distance</div>', unsafe_allow_html=True)
    fig = px.bar(
        chart_avg,
        x="METHOD",
        y="DISTANCE",
        color="TYPE",
        text="DISTANCE",
        title="Average Distance - project experiment",
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=60, b=20), xaxis_title="Algorithm", yaxis_title="Average distance")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown('<div class="section-title">Bảng kết quả dạng wide theo từng instance</div>', unsafe_allow_html=True)
    st.dataframe(wide_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Kết quả trung bình trong project</div>', unsafe_allow_html=True)
    st.dataframe(
        chart_avg.style.format({
            "DISTANCE": "{:.4f}",
            "VEHICLES": "{:.2f}",
            "RUNTIME_S": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

with tab3:
    st.markdown('<div class="section-title">Số liệu Paper RL-BS(10) từ bài báo</div>', unsafe_allow_html=True)
    st.dataframe(
        paper_df.style.format({"WINNING_RATE_%": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.info("Các số liệu này lấy từ paper Nazari et al. (NeurIPS 2018). Không trộn với route chi tiết do project tự chạy.")

with tab4:
    st.markdown('<div class="section-title">Winning rate trong project</div>', unsafe_allow_html=True)
    st.dataframe(
        winning_df.style.format({
            "A_BETTER_THAN_B_%": "{:.2f}",
            "TIE_%": "{:.2f}",
            "AVG_IMPROVEMENT_%": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    prop_rows = winning_df[winning_df["METHOD_A"] == "Proposed-AGIH-2opt"]
    if not prop_rows.empty:
        fig_win = px.bar(
            prop_rows,
            x="METHOD_B",
            y="A_BETTER_THAN_B_%",
            text="A_BETTER_THAN_B_%",
            title="Proposed-AGIH-2opt better than other project algorithms (%)",
        )
        fig_win.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig_win.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20), yaxis_title="Winning rate (%)", xaxis_title="Compared method")
        st.plotly_chart(fig_win, use_container_width=True)

with tab5:
    st.markdown('<div class="section-title">Bản đồ lộ trình</div>', unsafe_allow_html=True)

    method_options = [m for m in selected_methods if m in df[df["INSTANCE"] == selected_instance]["METHOD"].unique()]
    if not method_options:
        st.warning("Không có method để hiển thị route.")
        st.stop()

    selected_method = st.selectbox("Chọn thuật toán", method_options)
    row = df[(df["INSTANCE"] == selected_instance) & (df["METHOD"] == selected_method)].iloc[0]

    coords = json.loads(row["COORDS"])
    demands = json.loads(row["DEMANDS"])
    routes = json.loads(row["ROUTES"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Instance", selected_instance)
    m2.metric("Distance", f"{row['DISTANCE']:.4f}")
    m3.metric("Vehicles", int(row["VEHICLES"]))
    m4.metric("Runtime", f"{row['RUNTIME_S']:.4f}s")

    fig_map = go.Figure()
    fig_map.add_trace(go.Scatter(
        x=[c[0] for c in coords[1:]],
        y=[c[1] for c in coords[1:]],
        mode="markers+text",
        text=[str(i) for i in range(1, len(coords))],
        textposition="top center",
        marker=dict(size=10),
        name="Customers",
    ))
    fig_map.add_trace(go.Scatter(
        x=[coords[0][0]],
        y=[coords[0][1]],
        mode="markers+text",
        text=["Depot 0"],
        textposition="top center",
        marker=dict(size=18, symbol="square"),
        name="Depot",
    ))
    for idx, route in enumerate(routes, start=1):
        fig_map.add_trace(go.Scatter(
            x=[coords[node][0] for node in route],
            y=[coords[node][1] for node in route],
            mode="lines+markers",
            name=f"Route {idx}",
            line=dict(width=3),
        ))
    fig_map.update_layout(
        title=f"{selected_method} - {selected_instance}",
        height=680,
        xaxis_title="X coordinate",
        yaxis_title="Y coordinate",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig_map.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#e5e7eb")
    fig_map.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#e5e7eb")
    st.plotly_chart(fig_map, use_container_width=True)

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
