import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="CVRP Algorithm Comparison",
    page_icon="🚚",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .hero {
        background: linear-gradient(135deg, #1e3a8a, #2563eb);
        padding: 28px 34px;
        border-radius: 22px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 10px 28px rgba(37, 99, 235, 0.22);
    }
    .hero h1 { margin: 0; font-size: 34px; font-weight: 850; }
    .hero p { margin-top: 10px; font-size: 16px; opacity: 0.92; }
    .note-box {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #7c2d12;
        padding: 14px 18px;
        border-radius: 14px;
        margin-bottom: 16px;
    }
    .metric-card {
        background: white;
        padding: 20px 22px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
    }
    .metric-label { color: #64748b; font-size: 14px; font-weight: 700; }
    .metric-value { color: #0f172a; font-size: 25px; font-weight: 850; margin-top: 8px; }
    .section-title { font-size: 23px; font-weight: 850; color: #0f172a; margin-top: 12px; margin-bottom: 14px; }
    .route-box {
        background: white;
        padding: 16px 20px;
        border-radius: 15px;
        border: 1px solid #e5e7eb;
        margin-bottom: 10px;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.05);
    }
    .route-title { font-weight: 850; color: #1e3a8a; margin-bottom: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


summary_path = Path("results/results_summary.csv")
avg_path = Path("results/results_avg.csv")
win_path = Path("results/winning_rate.csv")
paper_path = Path("results/paper_reported_results.csv")
three_path = Path("results/three_method_comparison.csv")

missing = [str(p) for p in [summary_path, avg_path, win_path, paper_path, three_path] if not p.exists()]
if missing:
    st.error("Thiếu file kết quả. Hãy chạy: python -m src.experiments.run_compare")
    st.write("File đang thiếu:", missing)
    st.stop()


df = pd.read_csv(summary_path)
avg_df = pd.read_csv(avg_path)
winning_df = pd.read_csv(win_path)
paper_df = pd.read_csv(paper_path)
three_df = pd.read_csv(three_path)

st.markdown(
    """
    <div class="hero">
        <h1>🚚 CVRP Algorithm Comparison Dashboard</h1>
        <p>So sánh 3 nhóm: thuật toán cũ, thuật toán bài báo và thuật toán đề xuất cho CVRP</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="note-box">
        <b>Lưu ý học thuật:</b> Paper RL-BS(10) được trình bày bằng số liệu công bố trong bài báo gốc.
        Clarke-Wright, OR-Tools Reference và Proposed AGIH-2opt là các thuật toán chạy trực tiếp trong project.
        Không trộn route chi tiết tự chạy với số liệu paper khi paper không công bố route từng instance.
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("⚙️ Bộ lọc")
run_methods = sorted(df["Method"].unique())
selected_methods = st.sidebar.multiselect("Chọn thuật toán chạy trong project", run_methods, default=run_methods)
instance_list = sorted(df["Instance"].unique())
selected_instance = st.sidebar.selectbox("Chọn instance để xem route", instance_list)
filtered_df = df[df["Method"].isin(selected_methods)].copy()
filtered_avg = avg_df[avg_df["Method"].isin(selected_methods)].copy()

best_project = filtered_avg.sort_values("Distance").iloc[0]
fastest_project = filtered_avg.sort_values("Runtime").iloc[0]
prop_row = avg_df[avg_df["Method"] == "Proposed-AGIH-2opt"]
prop_distance = float(prop_row.iloc[0]["Distance"]) if not prop_row.empty else None

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Số instance test</div><div class="metric-value">{df['Instance'].nunique()}</div></div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Project distance tốt nhất</div><div class="metric-value">{best_project['Method']}</div></div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Project runtime nhanh nhất</div><div class="metric-value">{fastest_project['Method']}</div></div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Proposed Avg Distance</div><div class="metric-value">{prop_distance:.4f}</div></div>""", unsafe_allow_html=True)


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "✅ So sánh 3 thuật toán",
    "📊 Thực nghiệm project",
    "📄 Số liệu bài báo",
    "🏆 Winning rate",
    "🗺️ Bản đồ lộ trình",
])

with tab1:
    st.markdown('<div class="section-title">Bảng so sánh 3 nhóm thuật toán</div>', unsafe_allow_html=True)
    st.dataframe(three_df, use_container_width=True, hide_index=True)

    st.markdown(
        """
        **Cách đọc bảng:**
        - `Clarke-Wright Savings`: thuật toán cũ, chạy trực tiếp trong project.
        - `Paper RL-BS(10)`: thuật toán bài báo, lấy đúng số liệu công bố trong paper.
        - `Proposed AGIH-2opt`: thuật toán đề xuất, chạy trực tiếp trong project và có route visualization.
        """
    )

with tab2:
    st.markdown('<div class="section-title">Kết quả thực nghiệm chạy trực tiếp trong project</div>', unsafe_allow_html=True)
    st.dataframe(
        filtered_avg.style.format({
            "Distance": "{:.4f}",
            "Vehicles": "{:.2f}",
            "Runtime": "{:.4f}",
            "Gap_vs_OR_Tools_%": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        fig_distance = px.bar(
            filtered_avg,
            x="Method",
            y="Distance",
            color="Type",
            text="Distance",
            title="Average Distance - project experiment",
        )
        fig_distance.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_distance.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig_distance, use_container_width=True)

    with c2:
        fig_runtime = px.bar(
            filtered_avg,
            x="Method",
            y="Runtime",
            color="Type",
            text="Runtime",
            title="Average Runtime - project experiment",
        )
        fig_runtime.update_traces(texttemplate="%{text:.4f}s", textposition="outside")
        fig_runtime.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig_runtime, use_container_width=True)

    st.markdown('<div class="section-title">Chi tiết từng instance</div>', unsafe_allow_html=True)
    st.dataframe(
        filtered_df[["Instance", "Method", "Type", "Distance", "Vehicles", "Runtime", "Gap_vs_OR_Tools_%"]].style.format({
            "Distance": "{:.4f}",
            "Runtime": "{:.4f}",
            "Gap_vs_OR_Tools_%": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

with tab3:
    st.markdown('<div class="section-title">Số liệu thuật toán bài báo theo paper</div>', unsafe_allow_html=True)
    st.dataframe(paper_df, use_container_width=True, hide_index=True)

    fig_paper = px.bar(
        paper_df,
        x="Compared_Method",
        y="Value",
        color="Problem",
        barmode="group",
        text="Value",
        title="Paper RL-BS(10) winning rate reported in Nazari et al. (NeurIPS 2018)",
    )
    fig_paper.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_paper.update_layout(
        height=450,
        yaxis_title="Winning rate (%)",
        xaxis_title="Compared method",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig_paper, use_container_width=True)

with tab4:
    st.markdown('<div class="section-title">Winning rate giữa các thuật toán chạy trong project</div>', unsafe_allow_html=True)
    st.dataframe(
        winning_df.style.format({
            "A_better_than_B_%": "{:.2f}",
            "Tie_%": "{:.2f}",
            "Avg_Improvement_%": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    proposed_rows = winning_df[winning_df["Method_A"] == "Proposed-AGIH-2opt"]
    if not proposed_rows.empty:
        fig_win = px.bar(
            proposed_rows,
            x="Method_B",
            y="A_better_than_B_%",
            text="A_better_than_B_%",
            title="Proposed-AGIH-2opt better than other reproduced methods (%)",
        )
        fig_win.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig_win.update_layout(height=420, yaxis_title="Winning rate (%)", xaxis_title="Compared method")
        st.plotly_chart(fig_win, use_container_width=True)

with tab5:
    st.markdown('<div class="section-title">Trực quan hóa lộ trình xe</div>', unsafe_allow_html=True)

    method_options = df[df["Instance"] == selected_instance]["Method"].unique()
    selected_method = st.selectbox("Chọn thuật toán", method_options)

    row = df[(df["Instance"] == selected_instance) & (df["Method"] == selected_method)].iloc[0]
    coords = json.loads(row["Coords"])
    demands = json.loads(row["Demands"])
    routes = json.loads(row["Routes"])

    info1, info2, info3, info4 = st.columns(4)
    info1.metric("Instance", selected_instance)
    info2.metric("Distance", f"{row['Distance']:.4f}")
    info3.metric("Vehicles", int(row["Vehicles"]))
    info4.metric("Runtime", f"{row['Runtime']:.4f}s")

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
