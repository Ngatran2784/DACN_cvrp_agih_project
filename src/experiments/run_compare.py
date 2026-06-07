import json
import pickle
import pandas as pd
from pathlib import Path

from src.baselines.clarke_wright import solve_clarke_wright
from src.baselines.ortools_solver import solve_cvrp_ortools
from src.heuristic.proposed_insertion import solve_proposed_insertion_2opt


def add_result(rows, inst, method_name, method_type, routes, cost, runtime, ort_cost):
    coords = inst["coords"].tolist()
    demands = inst["demands"].tolist()
    capacity = int(inst["capacity"])

    gap_vs_ortools = (cost - ort_cost) / ort_cost * 100 if ort_cost > 0 else 0.0

    rows.append({
        "Instance": inst["id"],
        "Method": method_name,
        "Type": method_type,
        "Distance": round(float(cost), 4),
        "Vehicles": len(routes),
        "Runtime": round(float(runtime), 4),
        "Gap_vs_OR_Tools_%": round(float(gap_vs_ortools), 2),
        "Capacity": capacity,
        "Coords": json.dumps(coords),
        "Demands": json.dumps(demands),
        "Routes": json.dumps(routes),
    })


def compute_winning_rates(df):
    methods = sorted(df["Method"].unique())
    rows = []

    for method_a in methods:
        for method_b in methods:
            if method_a == method_b:
                continue

            a_df = df[df["Method"] == method_a][["Instance", "Distance"]]
            b_df = df[df["Method"] == method_b][["Instance", "Distance"]]

            merged = a_df.merge(b_df, on="Instance", suffixes=("_A", "_B"))
            if merged.empty:
                continue

            wins = (merged["Distance_A"] < merged["Distance_B"]).sum()
            ties = (merged["Distance_A"] == merged["Distance_B"]).sum()
            total = len(merged)

            avg_improvement = (
                (merged["Distance_B"] - merged["Distance_A"]) / merged["Distance_B"] * 100
            ).mean()

            rows.append({
                "Method_A": method_a,
                "Method_B": method_b,
                "A_better_than_B_%": round(wins / total * 100, 2),
                "Tie_%": round(ties / total * 100, 2),
                "Avg_Improvement_%": round(float(avg_improvement), 2),
                "Num_Instances": total,
            })

    return pd.DataFrame(rows)


def write_paper_reported_results():
    """
    Số liệu này lấy đúng theo bảng winning-rate trong bài Nazari et al. (NeurIPS 2018).
    Không trộn với kết quả chạy lại trên dataset của project.
    """
    rows = [
        {
            "Problem": "VRP50",
            "Paper_Method": "RL-BS(10)",
            "Compared_Method": "CW-Greedy",
            "Metric": "Winning rate (%)",
            "Value": 99.8,
            "Meaning": "RL-BS(10) cho route ngắn hơn CW-Greedy trong 99.8% mẫu VRP50 theo bài báo.",
        },
        {
            "Problem": "VRP50",
            "Paper_Method": "RL-BS(10)",
            "Compared_Method": "OR-Tools",
            "Metric": "Winning rate (%)",
            "Value": 60.2,
            "Meaning": "RL-BS(10) cho route ngắn hơn OR-Tools trong 60.2% mẫu VRP50 theo bài báo.",
        },
        {
            "Problem": "VRP100",
            "Paper_Method": "RL-BS(10)",
            "Compared_Method": "CW-Greedy",
            "Metric": "Winning rate (%)",
            "Value": 100.0,
            "Meaning": "RL-BS(10) cho route ngắn hơn CW-Greedy trong 100.0% mẫu VRP100 theo bài báo.",
        },
        {
            "Problem": "VRP100",
            "Paper_Method": "RL-BS(10)",
            "Compared_Method": "OR-Tools",
            "Metric": "Winning rate (%)",
            "Value": 62.2,
            "Meaning": "RL-BS(10) cho route ngắn hơn OR-Tools trong 62.2% mẫu VRP100 theo bài báo.",
        },
    ]

    paper_df = pd.DataFrame(rows)
    paper_df.to_csv("results/paper_reported_results.csv", index=False)
    return paper_df


def write_three_method_comparison(summary_df, winning_df, paper_df):
    def get_avg_distance(method):
        row = summary_df[summary_df["Method"] == method]
        if row.empty:
            return None
        return float(row.iloc[0]["Distance"])

    def get_avg_runtime(method):
        row = summary_df[summary_df["Method"] == method]
        if row.empty:
            return None
        return float(row.iloc[0]["Runtime"])

    def get_win_rate(a, b):
        row = winning_df[(winning_df["Method_A"] == a) & (winning_df["Method_B"] == b)]
        if row.empty:
            return None
        return float(row.iloc[0]["A_better_than_B_%"])

    cw_distance = get_avg_distance("Clarke-Wright")
    prop_distance = get_avg_distance("Proposed-AGIH-2opt")
    prop_win_cw = get_win_rate("Proposed-AGIH-2opt", "Clarke-Wright")

    paper_vs_cw = paper_df[
        (paper_df["Problem"] == "VRP100") &
        (paper_df["Paper_Method"] == "RL-BS(10)") &
        (paper_df["Compared_Method"] == "CW-Greedy")
    ].iloc[0]["Value"]

    rows = [
        {
            "Group": "Thuật toán cũ",
            "Algorithm": "Clarke-Wright Savings",
            "Source": "Project code",
            "Evidence": "Chạy trực tiếp trên bộ test ngẫu nhiên của project",
            "Main_Result": f"Avg Distance = {cw_distance:.4f}" if cw_distance is not None else "N/A",
            "Note": "Baseline heuristic truyền thống.",
        },
        {
            "Group": "Thuật toán bài báo",
            "Algorithm": "Paper RL-BS(10)",
            "Source": "Nazari et al., NeurIPS 2018 reported result",
            "Evidence": "Số liệu winning rate đúng theo bài báo, không phải model tự train lại",
            "Main_Result": f"Thắng CW-Greedy {paper_vs_cw:.1f}% trên VRP100 theo paper",
            "Note": "Dùng làm mốc tham chiếu bài báo. Route chi tiết không có vì paper không công bố route từng instance.",
        },
        {
            "Group": "Thuật toán đề xuất",
            "Algorithm": "Proposed AGIH-2opt",
            "Source": "Project code",
            "Evidence": "Chạy trực tiếp trên cùng bộ test với Clarke-Wright",
            "Main_Result": (
                f"Avg Distance = {prop_distance:.4f}; thắng Clarke-Wright {prop_win_cw:.2f}% instance"
                if prop_distance is not None and prop_win_cw is not None else "N/A"
            ),
            "Note": "Hybrid insertion + 2-opt + relocate; route chi tiết được hiển thị trong dashboard.",
        },
    ]

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv("results/three_method_comparison.csv", index=False)
    return comparison_df


def run_experiment(dataset_path="data/cvrp20_test.pkl", max_instances=20):
    Path("results").mkdir(exist_ok=True)

    with open(dataset_path, "rb") as f:
        instances = pickle.load(f)

    instances = instances[:max_instances]
    rows = []

    for inst in instances:
        coords = inst["coords"]
        demands = inst["demands"]
        capacity = int(inst["capacity"])

        print(f"Running {inst['id']}...")

        # OR-Tools không phải 1 trong 3 thuật toán chính; dùng làm reference solver cho gap.
        ort_routes, ort_cost, ort_time = solve_cvrp_ortools(
            coords, demands, capacity, time_limit=1
        )

        # 1. Thuật toán cũ
        cw_routes, cw_cost, cw_time = solve_clarke_wright(
            coords, demands, capacity
        )

        # 2. Thuật toán đề xuất
        prop_routes, prop_cost, prop_time = solve_proposed_insertion_2opt(
            coords, demands, capacity
        )

        # Lưu các thuật toán chạy được trên project.
        add_result(
            rows, inst, "Clarke-Wright", "Old heuristic",
            cw_routes, cw_cost, cw_time, ort_cost
        )

        add_result(
            rows, inst, "Proposed-AGIH-2opt", "Proposed method",
            prop_routes, prop_cost, prop_time, ort_cost
        )

        add_result(
            rows, inst, "OR-Tools Reference", "Reference solver",
            ort_routes, ort_cost, ort_time, ort_cost
        )

    df = pd.DataFrame(rows)
    df.to_csv("results/results_summary.csv", index=False)

    summary = df.groupby(["Method", "Type"]).agg({
        "Distance": "mean",
        "Vehicles": "mean",
        "Runtime": "mean",
        "Gap_vs_OR_Tools_%": "mean",
    }).reset_index()

    summary.to_csv("results/results_avg.csv", index=False)

    winning_df = compute_winning_rates(df)
    winning_df.to_csv("results/winning_rate.csv", index=False)

    paper_df = write_paper_reported_results()
    comparison_df = write_three_method_comparison(summary, winning_df, paper_df)

    print("\nSaved:")
    print("- results/results_summary.csv")
    print("- results/results_avg.csv")
    print("- results/winning_rate.csv")
    print("- results/paper_reported_results.csv")
    print("- results/three_method_comparison.csv")
    print("\nSummary:")
    print(summary)
    print("\nThree-method comparison:")
    print(comparison_df)


if __name__ == "__main__":
    run_experiment()
