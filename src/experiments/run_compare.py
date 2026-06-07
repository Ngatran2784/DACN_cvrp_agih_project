import json
import pickle
import pandas as pd
from pathlib import Path

from src.baselines.clarke_wright import solve_clarke_wright
from src.baselines.ortools_solver import solve_cvrp_ortools
from src.paper_method.rl_attention_greedy import solve_paper_attention_greedy
from src.heuristic.proposed_insertion import solve_proposed_insertion_2opt


def add_result(rows, inst, method_name, method_type, routes, cost, runtime, paper_cost, ort_cost):
    coords = inst["coords"].tolist()
    demands = inst["demands"].tolist()
    capacity = int(inst["capacity"])

    gap_vs_paper = (cost - paper_cost) / paper_cost * 100 if paper_cost > 0 else 0.0
    gap_vs_ortools = (cost - ort_cost) / ort_cost * 100 if ort_cost > 0 else 0.0

    rows.append({
        "Instance": inst["id"],
        "Method": method_name,
        "Type": method_type,
        "Distance": round(cost, 4),
        "Vehicles": len(routes),
        "Runtime": round(runtime, 4),
        "Gap_vs_Paper_%": round(gap_vs_paper, 2),
        "Gap_vs_OR_Tools_%": round(gap_vs_ortools, 2),
        "Capacity": capacity,
        "Coords": json.dumps(coords),
        "Demands": json.dumps(demands),
        "Routes": json.dumps(routes),
    })


def compute_winning_rates(df):
    """
    Tính winning rate giữa từng cặp thuật toán.

    A_better_than_B_%:
        Tỷ lệ instance mà Method_A có Distance nhỏ hơn Method_B.

    Avg_Improvement_%:
        Trung bình phần trăm cải thiện của A so với B.
        Giá trị dương nghĩa là A tốt hơn B.
        Giá trị âm nghĩa là A kém hơn B.
    """
    methods = sorted(df["Method"].unique())
    rows = []

    for method_a in methods:
        for method_b in methods:
            if method_a == method_b:
                continue

            a_df = df[df["Method"] == method_a][["Instance", "Distance"]]
            b_df = df[df["Method"] == method_b][["Instance", "Distance"]]

            merged = a_df.merge(
                b_df,
                on="Instance",
                suffixes=("_A", "_B"),
            )

            if merged.empty:
                continue

            wins = (merged["Distance_A"] < merged["Distance_B"]).sum()
            ties = (merged["Distance_A"] == merged["Distance_B"]).sum()
            total = len(merged)

            win_rate = wins / total * 100
            tie_rate = ties / total * 100

            avg_improvement = (
                (merged["Distance_B"] - merged["Distance_A"])
                / merged["Distance_B"]
                * 100
            ).mean()

            rows.append({
                "Method_A": method_a,
                "Method_B": method_b,
                "A_better_than_B_%": round(win_rate, 2),
                "Tie_%": round(tie_rate, 2),
                "Avg_Improvement_%": round(avg_improvement, 2),
                "Num_Instances": total,
            })

    return pd.DataFrame(rows)


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

        # OR-Tools chỉ dùng làm tham chiếu phụ, không đưa vào bảng 3 thuật toán chính
        ort_routes, ort_cost, ort_time = solve_cvrp_ortools(
            coords, demands, capacity, time_limit=1
        )

        # 1. Thuật toán cũ
        cw_routes, cw_cost, cw_time = solve_clarke_wright(
            coords, demands, capacity
        )

        # 2. Thuật toán bài báo
        paper_routes, paper_cost, paper_time = solve_paper_attention_greedy(
            coords, demands, capacity
        )

        # 3. Thuật toán đề xuất
        prop_routes, prop_cost, prop_time = solve_proposed_insertion_2opt(
            coords, demands, capacity
        )

        add_result(
            rows,
            inst,
            "Clarke-Wright",
            "Old heuristic",
            cw_routes,
            cw_cost,
            cw_time,
            paper_cost,
            ort_cost,
        )

        add_result(
            rows,
            inst,
            "Paper-Attention-Greedy",
            "Paper method",
            paper_routes,
            paper_cost,
            paper_time,
            paper_cost,
            ort_cost,
        )

        add_result(
            rows,
            inst,
            "Proposed-AGIH-2opt",
            "Proposed method",
            prop_routes,
            prop_cost,
            prop_time,
            paper_cost,
            ort_cost,
        )

    df = pd.DataFrame(rows)
    df.to_csv("results/results_summary.csv", index=False)

    summary = df.groupby(["Method", "Type"]).agg({
        "Distance": "mean",
        "Vehicles": "mean",
        "Runtime": "mean",
        "Gap_vs_Paper_%": "mean",
        "Gap_vs_OR_Tools_%": "mean",
    }).reset_index()

    summary.to_csv("results/results_avg.csv", index=False)

    winning_df = compute_winning_rates(df)
    winning_df.to_csv("results/winning_rate.csv", index=False)

    print("\nSaved:")
    print("- results/results_summary.csv")
    print("- results/results_avg.csv")
    print("- results/winning_rate.csv")
    print()
    print(summary)
    print()
    print(winning_df)


if __name__ == "__main__":
    run_experiment()
