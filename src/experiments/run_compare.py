import json
import pickle
from pathlib import Path

import pandas as pd

from src.baselines.clarke_wright import solve_clarke_wright
from src.heuristic.proposed_insertion import solve_proposed_insertion_2opt


def add_result(rows, inst, method_name, method_type, routes, cost, runtime):
    coords = inst["coords"].tolist()
    demands = inst["demands"].tolist()
    capacity = int(inst["capacity"])

    rows.append({
        "INSTANCE": inst["id"],
        "METHOD": method_name,
        "TYPE": method_type,
        "DISTANCE": round(float(cost), 4),
        "VEHICLES": len(routes),
        "RUNTIME_S": round(float(runtime), 4),
        "CAPACITY": capacity,
        "COORDS": json.dumps(coords),
        "DEMANDS": json.dumps(demands),
        "ROUTES": json.dumps(routes),
    })


def compute_winning_rates(df):
    methods = sorted(df["METHOD"].unique())
    rows = []

    for method_a in methods:
        for method_b in methods:
            if method_a == method_b:
                continue

            a_df = df[df["METHOD"] == method_a][["INSTANCE", "DISTANCE"]]
            b_df = df[df["METHOD"] == method_b][["INSTANCE", "DISTANCE"]]
            merged = a_df.merge(b_df, on="INSTANCE", suffixes=("_A", "_B"))

            if merged.empty:
                continue

            wins = (merged["DISTANCE_A"] < merged["DISTANCE_B"]).sum()
            ties = (merged["DISTANCE_A"] == merged["DISTANCE_B"]).sum()
            total = len(merged)
            avg_improvement = (((merged["DISTANCE_B"] - merged["DISTANCE_A"]) / merged["DISTANCE_B"]) * 100).mean()

            rows.append({
                "METHOD_A": method_a,
                "METHOD_B": method_b,
                "A_BETTER_THAN_B_%": round(float(wins / total * 100), 2),
                "TIE_%": round(float(ties / total * 100), 2),
                "AVG_IMPROVEMENT_%": round(float(avg_improvement), 2),
                "NUM_INSTANCES": total,
            })

    return pd.DataFrame(rows)


def create_project_wide(df):
    rows = []

    for instance, group in df.groupby("INSTANCE"):
        row = {"INSTANCE": instance}
        best_method = None
        best_distance = float("inf")

        for _, item in group.iterrows():
            key = item["METHOD"].upper().replace("-", "_").replace(" ", "_")
            row[f"{key}_DISTANCE"] = item["DISTANCE"]
            row[f"{key}_VEHICLES"] = item["VEHICLES"]
            row[f"{key}_RUNTIME_S"] = item["RUNTIME_S"]

            if item["DISTANCE"] < best_distance:
                best_distance = item["DISTANCE"]
                best_method = item["METHOD"]

        row["BEST_ALGORITHM"] = best_method
        rows.append(row)

    return pd.DataFrame(rows)


def create_paper_reported_results():
    # Reported from Nazari et al., NeurIPS 2018, Figure 3 tables.
    # Only keep paper-vs-old heuristic metrics here, so the dashboard stays focused on 3 algorithms.
    rows = [
        {
            "PAPER_METHOD": "Paper RL-BS(10)",
            "PROBLEM_SIZE": "VRP50",
            "COMPARED_WITH": "Clarke-Wright / CW-Greedy",
            "WINNING_RATE_%": 99.8,
            "SOURCE_NOTE": "Nazari et al. 2018, Figure 3c",
        },
        {
            "PAPER_METHOD": "Paper RL-BS(10)",
            "PROBLEM_SIZE": "VRP100",
            "COMPARED_WITH": "Clarke-Wright / CW-Greedy",
            "WINNING_RATE_%": 100.0,
            "SOURCE_NOTE": "Nazari et al. 2018, Figure 3d",
        },
    ]
    return pd.DataFrame(rows)


def create_three_algorithm_summary(project_avg, winning_df, paper_df):
    def get_avg(method, col):
        found = project_avg[project_avg["METHOD"] == method]
        if found.empty:
            return None
        return round(float(found.iloc[0][col]), 4)

    def get_win(method_a, method_b):
        found = winning_df[(winning_df["METHOD_A"] == method_a) & (winning_df["METHOD_B"] == method_b)]
        if found.empty:
            return None
        return round(float(found.iloc[0]["A_BETTER_THAN_B_%"]), 2)

    paper_vrp50 = paper_df[paper_df["PROBLEM_SIZE"] == "VRP50"]
    paper_vrp100 = paper_df[paper_df["PROBLEM_SIZE"] == "VRP100"]

    rows = [
        {
            "GROUP": "Thuật toán cũ",
            "METHOD": "Clarke-Wright",
            "DATA_SOURCE": "Project chạy trực tiếp",
            "AVG_DISTANCE_CVRP20": get_avg("Clarke-Wright", "DISTANCE"),
            "AVG_VEHICLES_CVRP20": get_avg("Clarke-Wright", "VEHICLES"),
            "AVG_RUNTIME_S_CVRP20": get_avg("Clarke-Wright", "RUNTIME_S"),
            "PROJECT_WIN_VS_CLARKE_WRIGHT_%": 0.0,
            "PAPER_WIN_VS_OLD_VRP50_%": None,
            "PAPER_WIN_VS_OLD_VRP100_%": None,
            "NOTE": "Baseline heuristic truyền thống",
        },
        {
            "GROUP": "Thuật toán bài báo",
            "METHOD": "Paper RL-BS(10)",
            "DATA_SOURCE": "Số liệu công bố trong bài báo",
            "AVG_DISTANCE_CVRP20": None,
            "AVG_VEHICLES_CVRP20": None,
            "AVG_RUNTIME_S_CVRP20": None,
            "PROJECT_WIN_VS_CLARKE_WRIGHT_%": None,
            "PAPER_WIN_VS_OLD_VRP50_%": round(float(paper_vrp50.iloc[0]["WINNING_RATE_%"]), 2) if not paper_vrp50.empty else None,
            "PAPER_WIN_VS_OLD_VRP100_%": round(float(paper_vrp100.iloc[0]["WINNING_RATE_%"]), 2) if not paper_vrp100.empty else None,
            "NOTE": "Paper công bố winning rate; không công bố route từng instance",
        },
        {
            "GROUP": "Thuật toán đề xuất",
            "METHOD": "Proposed-AGIH-2opt",
            "DATA_SOURCE": "Project chạy trực tiếp",
            "AVG_DISTANCE_CVRP20": get_avg("Proposed-AGIH-2opt", "DISTANCE"),
            "AVG_VEHICLES_CVRP20": get_avg("Proposed-AGIH-2opt", "VEHICLES"),
            "AVG_RUNTIME_S_CVRP20": get_avg("Proposed-AGIH-2opt", "RUNTIME_S"),
            "PROJECT_WIN_VS_CLARKE_WRIGHT_%": get_win("Proposed-AGIH-2opt", "Clarke-Wright"),
            "PAPER_WIN_VS_OLD_VRP50_%": None,
            "PAPER_WIN_VS_OLD_VRP100_%": None,
            "NOTE": "Hybrid insertion + 2-opt + relocate",
        },
    ]

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

        cw_routes, cw_cost, cw_time = solve_clarke_wright(coords, demands, capacity)
        prop_routes, prop_cost, prop_time = solve_proposed_insertion_2opt(coords, demands, capacity)

        add_result(rows, inst, "Clarke-Wright", "Old heuristic", cw_routes, cw_cost, cw_time)
        add_result(rows, inst, "Proposed-AGIH-2opt", "Proposed method", prop_routes, prop_cost, prop_time)

    df = pd.DataFrame(rows)
    df.to_csv("results/results_summary.csv", index=False)

    project_avg = df.groupby(["METHOD", "TYPE"]).agg({
        "DISTANCE": "mean",
        "VEHICLES": "mean",
        "RUNTIME_S": "mean",
    }).reset_index()
    project_avg.to_csv("results/results_avg.csv", index=False)

    project_wide = create_project_wide(df)
    project_wide.to_csv("results/project_wide.csv", index=False)

    winning_df = compute_winning_rates(df)
    winning_df.to_csv("results/winning_rate.csv", index=False)

    paper_df = create_paper_reported_results()
    paper_df.to_csv("results/paper_reported_results.csv", index=False)

    three_summary = create_three_algorithm_summary(project_avg, winning_df, paper_df)
    three_summary.to_csv("results/three_algorithm_summary.csv", index=False)

    print("\nSaved:")
    print("- results/results_summary.csv")
    print("- results/results_avg.csv")
    print("- results/project_wide.csv")
    print("- results/winning_rate.csv")
    print("- results/paper_reported_results.csv")
    print("- results/three_algorithm_summary.csv")
    print("\nProject average:")
    print(project_avg)
    print("\nThree algorithm summary:")
    print(three_summary)


if __name__ == "__main__":
    run_experiment()
