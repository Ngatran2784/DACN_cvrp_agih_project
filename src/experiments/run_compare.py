import json
import pickle
import time
from pathlib import Path

import pandas as pd

from src.baselines.clarke_wright import solve_clarke_wright
from src.heuristic.proposed_from_paper import solve_proposed_from_paper_routes


METHOD_OLD = "Clarke-Wright"
METHOD_PAPER = "Paper-RL-Attention"
METHOD_PROPOSED = "Proposed-AGIH-2opt"


def _safe_json_loads(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


def compute_winning_rates(df):
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

            if len(merged) == 0:
                continue

            wins = (merged["Distance_A"] < merged["Distance_B"]).sum()
            ties = (merged["Distance_A"] == merged["Distance_B"]).sum()
            total = len(merged)

            avg_improvement = (
                (merged["Distance_B"] - merged["Distance_A"])
                / merged["Distance_B"]
                * 100
            ).mean()

            rows.append({
                "Method_A": method_a,
                "Method_B": method_b,
                "A_better_than_B_%": round(wins / total * 100, 2),
                "Tie_%": round(ties / total * 100, 2),
                "Avg_Improvement_%": round(avg_improvement, 2),
                "Num_Instances": int(total),
            })

    return pd.DataFrame(rows)


def add_long_result(rows, inst, method, method_type, routes, cost, runtime):
    rows.append({
        "Instance": inst["id"],
        "Method": method,
        "Type": method_type,
        "Distance": round(float(cost), 4),
        "Vehicles": int(len(routes)),
        "Runtime": round(float(runtime), 4),
        "Capacity": int(inst["capacity"]),
        "Coords": json.dumps(inst["coords"].tolist()),
        "Demands": json.dumps(inst["demands"].tolist()),
        "Routes": json.dumps(routes),
    })


def create_paper_template(path):
    template = pd.DataFrame(columns=[
        "Instance",
        "Paper_Distance",
        "Paper_Vehicles",
        "Paper_Runtime",
        "Paper_Routes",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(path, index=False)


def load_paper_routes(paper_routes_path):
    paper_path = Path(paper_routes_path)

    if not paper_path.exists():
        create_paper_template(Path("results/paper_routes_template.csv"))
        raise FileNotFoundError(
            "Không tìm thấy results/paper_routes.csv. "
            "Hãy chạy thuật toán bài báo trên Colab và xuất file theo format: "
            "Instance, Paper_Distance, Paper_Vehicles, Paper_Runtime, Paper_Routes. "
            "Đã tạo file mẫu: results/paper_routes_template.csv"
        )

    paper_df = pd.read_csv(paper_path)

    required_cols = {
        "Instance",
        "Paper_Routes",
    }

    missing = required_cols - set(paper_df.columns)
    if missing:
        raise ValueError(
            f"paper_routes.csv thiếu cột {sorted(missing)}. "
            "Cột bắt buộc tối thiểu: Instance, Paper_Routes."
        )

    paper_by_instance = {}
    for _, row in paper_df.iterrows():
        routes = _safe_json_loads(row["Paper_Routes"])
        if routes is None:
            raise ValueError(f"Paper_Routes bị rỗng ở instance {row['Instance']}")

        routes = [[int(node) for node in route] for route in routes]

        paper_by_instance[row["Instance"]] = {
            "routes": routes,
            "reported_distance": float(row["Paper_Distance"]) if "Paper_Distance" in row and not pd.isna(row.get("Paper_Distance")) else None,
            "reported_runtime": float(row["Paper_Runtime"]) if "Paper_Runtime" in row and not pd.isna(row.get("Paper_Runtime")) else 0.0,
        }

    return paper_by_instance


def route_distance_from_routes(routes, dist_matrix):
    total = 0.0
    for route in routes:
        for i in range(len(route) - 1):
            total += float(dist_matrix[route[i], route[i + 1]])
    return total


def compute_distance_matrix(coords):
    import numpy as np

    n = len(coords)
    dist = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            dx = float(coords[i][0]) - float(coords[j][0])
            dy = float(coords[i][1]) - float(coords[j][1])
            dist[i, j] = (dx * dx + dy * dy) ** 0.5

    return dist


def make_wide_row(inst, cw_routes, cw_cost, cw_time, paper_routes, paper_cost, paper_time, prop_routes, prop_cost, prop_time):
    values = {
        METHOD_OLD: cw_cost,
        METHOD_PAPER: paper_cost,
        METHOD_PROPOSED: prop_cost,
    }

    best_method = min(values, key=values.get)

    return {
        "Instance": inst["id"],
        "CLARKE_WRIGHT_DISTANCE": round(float(cw_cost), 4),
        "CLARKE_WRIGHT_RUNTIME_S": round(float(cw_time), 4),
        "CLARKE_WRIGHT_VEHICLES": int(len(cw_routes)),
        "PAPER_RL_ATTENTION_DISTANCE": round(float(paper_cost), 4),
        "PAPER_RL_ATTENTION_RUNTIME_S": round(float(paper_time), 4),
        "PAPER_RL_ATTENTION_VEHICLES": int(len(paper_routes)),
        "PROPOSED_AGIH_2OPT_DISTANCE": round(float(prop_cost), 4),
        "PROPOSED_AGIH_2OPT_RUNTIME_S": round(float(prop_time), 4),
        "PROPOSED_AGIH_2OPT_VEHICLES": int(len(prop_routes)),
        "BEST_ALGORITHM": best_method,
    }


def run_experiment(
    dataset_path="data/cvrp20_test.pkl",
    paper_routes_path="results/paper_routes.csv",
    max_instances=20,
):
    Path("results").mkdir(exist_ok=True)

    with open(dataset_path, "rb") as f:
        instances = pickle.load(f)

    if max_instances is not None:
        instances = instances[:max_instances]

    paper_by_instance = load_paper_routes(paper_routes_path)

    missing_instances = [
        inst["id"] for inst in instances
        if inst["id"] not in paper_by_instance
    ]

    if missing_instances:
        raise ValueError(
            "paper_routes.csv chưa có đủ instance cần so sánh. "
            f"Thiếu {len(missing_instances)} instance, ví dụ: {missing_instances[:5]}"
        )

    long_rows = []
    wide_rows = []

    for inst in instances:
        coords = inst["coords"]
        demands = inst["demands"]
        capacity = int(inst["capacity"])
        dist = compute_distance_matrix(coords)

        print(f"Running {inst['id']}...")

        # 1. Old algorithm: Clarke-Wright
        cw_routes, cw_cost, cw_time = solve_clarke_wright(coords, demands, capacity)

        # 2. Paper algorithm: route produced by Nazari RL-Attention / paper reproduction from Colab
        paper_info = paper_by_instance[inst["id"]]
        paper_routes = paper_info["routes"]
        paper_cost = route_distance_from_routes(paper_routes, dist)
        paper_time = paper_info["reported_runtime"]

        # 3. Proposed algorithm: improve paper route using insertion/local search
        prop_routes, prop_cost, prop_time = solve_proposed_from_paper_routes(
            coords,
            demands,
            capacity,
            paper_routes,
        )

        add_long_result(
            long_rows,
            inst,
            METHOD_OLD,
            "Old heuristic",
            cw_routes,
            cw_cost,
            cw_time,
        )

        add_long_result(
            long_rows,
            inst,
            METHOD_PAPER,
            "Paper method",
            paper_routes,
            paper_cost,
            paper_time,
        )

        add_long_result(
            long_rows,
            inst,
            METHOD_PROPOSED,
            "Proposed method",
            prop_routes,
            prop_cost,
            prop_time,
        )

        wide_rows.append(
            make_wide_row(
                inst,
                cw_routes,
                cw_cost,
                cw_time,
                paper_routes,
                paper_cost,
                paper_time,
                prop_routes,
                prop_cost,
                prop_time,
            )
        )

    long_df = pd.DataFrame(long_rows)
    wide_df = pd.DataFrame(wide_rows)

    long_df.to_csv("results/results_summary.csv", index=False)
    wide_df.to_csv("results/project_wide.csv", index=False)

    avg_df = long_df.groupby(["Method", "Type"]).agg({
        "Distance": "mean",
        "Vehicles": "mean",
        "Runtime": "mean",
    }).reset_index()

    avg_df.to_csv("results/results_avg.csv", index=False)

    winning_df = compute_winning_rates(long_df)
    winning_df.to_csv("results/winning_rate.csv", index=False)

    print("\nSaved:")
    print("- results/results_summary.csv")
    print("- results/project_wide.csv")
    print("- results/results_avg.csv")
    print("- results/winning_rate.csv")
    print()
    print(avg_df)
    print()
    print(winning_df)


if __name__ == "__main__":
    run_experiment()