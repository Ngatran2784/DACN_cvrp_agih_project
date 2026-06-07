import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.baselines.ortools_solver import solve_cvrp_ortools
from src.heuristic.utils import flatten_routes


def build_features(coords, demands, capacity, served, current_node, remaining_load):
    features = []

    current_x, current_y = coords[current_node]

    for node in range(1, len(demands)):
        x, y = coords[node]

        dist_to_current = ((x - current_x) ** 2 + (y - current_y) ** 2) ** 0.5

        features.append([
            x,
            y,
            demands[node] / capacity,
            served[node],
            remaining_load / capacity,
            dist_to_current,
        ])

    return np.array(features, dtype=np.float32)


def create_paper_attention_samples(instances, max_instances=200, time_limit=1):
    samples = []

    for inst in tqdm(instances[:max_instances]):
        coords = inst["coords"]
        demands = inst["demands"]
        capacity = int(inst["capacity"])

        routes, _, _ = solve_cvrp_ortools(
            coords, demands, capacity, time_limit=time_limit
        )

        sequence = flatten_routes(routes)

        served = np.zeros(len(demands), dtype=np.float32)
        current_node = 0
        remaining_load = capacity

        for target in sequence:
            # Nếu customer tiếp theo không đủ tải thì coi như xe quay về depot mở route mới
            if demands[target] > remaining_load:
                current_node = 0
                remaining_load = capacity

            features = build_features(
                coords,
                demands,
                capacity,
                served,
                current_node,
                remaining_load,
            )

            # Mask customer đã phục vụ hoặc demand vượt tải còn lại
            mask = np.zeros(len(demands) - 1, dtype=np.bool_)

            for node in range(1, len(demands)):
                idx = node - 1

                if served[node] > 0.5:
                    mask[idx] = True

                if demands[node] > remaining_load:
                    mask[idx] = True

            samples.append({
                "features": features,
                "mask": mask,
                "target": int(target - 1),
            })

            served[target] = 1.0
            remaining_load -= demands[target]
            current_node = target

    return samples


if __name__ == "__main__":
    with open("data/cvrp20_train.pkl", "rb") as f:
        train_instances = pickle.load(f)

    samples = create_paper_attention_samples(
        train_instances,
        max_instances=200,
        time_limit=1,
    )

    Path("data/train_samples").mkdir(parents=True, exist_ok=True)

    with open("data/train_samples/paper_attention_cvrp20.pkl", "wb") as f:
        pickle.dump(samples, f)

    print("Saved training samples:", len(samples))
    print("File: data/train_samples/paper_attention_cvrp20.pkl")