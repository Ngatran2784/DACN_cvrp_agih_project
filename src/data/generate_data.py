import pickle
import numpy as np
from pathlib import Path


def generate_cvrp_instances(num_instances, n_customers, capacity, seed=42):
    rng = np.random.default_rng(seed)
    instances = []

    for instance_id in range(num_instances):
        coords = rng.random((n_customers + 1, 2), dtype=np.float32)

        demands = np.zeros(n_customers + 1, dtype=np.int32)
        demands[1:] = rng.integers(1, 10, size=n_customers)

        instances.append({
            "id": f"CVRP{n_customers}_{instance_id:04d}",
            "coords": coords,
            "demands": demands,
            "capacity": capacity,
        })

    return instances


def save_instances(instances, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(instances, f)


if __name__ == "__main__":
    configs = [
        (20, 30),
        (50, 40),
    ]

    for n_customers, capacity in configs:
        train = generate_cvrp_instances(
            num_instances=1000,
            n_customers=n_customers,
            capacity=capacity,
            seed=100 + n_customers,
        )

        test = generate_cvrp_instances(
            num_instances=100,
            n_customers=n_customers,
            capacity=capacity,
            seed=200 + n_customers,
        )

        save_instances(train, f"data/cvrp{n_customers}_train.pkl")
        save_instances(test, f"data/cvrp{n_customers}_test.pkl")

        print(f"Saved CVRP{n_customers}: train={len(train)}, test={len(test)}")