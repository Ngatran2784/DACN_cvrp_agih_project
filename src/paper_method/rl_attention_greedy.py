import time
import numpy as np
import torch

from src.model.attention_policy import AttentionPolicy
from src.heuristic.utils import compute_distance_matrix, solution_distance


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


def build_mask(demands, served, remaining_load):
    mask = np.zeros(len(demands) - 1, dtype=np.bool_)

    for node in range(1, len(demands)):
        idx = node - 1

        if served[node] > 0.5:
            mask[idx] = True

        if demands[node] > remaining_load:
            mask[idx] = True

    return mask


def solve_paper_attention_greedy(
    coords,
    demands,
    capacity,
    model_path="results/paper_attention_cvrp20.pt",
):
    """
    Paper-style Attention Greedy for CVRP.

    Đây là bản tái hiện đơn giản theo hướng bài báo:
    - Attention policy chọn customer tiếp theo.
    - Mask customer đã phục vụ hoặc demand vượt remaining_load.
    - Nếu không còn customer hợp lệ thì quay về depot, mở route mới.
    """
    start = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dist = compute_distance_matrix(coords)

    model = AttentionPolicy(input_dim=6, hidden_dim=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    n_customers = len(demands) - 1

    served = np.zeros(len(demands), dtype=np.float32)
    remaining_load = capacity
    current_node = 0

    routes = []
    current_route = [0]

    while served[1:].sum() < n_customers:
        mask_np = build_mask(demands, served, remaining_load)

        # Nếu không còn customer hợp lệ với tải hiện tại, quay về depot
        if mask_np.all():
            if current_route[-1] != 0:
                current_route.append(0)

            if len(current_route) > 2:
                routes.append(current_route)

            current_route = [0]
            current_node = 0
            remaining_load = capacity
            continue

        features = build_features(
            coords,
            demands,
            capacity,
            served,
            current_node,
            remaining_load,
        )

        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        mask = torch.tensor(mask_np, dtype=torch.bool).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x, mask)
            customer_idx = int(torch.argmax(logits, dim=1).item())

        customer = customer_idx + 1

        current_route.append(customer)
        served[customer] = 1.0
        remaining_load -= demands[customer]
        current_node = customer

    if current_route[-1] != 0:
        current_route.append(0)

    if len(current_route) > 2:
        routes.append(current_route)

    cost = solution_distance(routes, dist)
    runtime = time.time() - start

    return routes, float(cost), runtime