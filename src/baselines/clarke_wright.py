import time

from src.heuristic.utils import (
    compute_distance_matrix,
    solution_distance,
    route_load,
)


def solve_clarke_wright(coords, demands, capacity):
    """
    Clarke-Wright Savings heuristic for CVRP.

    Input:
        coords: tọa độ depot + customer, depot là node 0
        demands: demand của từng node, demand[0] = 0
        capacity: sức chứa xe

    Output:
        routes: danh sách route
        cost: tổng quãng đường
        runtime: thời gian chạy
    """
    start = time.time()

    dist = compute_distance_matrix(coords)
    n_customers = len(coords) - 1

    # Ban đầu mỗi customer là 1 route riêng: 0 -> i -> 0
    routes = {i: [0, i, 0] for i in range(1, n_customers + 1)}
    node_to_route = {i: i for i in range(1, n_customers + 1)}

    # Tính savings cho từng cặp customer
    savings = []
    for i in range(1, n_customers + 1):
        for j in range(i + 1, n_customers + 1):
            saving = dist[0, i] + dist[0, j] - dist[i, j]
            savings.append((saving, i, j))

    # Sắp xếp savings giảm dần
    savings.sort(reverse=True)

    for _, i, j in savings:
        route_i_id = node_to_route.get(i)
        route_j_id = node_to_route.get(j)

        if route_i_id is None or route_j_id is None:
            continue

        if route_i_id == route_j_id:
            continue

        route_i = routes[route_i_id]
        route_j = routes[route_j_id]

        # Chỉ merge nếu i ở cuối route_i và j ở đầu route_j
        can_merge_ij = route_i[-2] == i and route_j[1] == j

        # Hoặc j ở cuối route_j và i ở đầu route_i
        can_merge_ji = route_j[-2] == j and route_i[1] == i

        if not can_merge_ij and not can_merge_ji:
            continue

        total_load = route_load(route_i, demands) + route_load(route_j, demands)

        if total_load > capacity:
            continue

        if can_merge_ij:
            # 0 -> ... -> i -> 0  +  0 -> j -> ... -> 0
            # thành 0 -> ... -> i -> j -> ... -> 0
            new_route = route_i[:-1] + route_j[1:]
            keep_id = route_i_id
            remove_id = route_j_id
        else:
            # 0 -> ... -> j -> 0  +  0 -> i -> ... -> 0
            new_route = route_j[:-1] + route_i[1:]
            keep_id = route_j_id
            remove_id = route_i_id

        routes[keep_id] = new_route
        del routes[remove_id]

        for node in new_route:
            if node != 0:
                node_to_route[node] = keep_id

    final_routes = list(routes.values())
    cost = solution_distance(final_routes, dist)
    runtime = time.time() - start

    return final_routes, float(cost), runtime