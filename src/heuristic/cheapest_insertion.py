from src.heuristic.utils import route_load


def get_insertion_candidates(routes, customer, demands, capacity, dist):
    candidates = []

    # Chèn vào route hiện có
    for r_idx, route in enumerate(routes):
        if route_load(route, demands) + demands[customer] > capacity:
            continue

        for pos in range(len(route) - 1):
            u = route[pos]
            v = route[pos + 1]

            delta = dist[u, customer] + dist[customer, v] - dist[u, v]

            candidates.append({
                "type": "insert",
                "customer": customer,
                "route_idx": r_idx,
                "position": pos + 1,
                "delta": float(delta),
            })

    # Mở route mới: 0 -> customer -> 0
    if demands[customer] <= capacity:
        delta = dist[0, customer] + dist[customer, 0]
        candidates.append({
            "type": "new_route",
            "customer": customer,
            "route_idx": None,
            "position": None,
            "delta": float(delta),
        })

    return candidates


def apply_insertion(routes, candidate):
    customer = candidate["customer"]

    if candidate["type"] == "new_route":
        routes.append([0, customer, 0])
    else:
        r_idx = candidate["route_idx"]
        pos = candidate["position"]
        routes[r_idx].insert(pos, customer)

    return routes