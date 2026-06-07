from src.heuristic.utils import route_distance, route_load, solution_distance


def remove_empty_routes(routes):
    return [route for route in routes if len(route) > 2]


def relocate_solution(routes, demands, capacity, dist, max_iter=100):
    """
    Relocate local search:
    Di chuyển 1 customer từ route này sang vị trí khác nếu làm giảm tổng distance.
    """
    routes = [r[:] for r in routes]
    routes = remove_empty_routes(routes)

    improved = True
    iteration = 0

    while improved and iteration < max_iter:
        improved = False
        iteration += 1

        current_cost = solution_distance(routes, dist)
        best_move = None
        best_cost = current_cost

        for from_r_idx, from_route in enumerate(routes):
            for from_pos in range(1, len(from_route) - 1):
                customer = from_route[from_pos]

                for to_r_idx, to_route in enumerate(routes):
                    for to_pos in range(1, len(to_route)):
                        if from_r_idx == to_r_idx and (
                            to_pos == from_pos or to_pos == from_pos + 1
                        ):
                            continue

                        # Tạo routes mới để thử move
                        new_routes = [r[:] for r in routes]

                        moved_customer = new_routes[from_r_idx].pop(from_pos)

                        # Nếu cùng route và vị trí sau điểm xóa thì phải trừ 1
                        insert_pos = to_pos
                        if from_r_idx == to_r_idx and to_pos > from_pos:
                            insert_pos -= 1

                        new_routes[to_r_idx].insert(insert_pos, moved_customer)
                        new_routes = remove_empty_routes(new_routes)

                        # Check capacity
                        feasible = True
                        for r in new_routes:
                            if route_load(r, demands) > capacity:
                                feasible = False
                                break

                        if not feasible:
                            continue

                        new_cost = solution_distance(new_routes, dist)

                        if new_cost + 1e-9 < best_cost:
                            best_cost = new_cost
                            best_move = new_routes

        if best_move is not None:
            routes = best_move
            improved = True

    return routes