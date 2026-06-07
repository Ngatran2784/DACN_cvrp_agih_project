from src.heuristic.utils import route_distance


def two_opt_route(route, dist):
    best_route = route[:]
    best_cost = route_distance(best_route, dist)
    improved = True

    while improved:
        improved = False

        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                if j - i == 1:
                    continue

                new_route = best_route[:]
                new_route[i:j] = reversed(best_route[i:j])

                new_cost = route_distance(new_route, dist)

                if new_cost + 1e-9 < best_cost:
                    best_route = new_route
                    best_cost = new_cost
                    improved = True
                    break

            if improved:
                break

    return best_route


def two_opt_solution(routes, dist):
    return [two_opt_route(route, dist) for route in routes]