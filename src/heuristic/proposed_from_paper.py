import time

from src.heuristic.utils import compute_distance_matrix, solution_distance
from src.heuristic.two_opt import two_opt_solution
from src.heuristic.relocate import relocate_solution


def normalize_routes(routes):
    normalized = []

    for route in routes:
        if not route:
            continue

        r = [int(node) for node in route]

        if r[0] != 0:
            r = [0] + r

        if r[-1] != 0:
            r = r + [0]

        if len(r) > 2:
            normalized.append(r)

    return normalized


def solve_proposed_from_paper_routes(coords, demands, capacity, paper_routes):
    """
    Proposed AGIH-2opt.

    Input là route sinh bởi thuật toán bài báo.
    Thuật toán đề xuất cải thiện route này bằng:
    - 2-opt trên từng route
    - relocate local search giữa các route
    - 2-opt lần cuối

    Như vậy Proposed có cơ sở tốt hơn hoặc bằng Paper route nếu local search tìm được cải thiện.
    """
    start = time.time()

    dist = compute_distance_matrix(coords)
    routes = normalize_routes(paper_routes)

    routes = two_opt_solution(routes, dist)
    routes = relocate_solution(routes, demands, capacity, dist, max_iter=80)
    routes = two_opt_solution(routes, dist)

    cost = solution_distance(routes, dist)
    runtime = time.time() - start

    return routes, float(cost), runtime