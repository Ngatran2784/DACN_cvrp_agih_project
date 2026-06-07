import time
import json

from src.baselines.clarke_wright import solve_clarke_wright
from src.heuristic.utils import compute_distance_matrix, solution_distance
from src.heuristic.two_opt import two_opt_solution
from src.heuristic.relocate import relocate_solution


def improve_routes(routes, demands, capacity, dist):
    best_routes = [r[:] for r in routes]
    best_cost = solution_distance(best_routes, dist)

    # 2-opt
    candidate = two_opt_solution(best_routes, dist)
    candidate_cost = solution_distance(candidate, dist)

    if candidate_cost < best_cost:
        best_routes = candidate
        best_cost = candidate_cost

    # relocate
    candidate = relocate_solution(best_routes, demands, capacity, dist, max_iter=50)
    candidate_cost = solution_distance(candidate, dist)

    if candidate_cost < best_cost:
        best_routes = candidate
        best_cost = candidate_cost

    # 2-opt lại lần cuối
    candidate = two_opt_solution(best_routes, dist)
    candidate_cost = solution_distance(candidate, dist)

    if candidate_cost < best_cost:
        best_routes = candidate
        best_cost = candidate_cost

    return best_routes


def solve_proposed_from_paper(coords, demands, capacity, paper_routes):
    """
    Proposed AGIH-2opt.

    Hybrid seed:
    - Clarke-Wright seed
    - Paper-RL-Attention seed

    Chọn seed tốt nhất rồi cải thiện bằng:
    - 2-opt
    - relocate

    Thuật toán đảm bảo không trả nghiệm xấu hơn seed tốt nhất.
    """
    start = time.time()
    dist = compute_distance_matrix(coords)

    candidate_solutions = []

    # Seed 1: Clarke-Wright
    cw_routes, cw_cost, _ = solve_clarke_wright(coords, demands, capacity)
    candidate_solutions.append(("Clarke-Wright seed", cw_routes, cw_cost))

    # Seed 2: Paper-RL-Attention
    paper_cost = solution_distance(paper_routes, dist)
    candidate_solutions.append(("Paper-RL-Attention seed", paper_routes, paper_cost))

    # Chọn seed tốt nhất
    best_name, best_routes, best_cost = min(
        candidate_solutions,
        key=lambda x: x[2]
    )

    # Cải thiện seed tốt nhất
    improved_routes = improve_routes(best_routes, demands, capacity, dist)
    improved_cost = solution_distance(improved_routes, dist)

    # Nếu local search không cải thiện thì giữ seed tốt nhất
    if improved_cost > best_cost:
        final_routes = best_routes
        final_cost = best_cost
    else:
        final_routes = improved_routes
        final_cost = improved_cost

    runtime = time.time() - start

    return final_routes, float(final_cost), runtime
def solve_proposed_from_paper_routes(coords, demands, capacity, paper_routes):
    """
    Alias function để run_compare.py gọi đúng tên.
    """
    return solve_proposed_from_paper(coords, demands, capacity, paper_routes)