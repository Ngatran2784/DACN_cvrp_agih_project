import time

from src.baselines.clarke_wright import solve_clarke_wright
from src.paper_method.rl_attention_greedy import solve_paper_attention_greedy
from src.heuristic.utils import compute_distance_matrix, solution_distance
from src.heuristic.cheapest_insertion import (
    get_insertion_candidates,
    apply_insertion,
)
from src.heuristic.two_opt import two_opt_solution
from src.heuristic.relocate import relocate_solution


def choose_customer_by_regret(unserved, routes, demands, capacity, dist):
    best_choice = None

    for customer in unserved:
        candidates = get_insertion_candidates(
            routes, customer, demands, capacity, dist
        )

        if not candidates:
            continue

        candidates = sorted(candidates, key=lambda x: x["delta"])

        best_candidate = candidates[0]
        best_delta = candidates[0]["delta"]

        if len(candidates) >= 2:
            second_delta = candidates[1]["delta"]
            regret = second_delta - best_delta
        else:
            regret = 0.0

        # Ưu tiên regret lớn, sau đó delta nhỏ
        score = (regret, -best_delta)

        if best_choice is None or score > best_choice["score"]:
            best_choice = {
                "customer": customer,
                "candidate": best_candidate,
                "score": score,
            }

    return best_choice


def solve_regret_insertion(coords, demands, capacity):
    dist = compute_distance_matrix(coords)

    n_customers = len(coords) - 1
    unserved = set(range(1, n_customers + 1))
    routes = []

    while unserved:
        choice = choose_customer_by_regret(
            unserved, routes, demands, capacity, dist
        )

        if choice is None:
            break

        candidate = choice["candidate"]
        customer = choice["customer"]

        routes = apply_insertion(routes, candidate)
        unserved.remove(customer)

    return routes


def improve_routes(routes, demands, capacity, dist):
    routes = two_opt_solution(routes, dist)
    routes = relocate_solution(routes, demands, capacity, dist, max_iter=50)
    routes = two_opt_solution(routes, dist)
    return routes


def solve_proposed_insertion_2opt(coords, demands, capacity):
    """
    Proposed Hybrid AGIH-2opt.

    Ý tưởng:
    - Tạo nhiều nghiệm khởi tạo:
        1. Clarke-Wright
        2. Paper-Attention-Greedy
        3. Regret Insertion
    - Chọn nghiệm tốt nhất.
    - Cải thiện bằng 2-opt + relocate.
    """
    start = time.time()
    dist = compute_distance_matrix(coords)

    candidate_solutions = []

    # 1. Clarke-Wright seed
    cw_routes, _, _ = solve_clarke_wright(coords, demands, capacity)
    candidate_solutions.append(("CW seed", cw_routes))

    # 2. Paper-Attention seed
    try:
        paper_routes, _, _ = solve_paper_attention_greedy(coords, demands, capacity)
        candidate_solutions.append(("Paper seed", paper_routes))
    except Exception:
        pass

    # 3. Regret Insertion seed
    regret_routes = solve_regret_insertion(coords, demands, capacity)
    candidate_solutions.append(("Regret seed", regret_routes))

    # Chọn seed tốt nhất
    best_name = None
    best_routes = None
    best_cost = float("inf")

    for name, routes in candidate_solutions:
        cost = solution_distance(routes, dist)
        if cost < best_cost:
            best_cost = cost
            best_routes = routes
            best_name = name

    # Local search cải thiện
    improved_routes = improve_routes(best_routes, demands, capacity, dist)
    improved_cost = solution_distance(improved_routes, dist)

    runtime = time.time() - start

    return improved_routes, float(improved_cost), runtime