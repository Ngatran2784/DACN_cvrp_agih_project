import pickle

from src.baselines.clarke_wright import solve_clarke_wright
from src.paper_method.rl_attention_greedy import solve_paper_attention_greedy
from src.heuristic.proposed_insertion import solve_proposed_insertion_2opt
from src.heuristic.utils import route_load, compute_distance_matrix, route_distance


def print_routes(method_name, routes, demands, coords, total_cost, runtime):
    dist = compute_distance_matrix(coords)

    print(f"\n===== {method_name} =====")
    print("Total distance:", round(total_cost, 4))
    print("Vehicles:", len(routes))
    print("Runtime:", round(runtime, 4), "s")

    for idx, route in enumerate(routes, start=1):
        load = route_load(route, demands)
        cost = route_distance(route, dist)
        route_str = " -> ".join(map(str, route))
        print(f"Route {idx}: {route_str} | Load = {load} | Distance = {cost:.4f}")


def main():
    with open("data/cvrp20_test.pkl", "rb") as f:
        instances = pickle.load(f)

    inst = instances[0]

    coords = inst["coords"]
    demands = inst["demands"]
    capacity = inst["capacity"]

    print("Instance:", inst["id"])
    print("Capacity:", capacity)

    cw_routes, cw_cost, cw_runtime = solve_clarke_wright(
        coords, demands, capacity
    )

    paper_routes, paper_cost, paper_runtime = solve_paper_attention_greedy(
        coords, demands, capacity
    )

    prop_routes, prop_cost, prop_runtime = solve_proposed_insertion_2opt(
        coords, demands, capacity
    )

    print_routes(
        "Old Algorithm: Clarke-Wright",
        cw_routes,
        demands,
        coords,
        cw_cost,
        cw_runtime,
    )

    print_routes(
        "Paper Algorithm: Attention-Greedy",
        paper_routes,
        demands,
        coords,
        paper_cost,
        paper_runtime,
    )

    print_routes(
        "Proposed Algorithm: AGIH-2opt",
        prop_routes,
        demands,
        coords,
        prop_cost,
        prop_runtime,
    )


if __name__ == "__main__":
    main()