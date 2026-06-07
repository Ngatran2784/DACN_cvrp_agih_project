import time
import numpy as np

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from src.heuristic.utils import compute_distance_matrix


def solve_cvrp_ortools(coords, demands, capacity, time_limit=1):
    """
    Solve CVRP using Google OR-Tools.

    Input:
        coords: depot + customer coordinates
        demands: demand of each node
        capacity: vehicle capacity
        time_limit: solving time in seconds

    Output:
        routes: list of routes
        total_distance: total route distance
        runtime: running time
    """
    start = time.time()

    n_nodes = len(coords)
    dist_float = compute_distance_matrix(coords)

    # OR-Tools dùng số nguyên, nên scale khoảng cách lên
    distance_matrix = (dist_float * 10000).astype(int)

    depot = 0

    # Cho dư số xe, solver sẽ chỉ dùng xe cần thiết
    num_vehicles = n_nodes

    manager = pywrapcp.RoutingIndexManager(n_nodes, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return int(demands[from_node])

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        [int(capacity)] * num_vehicles,
        True,
        "Capacity",
    )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )

    search_parameters.time_limit.seconds = int(time_limit)

    solution = routing.SolveWithParameters(search_parameters)

    routes = []

    if solution is None:
        runtime = time.time() - start
        return routes, float("inf"), runtime

    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route = []

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            index = solution.Value(routing.NextVar(index))

        route.append(manager.IndexToNode(index))

        # Bỏ route rỗng dạng 0 -> 0
        if len(route) > 2:
            routes.append(route)

    total_distance = 0.0
    for route in routes:
        for i in range(len(route) - 1):
            total_distance += dist_float[route[i], route[i + 1]]

    runtime = time.time() - start

    return routes, float(total_distance), runtime