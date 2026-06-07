import math
import numpy as np


def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def compute_distance_matrix(coords):
    n = len(coords)
    dist = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            dist[i, j] = euclidean_distance(coords[i], coords[j])

    return dist


def route_distance(route, dist_matrix):
    return sum(dist_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))


def solution_distance(routes, dist_matrix):
    return sum(route_distance(route, dist_matrix) for route in routes)


def route_load(route, demands):
    return sum(demands[i] for i in route if i != 0)


def check_capacity(routes, demands, capacity):
    return all(route_load(route, demands) <= capacity for route in routes)


def flatten_routes(routes):
    seq = []
    for route in routes:
        for node in route:
            if node != 0:
                seq.append(node)
    return seq