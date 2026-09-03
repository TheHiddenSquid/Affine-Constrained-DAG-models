import math
import random

import networkx as nx
import numpy as np


# DAG generation
def get_random_DAG(num_nodes, edge_prob = 0.5):
    # Generate erdős-rényi under-triangular matrix
    edge_array = np.zeros((num_nodes, num_nodes), dtype=np.int64)
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i > j:
                if edge_prob > random.random():
                    edge_array[i,j] = 1      
    
    # Shuffle nodes
    G = nx.DiGraph(edge_array)
    node_mapping = dict(zip(G.nodes(), sorted(G.nodes(), key=lambda k: random.random())))
    G_new = nx.relabel_nodes(G, node_mapping)
    edge_array = nx.adjacency_matrix(G_new, node_mapping).todense()

    return edge_array

def generate_DAG_with_params(num_nodes, edge_prob = 0.5):
    # Generate DAG and lambda matrix
    adjacency_matrix = get_random_DAG(num_nodes, edge_prob)
    lambda_matrix = generate_lambda_matrix(adjacency_matrix, num_nodes)

    # Generate partition and omega matrix
    omega_matrix = generate_omega_matrix(num_nodes)

    return adjacency_matrix, lambda_matrix, omega_matrix


def generate_sample(size, lambda_matrix, omega_matrix):
    no_nodes = len(omega_matrix)

    # Generate all errors at once
    errors = np.random.normal(
        loc=0.0,
        scale=np.sqrt(omega_matrix)[:, None],
        size=(no_nodes, size)
    )

    X = np.linalg.inv(np.eye(no_nodes) - lambda_matrix).T

    # Matrix multiplication on all samples simultaneously
    sample = X @ errors

    return sample



def generate_omega_matrix(num_nodes):
    omega_matrix = [random.uniform(0.2,2) for _ in range(num_nodes)]
    return omega_matrix

def generate_lambda_matrix(adjacency_matrix, num_nodes):
    lambda_matrix = adjacency_matrix.astype(np.float64)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if lambda_matrix[i,j] == 1:
                lambda_matrix[i,j] = random.uniform(0.25,1) * random.randrange(-1,2,2)

    return lambda_matrix

# Graph functions

def is_DAG(A):
    """Return whether an adjacency matrix represents a directed acyclic graph."""

    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False

    indegree = np.count_nonzero(A, axis=0)
    ready = list(np.flatnonzero(indegree == 0))
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for child in np.flatnonzero(A[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    return visited == A.shape[0]

def get_log_lik(X, Lam, Omega):
    p, n = X.shape
    S = X @ X.T / n

    tmp = np.eye(p) - Lam
    inv_omega = np.diag([1/x for x in Omega])

    likelihood = n/2 * (-math.log(np.prod(Omega))+2*math.log(np.linalg.det(tmp))-np.trace(tmp @ inv_omega @ tmp.T @ S))
    return likelihood
