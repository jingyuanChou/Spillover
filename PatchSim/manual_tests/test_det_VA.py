import sys
import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import torch_scatter
from torch_sparse import SparseTensor
import torch
sys.path.append('../')
import patchsim as sim
def connection_probability(distance, scale=0.1):
    """Calculate connection probability based on distance."""
    return np.exp(-distance / scale)

configs = sim.read_config('cfg_VA_2023.txt')

# Parameters
n_nodes = 133
dimension = 20
mu = 1.0

# Step 1: Generate confounders for each node
covariance_matrix = mu * np.identity(dimension)
confounders = np.random.multivariate_normal(np.zeros(dimension), covariance_matrix, size=n_nodes)

# Step 2: Calculate pairwise Euclidean distances between confounder vectors
distances = squareform(pdist(confounders, 'euclidean'))
normalized_distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))

# Initialize graph
G = nx.Graph()
G.add_nodes_from(range(n_nodes))

# Step 3: Build the graph
for i in range(n_nodes):
    connections = []
    for j in range(n_nodes):
        if i != j:
            # Use a probabilistic function of distance to decide on connections
            prob = connection_probability(normalized_distances[i, j])
            if np.random.rand() < prob:
                connections.append((i, j, 1 - normalized_distances[i, j]))  # Weight is 1 - distance for similarity

    # Ensure 1-8 edges per node, randomly select if more than 8
    if len(connections) > 8:
        connections = list(np.random.choice(connections, size=8, replace=False))
    elif len(connections) == 0:  # Ensure at least 1 connection
        j = np.random.choice(list(set(range(n_nodes)) - {i}))
        connections.append((i, j, 1 - normalized_distances[i, j]))

    # Add edges to the graph
    G.add_weighted_edges_from(connections)

# Step 4: Create adjacency matrix from the graph
adj_matrix = nx.to_numpy_array(G, weight='weight')

adj_matrix[adj_matrix!=0] = 1


def get_edge_homophily(y, edge_index, edge_weight=None):
    """
    Return the weighted edge homophily, according to the weights in the provided adjacency matrix.
    """
    src, dst, edge_weight = get_weighted_edges(edge_index, edge_weight)

    return ((y[src] == y[dst]).float().squeeze() * edge_weight).sum() / edge_weight.sum()


def get_node_homophily(y, edge_index, edge_weight=None):
    """
    Return the weighted node homophily, according to the weights in the provided adjacency matrix.
    """
    src, dst, edge_weight = get_weighted_edges(edge_index, edge_weight)

    index = src
    mask = (y[src] == y[dst]).float().squeeze() * edge_weight
    per_node_masked_sum = torch_scatter.scatter_sum(mask, index)
    per_node_total_sum = torch_scatter.scatter_sum(edge_weight, index)

    non_zero_mask = per_node_total_sum != 0
    return (per_node_masked_sum[non_zero_mask] / per_node_total_sum[non_zero_mask]).mean()
def get_weighted_edges(edge_index, edge_weight=None):
    """
    Return (src, dst, edge_weight) tuple.
    """
    if isinstance(edge_index, SparseTensor):
        src, dst, edge_weight = edge_index.coo()
    else:
        src, dst = edge_index
        edge_weight = (
            edge_weight if edge_weight is not None else torch.ones((edge_index.size(1),), device=edge_index.device)
        )

    return src, dst, edge_weight



sim.run_disease_simulation(configs,write_epi=True,vaxs=None)


# suppose given a vector of treatments, run a simulation and take records
# for each county, we find the connected counties, then check if the treatments have '1', if so, flip '1' to '0' and
# rerun simulation, we take records of the outcome of itself. , and find the treatment outcome.

