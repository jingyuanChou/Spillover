import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform

def connection_probability(distance, scale=0.1):
    """Calculate connection probability based on distance."""
    return np.exp(-distance / scale)

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

# Step 5: Normalize the adjacency matrix
normalized_adj_matrix = np.divide(adj_matrix, adj_matrix.sum(axis=1, keepdims=True), out=np.zeros_like(adj_matrix), where=adj_matrix.sum(axis=1, keepdims=True)!=0)

import matplotlib.pyplot as plt
nx.draw_networkx(G, node_size=20, with_labels=False, pos=nx.spring_layout(G, seed=42))
plt.show()
print('2')