import networkx as nx
import numpy as np

def community_score(A, obs_frame):
    G = nx.from_scipy_sparse_array(A=A, create_using=nx.DiGraph)
    communities = []
    for v in obs_frame.unique():
        communities.append(set(np.where(obs_frame == v)[0]))

    return nx.community.modularity(G, communities)

def community_score_star(input):
    return community_score(*input)