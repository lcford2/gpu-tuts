from time import perf_counter as timer

import cugraph as cnx
import networkx as nx
import pandas as pd

# cuGraph aims to mimic the NetworkX API
# and functionality put perform the computation
# on GPUs


def nx_example(g):
    s = timer()
    nx.betweenness_centrality(g)
    e = timer()
    return e - s


def cugraph_example(g):
    s = timer()
    cnx.betweenness_centrality(g)
    e = timer()
    return e - s


if __name__ == "__main__":
    output = []
    for nodes in range(100, 1001, 100):
        for edges in range(2, 11, 2):
            G = nx.barabasi_albert_graph(nodes, edges)
            nx_time = nx_example(G)
            cu_time = cugraph_example(G)
            output.append([nodes, edges, nx_time / cu_time])
    df = pd.DataFrame.from_records(
        output, columns=["Nodes", "Edges per Node", "cuGraph Speedup"]
    )
    print(df.pivot(index="Nodes", columns="Edges per Node"))
