#!/usr/bin/env python
# coding: utf-8

if __name__ == '__main__':
    import timeit
    start = timeit.default_timer()
    import pandas as pd
    import os
    os.environ['USE_PYGEOS'] = '0'
    import geopandas as gpd
    import matplotlib.pyplot as plt
    pd.options.mode.chained_assignment = None  # default='warn'
    import networkx as nx
    import shapely
    import multiprocess as mp
    import numpy as np
    import math
    import igraph as ig
    crs_fr = 2154
    print('imports done')


    #--- Custom function (Anastassia)
    # get_ipython().run_line_magic('run', '-i packages.py')
    def make_attr_dict(*args, **kwargs): 
        
        argCount = len(kwargs)
        
        if argCount > 0:
            attributes = {}
            for kwarg in kwargs:
                attributes[kwarg] = kwargs.get(kwarg, None)
            return attributes
        else:
            return None # (if no attributes are given)

    print('functions imported')

    #--- Shapes

    # GPM outline
    GPM = gpd.read_file('data/raw/GPM.geojson').to_crs(crs_fr)

    # IRIS codes and shapes 
    IRIS_GPM = gpd.read_file('data/raw/IRIS_GPM.geojson')

    print('shapes imported')
    ## Creating the network and adding igraph IDs to the node table

    #--- Create the network
    # Retrieve edges
    edges_with_id = pd.read_csv('data/clean/initial_network_edges_complete.csv')
    edges_with_id["geometry"] = edges_with_id.apply(lambda x: shapely.wkt.loads(x.geometry), axis = 1)
    edges_with_id = gpd.GeoDataFrame(edges_with_id, geometry = 'geometry', crs = 4326).to_crs(2154)

    # Retrieve nodes
    nodes_carbike_centroids_RER_complete = pd.read_csv('data/clean/initial_network_nodes_complete.csv')
    nodes_carbike_centroids_RER_complete["geometry"] = nodes_carbike_centroids_RER_complete.apply(lambda x: shapely.wkt.loads(x.geometry), axis = 1)
    nodes_carbike_centroids_RER_complete = gpd.GeoDataFrame(nodes_carbike_centroids_RER_complete, geometry = 'geometry', crs = 2154)

    # Create the attr_dict
    nodes_carbike_centroids_RER_complete["attr_dict"] = nodes_carbike_centroids_RER_complete.apply(lambda x: make_attr_dict(
                                                                    nodetype = x.nodetype,
                                                                    centroid = x.centroid,
                                                                    RER = x.RER,
                                                                    IRIS = x.CODE_IRIS,
                                                                    #   pop_dens = x.pop_density
                                                                    ),
                                                                    axis = 1) 

    print('about to created network')
    #--- Create Graph with all nodes and edges
    G = nx.from_pandas_edgelist(edges_with_id, source='x', target='y', edge_attr=True)
    G.add_nodes_from(nodes_carbike_centroids_RER_complete.loc[:,["osmid", "attr_dict"]].itertuples(index = False))
    print('networkX network active')
    #--- Moving from NetworkX to igraph
    g_igraph = ig.Graph()
    networkx_graph = G
    g_igraph = ig.Graph.from_networkx(networkx_graph)
    print('igraph network active')
    ## Basic OD Matrix: shortest path between each pair of centroids

    # Isolate centroids

    seq = g_igraph.vs.select(centroid_eq = True)
    centroids = [v.index for v in seq]
    centroids = centroids[0:100]
    results = []
    print('centroids isolated')
    # Create OD matrix

    def remove_duplicates(lst):
        unique_lst = []
        for sublist in lst:
            if sorted(sublist) not in [sorted(unique_sublist) for unique_sublist in unique_lst]:
                unique_lst.append(sublist)
        return unique_lst

    def process_node(args):
        start_node, end_node = args
        shortest_path_length = g_igraph.shortest_paths_dijkstra(source=start_node, target=end_node, weights='weight')[0][0]
        return (start_node, end_node, shortest_path_length)

# Number of processes (cores) to use for parallel processing
    num_processes = 4

    # Create a pool of processes
    pool = mp.Pool(processes=num_processes)

    # Generate combinations of nodes for processing
    node_combinations = [(start_node, end_node) for start_node in centroids for end_node in centroids if start_node != end_node]
    node_combinations = remove_duplicates(node_combinations)

    # Apply the function to each node combination using parallel processing
    results = pool.map(process_node, node_combinations)

    # Create a dictionary to store the shortest path lengths
    output = {}
    for start_node, end_node, shortest_path_length in results:
        if start_node not in output:
            output[start_node] = {}
        output[start_node][end_node] = shortest_path_length

    # Create an empty adjacency matrix
    matrix = np.zeros((len(centroids), len(centroids)))

    # Fill the adjacency matrix with shortest path lengths
    for i, start_node in enumerate(centroids):
        for j, end_node in enumerate(centroids):
            if start_node in output and end_node in output[start_node]:
                matrix[i, j] = output[start_node][end_node]
                matrix[j, i] = output[start_node][end_node]

    # Close the pool
    pool.close()
    pool.join()

    print(matrix.shape)

    stop = timeit.default_timer()

    print('Time: ', stop - start)  
