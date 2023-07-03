#!/usr/bin/env python
# coding: utf-8

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

if __name__ == '__main__':
    crs_fr = 2154
    def make_attr_dict(*args, **kwargs): 
        
        argCount = len(kwargs)
        
        if argCount > 0:
            attributes = {}
            for kwarg in kwargs:
                attributes[kwarg] = kwargs.get(kwarg, None)
            return attributes
        else:
            return None # (if no attributes are given)



    def equalization_all(od, variable, colname, delta, centroids): #For equalisation matrices (Jin)
        
        od_ = od.copy()
        variable_ = variable.copy()
        
        variable_average = np.mean(variable_[colname]) 
        
        variable_['weight'] = variable_[colname].apply(lambda x: (x/variable_average)**-delta)

        i =0
        for val in variable_['ig']:
            weight = variable_.loc[variable_['ig']==val]['weight'].iloc[0]
            try:
                od_[centroids.index(val)] *= weight 
                od_.loc[centroids.index(val)] *= weight 
            except:
                continue
    #             print(val, ' not found')
            i +=1
        
        return od_



    #--- Create the network in NetworkX
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
                                                                    pop_dens = x.pop_density,
                                                                    active_pop_density = x.active_pop_density,
                                                                    school_pop_density = x.school_pop_density,
                                                                    school_count = x.school_count,
                                                                    num_jobs = x.num_jobs,
                                                                    ),
                                                                    axis = 1) 

    # Create Graph with all nodes and edges
    G = nx.from_pandas_edgelist(edges_with_id, source='x', target='y', edge_attr=True)
    G.add_nodes_from(nodes_carbike_centroids_RER_complete.loc[:,["osmid", "attr_dict"]].itertuples(index = False))


    #--- Moving from NetworkX to igraph
    g_igraph = ig.Graph()
    networkx_graph = G
    g_igraph = ig.Graph.from_networkx(networkx_graph)

    # # eids: "conversion table" for edge ids from igraph to nx 
    # eids_nx = [tuple(sorted(literal_eval(g_igraph.es(i)["edge_id"][0]))) for i in range(len(g_igraph.es))]
    # eids_ig = [i for i in range(len(g_igraph.es))]
    # eids_conv = pd.DataFrame({"nx": eids_nx, "ig": eids_ig})

    # # nids: "conversion table" for node ids from igraph to nx
    # nids_nx = [g_igraph.vs(i)["_nx_name"][0] for i in range(len(g_igraph.vs))]
    # nids_ig = [i for i in range(len(g_igraph.vs))]
    # nids_conv = pd.DataFrame({"nx": nids_nx, "ig": nids_ig})

    # nids_conv['nx'] = nids_conv['nx'].astype(int)

    # combine the conversion table with nodes_carbike_centroids_RER_complete
    nodes_carbike_centroids_RER_complete = nodes_carbike_centroids_RER_complete.merge(nids_conv, left_on = "osmid", right_on = "nx", how = "left")
    nodes_carbike_centroids_RER_complete = nodes_carbike_centroids_RER_complete.drop(columns = ["nx"])



    # Isolate centroids
    from itertools import combinations
    seq = g_igraph.vs.select(centroid_eq = True)
    centroids = [v.index for v in seq]
    centroids = centroids[0:3]

    node_combinations = list(combinations(centroids, 2))

    # Create OD matrix
    def process_node(args):
        start_node, end_node = args
        global g_igraph
        shortest_path_length = g_igraph.shortest_paths_dijkstra(source=start_node, target=end_node, weights='weight')[0][0]
        return (start_node, end_node, shortest_path_length)


    # Number of processes (cores) to use for parallel processing
    num_processes = 4

    # Create a pool of processes
    pool = mp.Pool(processes=num_processes)

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

