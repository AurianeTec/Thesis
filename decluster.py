if __name__ == '__main__':
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
    import igraph as ig
    import ast
    import itertools
    import concurrent.futures


    ### CREATE NETWORK ###
    #--- Import cusotm function
    def make_attr_dict(*args, **kwargs): 
        argCount = len(kwargs)
        
        if argCount > 0:
            attributes = {}
            for kwarg in kwargs:
                attributes[kwarg] = kwargs.get(kwarg, None)
            return attributes
        else:
            return None # (if no attributes are given)
    
    #--- Create the network in NetworkX and move to igraph
    # Retrieve edges
    edges_with_id = pd.read_csv('data/clean/initial_network_edges.csv')
    edges_with_id["geometry"] = edges_with_id.apply(lambda x: shapely.wkt.loads(x.geometry), axis = 1)
    edges_with_id = gpd.GeoDataFrame(edges_with_id, geometry = 'geometry', crs = 4326).to_crs(2154)
    edges_with_id = edges_with_id.rename(columns={"id": "G"})

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
                                                                    num_schools = x.school_count,
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

    # eids: "conversion table" for edge ids from igraph to nx 
    eids_nx = [g_igraph.es[i]["G"] for i in range(len(g_igraph.es))]
    eids_ig = [i for i in range(len(g_igraph.es))]
    eids_conv = pd.DataFrame({"nx": eids_nx, "ig": eids_ig})

    # nids: "conversion table" for node ids from igraph to nx
    nids_nx = [g_igraph.vs(i)["_nx_name"][0] for i in range(len(g_igraph.vs))]
    nids_ig = [i for i in range(len(g_igraph.vs))]
    nids_conv = pd.DataFrame({"nx": nids_nx, "ig": nids_ig})
    nids_conv['nx'] = nids_conv['nx'].astype(int)

    # combine the conversion table with nodes_carbike_centroids_RER_complete
    nodes_carbike_centroids_RER_complete = nodes_carbike_centroids_RER_complete.merge(nids_conv, left_on = "osmid", right_on = "nx", how = "left")
    nodes_carbike_centroids_RER_complete = nodes_carbike_centroids_RER_complete.drop(columns = ["nx"])
    seq = g_igraph.vs.select(centroid_eq = True)
    centroids = [v.index for v in seq]
        



    ### DECLUSTER ###
    #--- Get data
    combined_df_schools = pd.read_csv('data/results/gaps_with_benefit_schools_all.csv')
    combined_df_schools['path'] = combined_df_schools['path'].apply(lambda x: ast.literal_eval(str(x)))
    schools_base = pd.read_csv('data/results/edge_betweenness_baseline_school.csv').rename(columns = {"Edge": "edge_ig", "Betweenness Centrality" : "ebc"})
    schools_0_5 = pd.read_csv('data/results/edge_betweenness_school_delta05.csv').rename(columns = {"Edge": "edge_ig", "Betweenness Centrality" : "ebc"})
    schools_1 = pd.read_csv('data/results/edge_betweenness_school_delta1.csv').rename(columns = {"Edge": "edge_ig", "Betweenness Centrality" : "ebc"})
    schools_1_5 = pd.read_csv('data/results/edge_betweenness_school_delta1.5.csv').rename(columns = {"Edge": "edge_ig", "Betweenness Centrality" : "ebc"})
    ced = nx.get_edge_attributes(G, "geometry") # coordinates of edges dictionary ced
    ced = {tuple(sorted(key)): value for key, value in ced.items()}


    # Create pairs for declustering
    data_frames = ['schools_base', 'schools_0_5', 'schools_1', 'schools_1_5']
    data_trouples = [(schools_base, combined_df_schools.sort_values(by=['schools_base_B'], ascending=False).head(5000), 'schools_base'),
                (schools_0_5, combined_df_schools.sort_values(by=['schools_0_5_B'], ascending=False).head(5000), 'schools_0_5'),
                (schools_1, combined_df_schools.sort_values(by=['schools_1_B'], ascending=False).head(5000), 'schools_1'),
                (schools_1_5, combined_df_schools.sort_values(by=['schools_1_5_B'], ascending=False).head(5000), 'schools_1_5')]


    def process_trouple(trouple):
        ebc, mygaps, name = trouple

        # ADDING EBC VALUES / RANKING METRICS TO EACH EDGE in network "f" (which will be used for declustering)
        f = g_igraph.copy()
        for i in range(len(ebc)):
            f.es[i]["ebc"] = ebc.loc[i, "ebc"]

        print('make a subgraph of f that contains only overlapping gaps')
        my_edges = list(set([item for sublist in mygaps["path"] for item in sublist]))
        c = f.copy()
        c = c.subgraph_edges(edges = my_edges, 
                                delete_vertices=True)

        cl = c.decompose() ### cl contains disconnected components 

        gapranks = []
        gapcoords = []
        gapedgeids = []

        for comp in range(len(cl)):
            print(len(cl))
            mc = cl[comp].copy() # mc: my component (current)

            #### decluster component:

            while len(mc.es) > 0:
                print(len(mc.es))
                nodestack = [node.index for node in mc.vs() if mc.vs[node.index]["nodetype"]=="both" and mc.degree(node.index)!=2]
                nodecomb = [comb for comb in itertools.combinations(nodestack, 2)] ### all possible OD combis on that cluster
                sp = [] ### list of shortest paths between d1 multi nodes on that cluster:
                for mycomb in nodecomb:
                    gsp = mc.get_shortest_paths(mycomb[0], 
                                                        mycomb[1],
                                                        weights = "length", 
                                                        mode = "out", 
                                                        output = "epath")[0]
                    if gsp:
                        sp.append(gsp)
                    

                ### compute metrics for each path:
                if not sp:
                    break

                lens = []
                cycs = []
                
                for p in sp:
                    lens += [np.sum([mc.es[e]["length"] for e in p])]
                    cycs += [np.sum([mc.es[e]["length"]*mc.es[e]["ebc"] for e in p])]
                    
                norms = list(np.array(cycs)/np.array(lens))
                maxpath = sp[norms.index(max(norms))]
                gapranks.append(np.round(max(norms), 2))
                
                gapcoord_current = []
                edgeids_current = []
                
                
                for e in maxpath:
                    edge_id = sorted(ast.literal_eval(mc.es[e]["edge_id"]))
                    for i , id in enumerate(edge_id):
                        edge_id[i] = ast.literal_eval(id)
                    edge_coords = [(c[1], c[0]) for c in ced[tuple(sorted(edge_id))].coords]
                    gapcoord_current.append(edge_coords)
                    edgeids_current.append(edge_id)
                    
                mc.delete_edges(maxpath)
                gapcoords.append(gapcoord_current)
                gapedgeids.append(edgeids_current)


        gap_dec = pd.DataFrame({"rank": gapranks, "coord": gapcoords, "id": gapedgeids})
        gap_dec = gap_dec.sort_values(by = "rank", ascending = False).reset_index(drop = True)
        gap_dec.to_csv( "./analysis/TEST_gaps_declustered_schools_"+str(name)+".csv")

    # Number of worker threads to use
    num_workers = len(data_frames) 

    # Run the processing in parallel

    with mp.Pool(processes=num_workers) as pool:
        pool.map(process_trouple, data_trouples)
