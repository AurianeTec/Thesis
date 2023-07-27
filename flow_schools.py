if __name__ == '__main__':
    import pandas as pd
    import os
    os.environ['USE_PYGEOS'] = '0'
    import geopandas as gpd
    pd.options.mode.chained_assignment = None  # default='warn'
    import networkx as nx
    import shapely
    import multiprocess as mp
    import numpy as np
    import igraph as ig
    from ta_lab.assignment.assign import frank_wolfe
    from ta_lab.assignment.line import *
    from ta_lab.assignment.graph import *
    from ta_lab.assignment.shortest_path import ShortestPath as SPP
    from ast import literal_eval

    crs_fr = 2154

    #### FUNCTION IMPORTS ####
    #--- Custom function (Anastassia)
    # Create a dictionary of attributes (useful for networkX)
    def make_attr_dict(*args, **kwargs): 
        
        argCount = len(kwargs)
        
        if argCount > 0:
            attributes = {}
            for kwarg in kwargs:
                attributes[kwarg] = kwargs.get(kwarg, None)
            return attributes
        else:
            return None # (if no attributes are given)


    #--- Custom function (adapted from Jin)
    # equalize an OD for DIFFFERENT attributes in O and D
    # multiply baseline with the number of opportunities in the destination and with the attribute of i over the avg **-delta

    def equalization_all_2attributes(od, variable, colnameO, delta, centroids): 
        
        od_ = od.copy()
        variable_ = variable.copy()
        
        variable_average1 = np.mean(variable_[colnameO])
        
        # calculate the attribute of i over the avg and **-delta 
        variable_['weightO'] = variable_[colnameO].apply(lambda x: (x / variable_average1) ** -delta)

        i = 0
        for val in variable_['ig']:
            weightO = variable_.loc[variable_['ig'] == val]['weightO'].iloc[0]
            try:
                od_.loc[centroids.index(val)] *= weightO #row = origin
            except:
                continue
            i += 1
        
        return od_

    #--- Custom function to use the function above in a batch
    def equalization_with_2attributes(nodes_carbike_centroids_RER_complete, baseline_df, centroids, COLOFINTEREST1, delta):
        col_tokeep = ['osmid', 'ig', 'CODE_IRIS', COLOFINTEREST1]
        COLSOFINTEREST_df = nodes_carbike_centroids_RER_complete.loc[nodes_carbike_centroids_RER_complete['centroid'] == True].copy()
        COLSOFINTEREST_df = COLSOFINTEREST_df[col_tokeep]
        
        equalized_od = equalization_all_2attributes(baseline_df, COLSOFINTEREST_df, COLOFINTEREST1, delta, centroids)
        
        equalized_od_name = "OD_equalization_" + COLOFINTEREST1 + "_O_delta_" + str(delta)
        
        return {equalized_od_name: equalized_od}

    #### CREATE NETWORK IN NETWORKX AND IGRAPH ####

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
    eids_nx = [tuple(sorted(literal_eval(g_igraph.es(i)["edge_id"][0]))) for i in range(len(g_igraph.es))]
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

    # Isolate centroids
    from itertools import combinations
    seq = g_igraph.vs.select(centroid_eq = True)
    centroids = [v.index for v in seq]
    # centroids = centroids[0:3] #TODO
    node_combinations = list(combinations(centroids, 2))

    #### CALCULATE BASELINE OD MATRIX ####
    #--- Shortest path length 
    matrix = pd.read_csv('data/clean/shortest_paths.csv').drop(columns=['Unnamed: 0'])
    matrix = matrix.to_numpy()

    #--- School population density in i, deterrence factor, num schools in j 
    num_processes = mp.cpu_count()

    def process_node(args):
        global matrix
        o, d = args
        if o == d:
            return (o, d, 0)
        else:
            normalized_dist = matrix[o][d] / matrix.max()
            demand = (
                (g_igraph.vs[centroids[o]]['school_pop_density'])
                * (g_igraph.vs[centroids[d]]['num_schools'])
                * dist_decay * np.exp(-1 * normalized_dist)
            )
            return (o, d, demand)

    baseline = np.zeros((len(centroids), len(centroids)))
    maxtrips = 100
    dist_decay = 1

    pool = mp.Pool(processes=num_processes)

    # Create node combinations
    node_combinations = [(o, d) for o in range(len(centroids)) for d in range(len(centroids))]
    
    # Calculate demand for each node combination using multiprocessing
    results = pool.map(process_node, node_combinations)

    # Update baseline matrix with calculated demand
    for o, d, demand in results:
        baseline[o][d] = demand

    # Normalize the matrix to the number of maxtrips
    baseline = ((baseline / baseline.max()) * maxtrips)

    # Round up to ensure each journey is made at least once
    baseline = np.ceil(baseline).astype(int)
    school_pop_dens_school_count_noEQ = pd.DataFrame(baseline)

    pool.close()
    pool.join()

    
    #### CALCULATE OD MATRICES FOR EQ ####
    #--- Schools/education level/school pop density
    def process_combination(COLOFINTEREST1):
        delta_list = [0.5, 1, 1.5]
        results = {}
        for delta in delta_list:
            result = equalization_with_2attributes(nodes_carbike_centroids_RER_complete, school_pop_dens_school_count_noEQ, centroids, COLOFINTEREST1, delta)
            results.update(result)
        return results

    COLOFINTEREST1 = ['edu_level']
    pool = mp.Pool(processes=num_processes)
    Results = pool.map(process_combination, COLOFINTEREST1)
    pool.close()
    pool.join()

    #### TRAVEL ASSIGNMENT ####
    #--- Create network compatible with frank_wolfe function
    nt = Network('net')
    node = Vertex("a")

    # Use the file created above
    with open("data/clean/test_network.csv") as fo: #TODO
        lines = fo.readlines()[1:]
        for ln in lines:
            eg = ln.split(',')
            nt.add_edge(Edge(eg))
    nt.init_cost()       
    g_df = pd.read_csv("data/clean/test_network.csv") #TODO
    #--- Prepare batch run
    # Gather all result OD matrices
    OD_matrix_names = []
    OD_matrix = []

    for result in Results:
        OD_matrix_names.append(list(result.keys()))
        OD_matrix.append(list(result.values()))

    OD_matrices_names = [item for sublist in OD_matrix_names for item in sublist]
    OD_matrices_names.append('school_pop_dens_school_count_noEQ')
    OD_matrices = [dataframe for sublist in OD_matrix for dataframe in sublist]
    OD_matrices.append(school_pop_dens_school_count_noEQ)

    # create dictionary of igraph ID to modified osmID
    centroid_igraph_to_mod_osmID = {}
    for i in range(len(centroids)):
        centroid_igraph_to_mod_osmID[i] = nodes_carbike_centroids_RER_complete.loc[nodes_carbike_centroids_RER_complete['ig'] == centroids[i]]['osmid'].apply(lambda x: 'N'+ (str(x) + '.0').zfill(5)).values[0]

    # Create dictionary of matrix names and dict index
    dict_index = {}
    for i in range(len(OD_matrices_names)):
        dict_index[i] = OD_matrices_names[i]

    #--- Run frank-wolfe
    dicts = []
    for name in OD_matrices_names:
        vol2 = None

        # Get OD matrix
        OD = OD_matrices[OD_matrices_names.index(name)]

        # Rename the columns and rows according to the modified osmID 
        OD = OD.rename(columns = {i : centroid_igraph_to_mod_osmID[i] for i in range(len(OD))}) #rename index of centroid as osmid of centroid
        OD.index = OD.columns

        # From all centroids to all centroids
        origins = OD.columns
        destinations = origins
        
        vol2 = frank_wolfe(nt, OD, origins, destinations)
        dicts.append(vol2)

    #### CALCULATE BENEFIT METRIC ####
    # Define the file path
    file_path = './data/clean/identified_gaps.csv' 
    mygaps = pd.read_csv(file_path, chunksize=100000) 

    def process_chunk(chunk):
            global g_df, dicts, g_igraph
            chunk['path'] = chunk['path'].apply(eval)
            for j in range(len(dicts)):
                    chunk["B_star"+str(j)] = chunk.apply(lambda x: 
                                    np.sum([dicts[j][g_df.loc[g_df['id'] == i]['edge'].values[0]] * \
                                            g_igraph.es[i]["length"] \
                                            for i in x.path]), 
                                    axis=1)
                    chunk["B"+str(j)] = chunk["B_star"+str(j)] / chunk["length"]
            rows_to_delete = chunk[['B'+str(j) for j in range(len(dicts))]].apply(lambda x: all(val == 0.000000 for val in x), axis=1)
            chunk = chunk[~rows_to_delete]
            print('processed one chunk!')
            return chunk

    pool = mp.Pool(processes=num_processes)
    results = pool.map(process_chunk, mygaps)

    pool.close()
    pool.join()

    dicts_df = pd.DataFrame.from_dict(dicts)
    dicts_df.to_csv('./data/clean/dicts_schools.csv')
    #### SAVE RESULTS ####
    # Open the output file in append mode
    output_file = "./data/clean/gaps_benefit_metric_schools.csv"
    with open(output_file, "a") as f:
            for df_chunk in results:
                    df_chunk.to_csv(f, header=f.tell() == 0, index=False)
