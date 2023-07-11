
# # Creating the OD Matrices
# - Matrix 0: shortest trips between centroids
# - Baseline: pop density and exp(normalized distance) -> gravity model baseline like Yap et al.
# - Matrix set 1: equalizing for median income, education level, number of schools and number of jobs SEPARATELY
# - Matrix set 2: equalizing for different attributes in O and D. O/D equalized for education level/number of schools, median income/number of jobs

if __name__ == '__main__':

    import pandas as pd
    import os
    os.environ['USE_PYGEOS'] = '0'
    import geopandas as gpd
    import networkx as nx
    import shapely
    import multiprocess as mp
    import numpy as np
    from itertools import combinations
    import igraph as ig
    from ta_lab.assignment.assign import frank_wolfe
    from ta_lab.assignment.line import *
    from ta_lab.assignment.graph import *
    from ta_lab.assignment.shortest_path import ShortestPath as SPP
    from ast import literal_eval
    crs_fr = 2154

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
        

    #--- Custom function
    # equalize an OD for the same attribute in O and D
    def equalization_all(od, variable, colname, delta, centroids): 
        
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
            i +=1
        
        return od_

    #--- Custom function to use the function above in a batch
    def clean_data_with_od_matrices(nodes_carbike_centroids_RER_complete, baseline_df, centroids, COLOFINTEREST, delta):
        col_tokeep = ['osmid', 'ig', 'CODE_IRIS', COLOFINTEREST]
        COLOFINTEREST_df = nodes_carbike_centroids_RER_complete.loc[nodes_carbike_centroids_RER_complete['centroid'] == True].copy()
        COLOFINTEREST_df = COLOFINTEREST_df[col_tokeep]
        
        OD_equalization = equalization_all(baseline_df, COLOFINTEREST_df, COLOFINTEREST, delta, centroids)
        
        OD_equalization_name = "OD_equalization_" + COLOFINTEREST + "_" + str(delta)
        
        return {OD_equalization_name: OD_equalization}

    #--- Custom function (adapted from Jin)
    # equalize an OD for DIFFFERENT attributes in O and D
    # multiply baseline with the number of opportunities in the destination and with the attribute of i over the avg **-delta

    def equalization_all_2attributes(od, variable, colnameO, colnameD, delta, centroids): 
        
        od_ = od.copy()
        variable_ = variable.copy()
        
        variable_average1 = np.mean(variable_[colnameO])
        
        # calculate the attribute of i over the avg and **-delta 
        variable_['weightO'] = variable_[colnameO].apply(lambda x: (x / variable_average1) ** -delta)

        # get the number of opportunities at j
        variable_['weightD'] = variable_[colnameD]
        
        i = 0
        for val in variable_['ig']:
            weightO = variable_.loc[variable_['ig'] == val]['weightO'].iloc[0]
            weightD = variable_.loc[variable_['ig'] == val]['weightD'].iloc[0]
            try:
                od_.loc[centroids.index(val)] *= weightO #row = origin
                od_[centroids.index(val)] *= weightD #column = destination
            except:
                continue
            i += 1
        
        return od_

    #--- Custom function to use the function above in a batch
    def equalization_with_2attributes(nodes_carbike_centroids_RER_complete, baseline_df, centroids, COLOFINTEREST1, COLOFINTEREST2, delta):
        col_tokeep = ['osmid', 'ig', 'CODE_IRIS', COLOFINTEREST1, COLOFINTEREST2]
        COLSOFINTEREST_df = nodes_carbike_centroids_RER_complete.loc[nodes_carbike_centroids_RER_complete['centroid'] == True].copy()
        COLSOFINTEREST_df = COLSOFINTEREST_df[col_tokeep]
        
        equalized_od = equalization_all_2attributes(baseline_df, COLSOFINTEREST_df, COLOFINTEREST1, COLOFINTEREST2, delta, centroids)
        
        equalized_od_name = "OD_equalization_" + COLOFINTEREST1 + "_O_"+ COLOFINTEREST2 + "_D_delta_" + str(delta)
        
        return {equalized_od_name: equalized_od}


    ### Creating the network in both NetworkX and igraph

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
    seq = g_igraph.vs.select(centroid_eq = True)
    centroids = [v.index for v in seq]
    centroids = centroids[0:2] #for testing purposes 
    node_combinations = list(combinations(centroids, 2))


    # ## Matrix 0: shortest path between each pair of centroids
    # Create OD matrix
    def process_node(args):
        start_node, end_node = args
        global g_igraph
        shortest_path_length = g_igraph.shortest_paths_dijkstra(source=start_node, target=end_node, weights='weight')[0][0]
        return (start_node, end_node, shortest_path_length)

    # Number of processes (cores) to use for parallel processing
    num_processes = mp.cpu_count()
    g_igraph

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


    ## Matrix Set 1: equalizing for median income, education level, number of schools and number of jobs SEPARATELY (and multiplying with the population density of j)

    ### Baseline: population density of i and j and exponential term with normalised distance

    def process_node(args):
        global matrix
        o, d = args
        if o == d:
            return (o, d, 0)
        else:
            normalized_dist = matrix[o][d] / matrix.max()
            demand = (
                (g_igraph.vs[centroids[o]]['pop_dens'] * g_igraph.vs[centroids[d]]['pop_dens'])
                * dist_decay * np.exp(-1 * normalized_dist)
            )
            return (o, d, demand)

    baseline = np.zeros((len(centroids), len(centroids)))
    maxtrips = 100
    dist_decay = 1

    num_processes = mp.cpu_count()
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
    baseline_df1 = pd.DataFrame(baseline)

    pool.close()
    pool.join()

    # ### Running the equalisation code like in Jin's paper

    def process_data(args):
        COLOFINTEREST = args
        delta_list = [0.5, 1, 1.5]
        results = {}
        for delta in delta_list:
            result = clean_data_with_od_matrices(nodes_carbike_centroids_RER_complete, baseline_df1, centroids, COLOFINTEREST, delta)
            results.update(result)
        return results

    num_processes = mp.cpu_count()

    # Create a pool of processes
    pool = mp.Pool(processes=num_processes)    
    COLOFINTEREST_list = ['median_income', 'school_count', 'num_jobs', 'edu_level']
    arguments = [COLOFINTEREST for COLOFINTEREST in COLOFINTEREST_list]
    results = pool.map(process_data, arguments)
    pool.close()
    pool.join()

    results_JinEQ = results

    ######### Matrix Set 2: equalize for O/D attributes median income/ number of jobs, education level/number of schools
    # Note that this is a MODIFIED equation: population density of j is replaced by the number of relevant opportunities in j, and only the attribute of the origin is equalized for, not the attribute of the destination. 
    # ### New baseline: pop density of i ONLY and distance decay

    def process_node(args):
        global matrix
        o, d = args
        if o == d:
            return (o, d, 0)
        else:
            normalized_dist = matrix[o][d] / matrix.max()
            demand = (
                (g_igraph.vs[centroids[o]]['pop_dens'])
                * dist_decay * np.exp(-1 * normalized_dist)
            )
            return (o, d, demand)

    baseline = np.zeros((len(centroids), len(centroids)))
    maxtrips = 100
    dist_decay = 1

    num_processes = mp.cpu_count()
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
    baseline_df2 = pd.DataFrame(baseline)

    pool.close()
    pool.join()

    # New equation: population density i * number of opportunities in j * distance decay * (attribute i / avg attribute)**-delta

    def process_combination(combination):
        COLOFINTEREST1, COLOFINTEREST2 = combination
        delta_list = [0.5, 1, 1.5]
        results = {}
        for delta in delta_list:
            result = equalization_with_2attributes(nodes_carbike_centroids_RER_complete, baseline_df2, centroids, COLOFINTEREST1, COLOFINTEREST2, delta)
            results.update(result)
        return results

    combinations = [['median_income', 'num_jobs'], ['edu_level', 'school_count']]
    num_processes = mp.cpu_count()
    pool = mp.Pool(processes=num_processes)
    Results = pool.map(process_combination, combinations)
    pool.close()
    pool.join()

    ####### Assign traffic flow

    #--- Create dataframe of edges compatible with frank_wolfe function
    # goal columns: edge name, source, target, free flow time, capacity, alpha, beta

    g_df = nx.to_pandas_edgelist(G)

    # Create compatible edge names
    g_df['edge'] = g_df.index + 1
    g_df['edge'] = g_df['edge'].apply(lambda x: 'E'+ str(x).zfill(4))

    # Adding the columns we don't have from the NetworkX network
    g_df = g_df[['edge', 'source', 'target', 'length', 'geometry', 'id']]
    g_df['capacity'] = 1e10
    g_df['alpha'] = 0.15 #no idea how this is set
    g_df['beta'] = 4.0 #same here

    # Create compatible node names based on the osmIDs
    g_df['source'] = g_df['source'].apply(lambda x: 'N'+ str(x).zfill(5))
    g_df['target'] = g_df['target'].apply(lambda x: 'N'+ str(x).zfill(5))
    g_df.reset_index(inplace=True)

    # We have to explicitly say, and assume, that each link is a two-way road
    g_df2 = g_df.copy()
    g_df2['source'] = g_df['target']
    g_df2['target'] = g_df['source']
    g_df2['edge'] = g_df2.index + 1 + len(g_df)
    g_df2['edge'] = g_df2['edge'].apply(lambda x: 'E'+ str(x).zfill(4))
    g_df = pd.concat([g_df, g_df2])
    geoms = g_df[['edge', 'geometry', 'index']]

    # Clean-up
    g_df.drop(['geometry', 'index'], axis=1, inplace=True)

    # Correct order of columns
    g_df = g_df[['edge', 'source', 'target', 'length', 'capacity', 'alpha', 'beta', 'id']]

    # Save to csv
    g_df.to_csv("network.csv", index=False)


    #--- Create network compatible with frank_wolfe function
    nt = Network('net')
    node = Vertex("a")

    # Use the file created above
    with open("network.csv") as fo:
        lines = fo.readlines()[1:]
        for ln in lines:
            eg = ln.split(',')
            nt.add_edge(Edge(eg))

    nt.init_cost()       

    #--- Make it a batch run
    # Gather all result OD matrices
    results_JinEQ.extend(Results)
    OD_matrix_names = []
    OD_matrix = []

    for result in results_JinEQ:
        OD_matrix_names.append(list(result.keys()))
        OD_matrix.append(list(result.values()))

    OD_matrices_names = [item for sublist in OD_matrix_names for item in sublist]
    OD_matrices = [dataframe for sublist in OD_matrix for dataframe in sublist]

    # create dictionary of igraph ID to modified osmID
    centroid_igraph_to_mod_osmID = {}
    for i in range(len(centroids)):
        centroid_igraph_to_mod_osmID[i] = nodes_carbike_centroids_RER_complete.loc[nodes_carbike_centroids_RER_complete['ig'] == centroids[i]]['osmid'].apply(lambda x: 'N'+ (str(x) + '.0').zfill(5)).values[0]

    # create dictionary of dict index to matrix name
    dict_index = {}
    for i in range(len(OD_matrices_names)):
        dict_index[i] = OD_matrices_names[i]


    # run frank-wolfe on all of them 
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


    # Function to calculate the Benefit metric of each gap
    def process_chunk(file_path, chunk_size):
        chunks = []
        for chunk in pd.read_csv(file_path, nrows = 150, chunksize=chunk_size):
            chunk['path'] = chunk['path'].apply(eval)
            for j in range(len(dicts)):
                chunk["B_star"+str(j)] = chunk.apply(lambda x: 
                                        np.sum([dicts[j][g_df.loc[g_df['id'] == i]['edge'].values[0]] * \
                                                g_igraph.es[i]["length"] \
                                                for i in x.path]), 
                                        axis=1)
                chunk["B"+str(j)] = chunk["B_star"+str(j)] / chunk["length"]
            chunks.append(chunk)
        return chunks

    # Define the file path
    file_path = './data/clean/mygaps2.csv'

    # Define the chunk size
    chunk_size = 150
    num_processes = mp.cpu_count()
    pool = mp.Pool(processes=num_processes)

    results = pool.apply(process_chunk, args=(file_path, chunk_size))
    pool.close()
    pool.join()

    # Concatenate the processed chunks into a single DataFrame
    mygaps = pd.concat(results)

    # Sort the dataframe by each benefit metric and save in separate csvs
    for j in range(len(dicts)):
        mygaps.sort_values(by=['B'+str(j)], ascending=False).head(10).to_csv('./data/clean/mygaps_B'+str(j)+'.csv', index=False)

    dict_index = pd.DataFrame.from_dict(dict_index, orient='index')
    dict_index.to_csv('data/clean/dict_index.csv')


