Worflow:

- get and clean data:
    - get_geofabrik_data_IDF.ipynb: get all available OSM data for the region of Ile-de-France (surrounding GPM). 
    
- create each layer of the network with separate notebooks:
    - create_grid.ipynb -> grid of GPM with centroids -> saved in data/processed/grid_GPM.csv
    - create_public_transport_networks.ipynb -> use geofabrik data (clipped to GPM) to create RER and metro networks -> saved as metro.csv and RER.csv in data/processed
    - Create_car_bike_networks.ipynb -> car and bike layers -> saved as carbike_edges.csv and carbike_nodes.csv in data/processed 

- combine everything into one network:
    - create_complete_network.ipynb 
