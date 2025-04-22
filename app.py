import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import folium_static
import math
from scipy.spatial import distance
import contextily as cx
import networkx as nx
from shapely.geometry import Point, LineString
import heapq
from shapely.ops import nearest_points

# Set page configuration
st.set_page_config(
    page_title="LIC BID Pop-up Event Location Selection Tool",
    page_icon="🗺️",
    layout="wide"
)

# Custom CSS to improve app appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3D59;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E3D59;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #666;
    }
    .highlight {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #FFD700;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">LIC BID Pop-up Event Location Selection Tool</p>', unsafe_allow_html=True)

st.markdown("""
<p class="info-text">This tool helps identify optimal parking lot locations for pop-up events 
in the Long Island City BID South Expansion area using path planning algorithms.</p>
""", unsafe_allow_html=True)

# Function to load data
@st.cache_data
def load_data():
    """Load all required datasets"""
    try:
        # Load parking lots
        parking_lots = gpd.read_file('EXPORT/parking_lots.geojson')
        
        # Load street pavement ratings
        pavement = gpd.read_file('EXPORT/pavement_ratings.geojson')
        
        # Load subway entrances
        subway = gpd.read_file('EXPORT/subway_entrances.geojson')
        
        # Load population/housing data
        population = gpd.read_file('EXPORT/population.geojson')
        
        # Load Citi Bike stations
        bike_stations = gpd.read_file('EXPORT/citibike_stations.geojson')
        
        # Load bike lanes
        bike_lanes = gpd.read_file('EXPORT/bike_lanes.geojson')
        
        # Load traffic flow data
        traffic = gpd.read_file('EXPORT/traffic_flow.geojson')
        
        # Load context area (LIC South Expansion)
        context_area = gpd.read_file('DATA/LIC_South_Expansion.geojson')
        
        # Load POIs
        poi = gpd.read_file('EXPORT/poi.geojson')
        
        return parking_lots, pavement, subway, population, bike_stations, bike_lanes, traffic, context_area, poi
        
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # If there's an error, use sample data for demonstration
        return None, None, None, None, None, None, None, None, None

# Load data
with st.spinner('Loading data...'):
    parking_lots, pavement_data, subway, population, bike_stations, bike_lanes, traffic, context_area, poi = load_data()
    
# Check if data is loaded successfully
if parking_lots is None:
    st.error("Failed to load data. Please check data paths and formats.")
    st.stop()

# After loading data, ensure all datasets have valid CRS
def ensure_valid_crs(gdf, default_crs="EPSG:4326", name="dataset"):
    """Ensure a GeoDataFrame has a valid CRS, setting a default if needed"""
    if gdf is None or gdf.empty:
        return gdf
        
    if gdf.crs is None:
        st.warning(f"{name} has no CRS. Setting to {default_crs}.")
        gdf = gdf.set_crs(default_crs)
    
    return gdf

# Apply to all datasets
parking_lots = ensure_valid_crs(parking_lots, name="Parking lots")
pavement_data = ensure_valid_crs(pavement_data, name="Pavement data")
subway = ensure_valid_crs(subway, name="Subway")
population = ensure_valid_crs(population, name="Population")
bike_stations = ensure_valid_crs(bike_stations, name="Bike stations")
bike_lanes = ensure_valid_crs(bike_lanes, name="Bike lanes")
traffic = ensure_valid_crs(traffic, name="Traffic")
context_area = ensure_valid_crs(context_area, name="Context area")

# Display dataset information
st.sidebar.markdown("## Datasets")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Parking Lots", len(parking_lots))
with col2:
    st.metric("Subway Entrances", len(subway))
with col3:
    st.metric("Citi Bike Stations", len(bike_stations))
with col4:
    st.metric("Housing Units", int(population['Units_CO'].sum()) if 'Units_CO' in population.columns else "N/A")

# # Create a debugging expander
# with st.expander("Debug Dataset Information"):
#     # Display dataset schemas
#     st.write("Pavement Data Columns:", pavement_data.columns.tolist())
#     st.write("Pavement Data Sample:", pavement_data.head(2))
    
#     st.write("Population Data Columns:", population.columns.tolist())
#     st.write("Population Data Sample:", population.head(2))
    
#     st.write("Subway Data Columns:", subway.columns.tolist())
#     st.write("Subway Data Sample:", subway.head(2))
    
#     st.write("Bike Stations Data Columns:", bike_stations.columns.tolist())
#     st.write("Bike Stations Data Sample:", bike_stations.head(2))

# Sidebar for weights and parameters
st.sidebar.markdown("## Model Parameters")

# In the sidebar weight sliders section:
st.sidebar.markdown("### Criteria Weights")
subway_weight = st.sidebar.slider("Subway Accessibility", 0.0, 1.0, 0.30, 0.05)
housing_weight = st.sidebar.slider("Housing Proximity", 0.0, 1.0, 0.30, 0.05)
biking_weight = st.sidebar.slider("Biking Infrastructure", 0.0, 1.0, 0.15, 0.05)
poi_weight = st.sidebar.slider("Points of Interest", 0.0, 1.0, 0.15, 0.05)
traffic_weight = st.sidebar.slider("Traffic Flow (lower is better)", 0.0, 1.0, 0.10, 0.05)

# Normalize weights to sum to 1
total_weight = subway_weight + housing_weight + biking_weight + poi_weight + traffic_weight
subway_weight = subway_weight / total_weight
housing_weight = housing_weight / total_weight
biking_weight = biking_weight / total_weight
poi_weight = poi_weight / total_weight
traffic_weight = traffic_weight / total_weight

# Additional parameters
st.sidebar.markdown("### Distance Parameters")
max_walking_dist = st.sidebar.slider("Max Walking Distance (meters)", 100, 2000, 800, 100)
max_biking_dist = st.sidebar.slider("Max Biking Distance (meters)", 200, 5000, 2000, 100)

# Street quality preference parameters
st.sidebar.markdown("### Street Quality Parameters")
good_street_factor = st.sidebar.slider("Good Street Preference", 1.0, 5.0, 3.0, 0.1, 
                                     help="How much people prefer good quality streets (higher = stronger preference)")
fair_street_factor = st.sidebar.slider("Fair Street Preference", 1.0, 3.0, 1.5, 0.1,
                                      help="How much people prefer fair quality streets (higher = stronger preference)")
poor_street_factor = st.sidebar.slider("Poor Street Penalty", 1.0, 10.0, 5.0, 0.1,
                                      help="How much people avoid poor quality streets (higher = stronger avoidance)")

# Function to prepare street network for path planning
def create_street_network_with_intersections(pavement_data, tolerance=1e-8):
    """
    Create a NetworkX graph from street segments, connecting intersecting streets
    
    Args:
        pavement_data: GeoDataFrame with street segments
        tolerance: Distance tolerance for identifying shared points
        
    Returns:
        G: NetworkX graph with nodes and edges
    """
    # Check input data
    if pavement_data is None or pavement_data.empty or 'geometry' not in pavement_data.columns:
        st.error("Invalid pavement data for network creation")
        return nx.Graph(), gpd.GeoDataFrame(), gpd.GeoDataFrame()
    
    # Create empty graph
    G = nx.Graph()
    
    # Extract all line endpoints to identify intersections
    all_points = {}
    point_to_node_id = {}
    node_counter = 0
    
    # First pass: collect all endpoints and possible intersections
    for idx, row in pavement_data.iterrows():
        try:
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
                
            # Handle different geometry types
            linestrings = []
            if geom.geom_type == 'LineString':
                linestrings = [geom]
            elif geom.geom_type == 'MultiLineString':
                linestrings = list(geom.geoms)
            else:
                continue
                
            # Process each linestring
            for line_idx, line in enumerate(linestrings):
                if not hasattr(line, 'coords') or len(list(line.coords)) < 2:
                    continue
                    
                # Get coordinates as tuples (rounded to handle floating point issues)
                coords = [(round(x, 10), round(y, 10)) for x, y in line.coords]
                
                # Add start and end points to collection
                start_point = coords[0]
                end_point = coords[-1]
                
                # Add to points dictionary for later processing
                for point in [start_point, end_point]:
                    if point in all_points:
                        all_points[point].append((idx, line_idx))
                    else:
                        all_points[point] = [(idx, line_idx)]
        except Exception as e:
            st.write(f"Error processing segment {idx} for points: {e}")
            continue
    
    # Second pass: create nodes for all unique points
    for point, segments in all_points.items():
        node_id = f"node_{node_counter}"
        point_to_node_id[point] = node_id
        G.add_node(node_id, pos=point)
        node_counter += 1
    
    # Third pass: add edges between nodes
    edges = []
    for idx, row in pavement_data.iterrows():
        try:
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
                
            # Handle different geometry types
            linestrings = []
            if geom.geom_type == 'LineString':
                linestrings = [geom]
            elif geom.geom_type == 'MultiLineString':
                linestrings = list(geom.geoms)
            else:
                continue
                
            # Get rating
            if 'RatingLaye' in row:
                rating = str(row['RatingLaye']).lower()
            elif 'rating' in row:
                rating = str(row['rating']).lower()
            else:
                rating = 'fair'
                
            # Process each linestring
            for line_idx, line in enumerate(linestrings):
                if not hasattr(line, 'coords') or len(list(line.coords)) < 2:
                    continue
                    
                # Get coordinates as tuples (rounded)
                coords = [(round(x, 10), round(y, 10)) for x, y in line.coords]
                start_point = coords[0]
                end_point = coords[-1]
                
                # Get node IDs
                start_node_id = point_to_node_id[start_point]
                end_node_id = point_to_node_id[end_point]
                
                # Skip self-loops
                if start_node_id == end_node_id:
                    continue
                
                # Calculate edge weight
                length = line.length
                
                # Apply quality factors
                if 'good' in rating:
                    weight = length / good_street_factor
                elif 'fair' in rating:
                    weight = length / fair_street_factor
                else:
                    weight = length * poor_street_factor
                
                # Add edge
                G.add_edge(start_node_id, end_node_id, weight=weight, length=length, rating=rating)
                
                # Add to edges list for GeoDataFrame
                edges.append({
                    'start_node': start_node_id,
                    'end_node': end_node_id,
                    'weight': weight,
                    'length': length,
                    'rating': rating,
                    'geometry': line
                })
        except Exception as e:
            st.write(f"Error processing segment {idx} for edges: {e}")
            continue
    # # Debug: Check a few edges in the graph
    # st.write("### Debug: Sample Edge Data")
    # sample_edges = list(G.edges(data=True))[:5]
    # for i, (u, v, data) in enumerate(sample_edges):
    #     st.write(f"Edge {i}: {u} → {v}")
    #     st.write(f"  Data: {data}")
    #     st.write(f"  Weight: {data.get('weight')}")
    #     st.write(f"  Length: {data.get('length')}")
    
    # Create GeoDataFrames
    nodes = []
    for node_id, data in G.nodes(data=True):
        pos = data.get('pos')
        if pos:
            nodes.append({
                'node_id': node_id,
                'geometry': Point(pos)
            })
    
    try:
        # Create node GeoDataFrame
        node_gdf = gpd.GeoDataFrame(nodes, geometry='geometry', crs=pavement_data.crs)
        
        # Create edge GeoDataFrame
        edge_gdf = gpd.GeoDataFrame(edges, geometry='geometry', crs=pavement_data.crs)
        
        # Report on created network
        # st.info(f"Created network with {len(G.nodes)} nodes and {len(G.edges)} edges")
        
        # Check connectivity
        connected_components = list(nx.connected_components(G))
        # st.info(f"Network has {len(connected_components)} connected components")
        largest_component = max(connected_components, key=len)
        # st.info(f"Largest component has {len(largest_component)} nodes ({len(largest_component)/len(G.nodes)*100:.1f}% of network)")
        
        return G, node_gdf, edge_gdf
    except Exception as e:
        st.error(f"Error creating GeoDataFrames: {e}")
        return nx.Graph(), gpd.GeoDataFrame(), gpd.GeoDataFrame()
    
# Function to connect origins and destinations to the street network
def connect_points_to_network(points_gdf, node_gdf, max_distance=100, name="points"):
    """
    Connect points (origins or destinations) to the nearest nodes in the street network
    
    Args:
        points_gdf: GeoDataFrame with points to connect
        node_gdf: GeoDataFrame with network nodes
        max_distance: Maximum connection distance
        name: Name for logging purposes
        
    Returns:
        connected_points: Dict mapping point indices to nearest node ids
    """
    connected_points = {}
    
    # Check if inputs are valid
    if points_gdf is None or points_gdf.empty:
        st.info(f"Empty {name} GeoDataFrame")
        return connected_points
    
    if node_gdf is None or node_gdf.empty:
        st.info(f"Empty node GeoDataFrame")
        return connected_points
    
    # # Print debug information
    # st.write(f"Connecting {name}:")
    # st.write(f"- {name} CRS: {points_gdf.crs}")
    # st.write(f"- node_gdf CRS: {node_gdf.crs}")
    
    # Check if points_gdf has a valid CRS
    if points_gdf.crs is None:
        st.warning(f"{name} GeoDataFrame has no CRS. Assuming same as node_gdf.")
        # Set CRS to match node_gdf without transforming
        points_gdf = points_gdf.set_crs(node_gdf.crs)
    
    # Check if node_gdf has a valid CRS
    if node_gdf.crs is None:
        st.warning("Network nodes GeoDataFrame has no CRS. Assuming EPSG:4326.")
        node_gdf = node_gdf.set_crs("EPSG:4326")
    
    # Ensure CRS match without transforming if one was None
    if points_gdf.crs != node_gdf.crs:
        st.info(f"Converting {name} CRS to match network nodes")
        try:
            points_gdf = points_gdf.to_crs(node_gdf.crs)
        except ValueError as e:
            st.error(f"CRS transformation error: {e}")
            # If transformation fails, try setting both to the same CRS without transforming
            st.warning("Attempting alternative approach: setting same CRS without transforming")
            common_crs = "EPSG:4326"  # Use WGS84 as a common reference
            points_gdf = points_gdf.set_crs(common_crs, allow_override=True)
            node_gdf = node_gdf.set_crs(common_crs, allow_override=True)
    
    # Process each point
    for idx, point in points_gdf.iterrows():
        try:
            # Get geometry - use centroid if it's not a point
            geom = point.geometry
            if geom is None or geom.is_empty:
                continue
                
            if geom.geom_type == 'Point':
                point_geom = geom
            else:
                # For polygons or other geometries, use centroid
                point_geom = geom.centroid
            
            # Calculate distances to all nodes
            distances = node_gdf.geometry.distance(point_geom)
            
            # Find nearest node within max distance
            min_distance = distances.min()
            if min_distance <= max_distance:
                nearest_idx = distances.idxmin()
                nearest_node_id = node_gdf.loc[nearest_idx, 'node_id']
                connected_points[idx] = nearest_node_id
            else:
                # If closest node is too far, log it
                if idx < 5:  # Only log a few to avoid flooding
                    st.info(f"{name} {idx} closest node is {min_distance:.1f}m away (max is {max_distance}m)")
        
        except Exception as e:
            # Log the error but continue processing other points
            if idx < 5:  # Limit error messages
                st.info(f"Error connecting {name} {idx}: {e}")
            continue
    
    # Report success rate
    success_rate = len(connected_points) / len(points_gdf) * 100 if len(points_gdf) > 0 else 0
    st.info(f"Connected {len(connected_points)} out of {len(points_gdf)} {name} ({success_rate:.1f}%)")
    
    return connected_points

# Function to calculate shortest paths
def calculate_path_scores(G, origin_connections, destination_connections, max_distance):
    """
    Calculate path scores between origins and destinations
    
    Args:
        G: NetworkX graph
        origin_connections: Dict mapping origin indices to node ids
        destination_connections: Dict mapping destination indices to node ids
        max_distance: Maximum path distance to consider
        
    Returns:
        path_scores: Dict of dicts with path scores
    """
    path_scores = {}
    
    # Debug output
    # st.write(f"Calculating paths between {len(origin_connections)} origins and {len(destination_connections)} destinations")
    
    # Check if network has connected components
    connected_components = list(nx.connected_components(G))
    # st.write(f"Network has {len(connected_components)} connected components")
    
    # Find largest component
    largest_component = max(connected_components, key=len) if connected_components else set()
    largest_pct = len(largest_component) / len(G.nodes()) * 100 if G.nodes() else 0
    # st.write(f"Largest component has {len(largest_component)} nodes ({largest_pct:.1f}% of network)")
    
    # Distance scaling factor (convert from degrees to meters)
    # This is an approximation - 1 degree is roughly 111,000 meters at the equator
    # For more precision, you could use a proper geodesic calculation
    SCALE_FACTOR = 111000  # Approximate meters per degree
    
    # For each origin
    for origin_idx, origin_node in origin_connections.items():
        path_scores[origin_idx] = {}
        
        # Skip if not in graph
        if origin_node not in G:
            continue
        
        try:
            # Calculate shortest paths from this origin to all other nodes
            shortest_paths = nx.single_source_dijkstra(G, origin_node, weight='weight', cutoff=max_distance/SCALE_FACTOR)
            
            # Extract distances and paths
            distances = shortest_paths[0]  # Dictionary of distances (in weight units)
            paths = shortest_paths[1]      # Dictionary of paths
            
            # For each destination
            for dest_idx, dest_node in destination_connections.items():
                # If destination is reachable
                if dest_node in distances:
                    # Get path
                    path = paths[dest_node]
                    
                    # Calculate the ACTUAL distance by summing edge lengths and scaling to meters
                    actual_distance = 0
                    
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        if G.has_edge(u, v):
                            # Get edge data
                            edge_data = G.get_edge_data(u, v)
                            
                            # Get length and scale to meters
                            segment_length = edge_data.get('length', 0) * SCALE_FACTOR
                            actual_distance += segment_length
                    
                    # Calculate score (1 for shortest path, decreasing for longer paths)
                    if actual_distance <= max_distance:
                        score = 1.0 - (actual_distance / max_distance)
                    else:
                        score = 0.0
                    
                    # Store score with actual distance in meters
                    path_scores[origin_idx][dest_idx] = {
                        'score': score,
                        'distance': actual_distance,
                        'path': path
                    }
                else:
                    # Destination not reachable within limit
                    path_scores[origin_idx][dest_idx] = {
                        'score': 0.0,
                        'distance': float('inf'),
                        'path': []
                    }
                    
        except nx.NetworkXNoPath:
            st.write(f"No path found from origin {origin_idx}")
            continue
        except Exception as e:
            # Skip this origin if there's an error in path calculation
            st.write(f"Error calculating paths from origin {origin_idx}: {e}")
            continue
    
    # Check if any paths were found
    path_count = sum(1 for origin in path_scores for dest in path_scores[origin] 
                if path_scores[origin][dest]['score'] > 0)
    st.write(f"Found {path_count} valid paths with scores > 0")
    
    return path_scores
        
# Function to aggregate path scores into location scores
def calculate_location_scores(path_scores, origins_gdf, destinations_gdf, weight_column=None):
    """
    Aggregate path scores into scores for each destination location
    
    Args:
        path_scores: Dict of dicts with path scores
        origins_gdf: GeoDataFrame with origin points
        destinations_gdf: GeoDataFrame with destination points
        weight_column: Optional column in origins_gdf to use as weights
        
    Returns:
        location_scores: Dict with aggregated scores for each destination
    """
    location_scores = {}
    
    # Initialize scores for each destination
    for dest_idx in destinations_gdf.index:
        location_scores[dest_idx] = {
            'total_score': 0.0,
            'accessible_origins': 0,
            'avg_distance': 0.0,
            'weighted_score': 0.0
        }
    
    # Calculate total weights if using weighted scoring
    total_weight = 1.0
    if weight_column and weight_column in origins_gdf.columns:
        total_weight = origins_gdf[weight_column].sum()
        if total_weight == 0:
            total_weight = 1.0  # Avoid division by zero
    
    # For each origin
    for origin_idx in path_scores:
        # Get origin weight
        origin_weight = 1.0
        if weight_column and weight_column in origins_gdf.columns:
            if origin_idx in origins_gdf.index:
                origin_weight = origins_gdf.loc[origin_idx, weight_column]
                if pd.isna(origin_weight):
                    origin_weight = 1.0
        
        # Normalize weight
        normalized_weight = origin_weight / total_weight
        
        # For each destination
        for dest_idx in path_scores[origin_idx]:
            # Get path data
            path_data = path_scores[origin_idx][dest_idx]
            score = path_data['score']
            distance = path_data['distance']
            
            # Skip unreachable destinations
            if score <= 0 or np.isinf(distance):
                continue
            
            # Update location score
            if dest_idx in location_scores:
                location_scores[dest_idx]['total_score'] += score
                location_scores[dest_idx]['accessible_origins'] += 1
                location_scores[dest_idx]['avg_distance'] += distance
                location_scores[dest_idx]['weighted_score'] += score * normalized_weight
    
    # Calculate averages
    for dest_idx in location_scores:
        accessible_origins = location_scores[dest_idx]['accessible_origins']
        if accessible_origins > 0:
            location_scores[dest_idx]['avg_distance'] /= accessible_origins
        else:
            # No accessible origins
            location_scores[dest_idx]['total_score'] = 0
            location_scores[dest_idx]['weighted_score'] = 0
            location_scores[dest_idx]['avg_distance'] = float('inf')
    
    return location_scores

# Function to calculate traffic scores
def calculate_traffic_scores(parking_lots, traffic_data, buffer_distance=150):
    """
    Calculate traffic score for each parking lot (lower traffic is better)
    
    Args:
        parking_lots: GeoDataFrame with parking lots
        traffic_data: GeoDataFrame with traffic information
        buffer_distance: Distance around parking lots to consider
        
    Returns:
        traffic_scores: Dict with traffic scores
    """
    traffic_scores = {}
    
    # Check if traffic data has required columns
    traffic_column = 'avg_daily_vol'
    if traffic_column not in traffic_data.columns:
        # Use a default score if no traffic data available
        for idx in parking_lots.index:
            traffic_scores[idx] = 0.5
        return traffic_scores
    
    # Get global min and max for normalization
    global_min = traffic_data[traffic_column].min()
    global_max = traffic_data[traffic_column].max()
    
    # Calculate score for each parking lot
    for idx, lot in parking_lots.iterrows():
        # Buffer the parking lot
        buffer = lot.geometry.buffer(buffer_distance)
        
        # Find intersecting streets
        nearby_streets = traffic_data[traffic_data.intersects(buffer)]
        
        if len(nearby_streets) == 0:
            # No traffic data available for this location
            traffic_scores[idx] = 0.5
            continue
        
        # Calculate average traffic
        avg_traffic = nearby_streets[traffic_column].mean()
        
        # Normalize and invert (lower traffic = higher score)
        if global_max > global_min:
            normalized_traffic = (avg_traffic - global_min) / (global_max - global_min)
        else:
            normalized_traffic = 0.5
            
        # Invert so lower traffic = higher score
        traffic_scores[idx] = 1.0 - normalized_traffic
    
    return traffic_scores

# Function to calculate POI proximity scores
def calculate_poi_scores(parking_lots, poi_data, max_distance=500):
    """
    Calculate POI proximity score for each parking lot
    
    Args:
        parking_lots: GeoDataFrame with parking lots
        poi_data: GeoDataFrame with points of interest
        max_distance: Maximum distance to consider (meters)
        
    Returns:
        poi_scores: Dict with POI proximity scores
    """
    poi_scores = {}
    
    # Check if datasets have compatible CRS
    if parking_lots.crs != poi_data.crs:
        poi_data = poi_data.to_crs(parking_lots.crs)
    
    # Distance scaling factor (convert from degrees to meters)
    SCALE_FACTOR = 111000  # Approximate meters per degree
    
    # Calculate score for each parking lot
    for idx, lot in parking_lots.iterrows():
        # Get lot centroid
        if hasattr(lot.geometry, 'centroid'):
            lot_point = lot.geometry.centroid
        else:
            lot_point = lot.geometry
            
        # Calculate distances to all POIs
        distances = [lot_point.distance(poi.geometry) * SCALE_FACTOR for _, poi in poi_data.iterrows()]
        
        if not distances:
            poi_scores[idx] = 0.0
            continue
            
        # Calculate score based on nearby POIs
        # The more POIs within max_distance, the higher the score
        nearby_count = sum(1 for d in distances if d <= max_distance)
        
        # Weight by inverse distance (closer POIs count more)
        weighted_distances = sum(max(0, 1 - (d / max_distance)) for d in distances if d <= max_distance)
        
        # Combine count and weighted distance metrics
        if nearby_count > 0:
            poi_scores[idx] = weighted_distances / nearby_count
        else:
            poi_scores[idx] = 0.0
    
    # Normalize scores to 0-1 range
    max_score = max(poi_scores.values()) if poi_scores else 1.0
    if max_score > 0:
        for idx in poi_scores:
            poi_scores[idx] = poi_scores[idx] / max_score
    
    return poi_scores

# Main function to calculate all scores
def calculate_all_scores(G, node_gdf, parking_lots, subway, population, bike_stations, 
                         traffic, poi, max_walking_dist, max_biking_dist, connection_distance=100):
    """
    Calculate all scores for parking lots
    
    Args:
        G: NetworkX graph of the street network
        node_gdf: GeoDataFrame with network nodes
        parking_lots, subway, population, bike_stations, traffic, poi: Required datasets
        max_walking_dist, max_biking_dist: Distance parameters
        connection_distance: Max distance to connect points to network
        
    Returns:
        results: GeoDataFrame with parking lots and scores
    """
    # Connect points to network
    parking_connections = connect_points_to_network(parking_lots, node_gdf, connection_distance)
    subway_connections = connect_points_to_network(subway, node_gdf, connection_distance)
    population_connections = connect_points_to_network(population, node_gdf, connection_distance)
    bike_connections = connect_points_to_network(bike_stations, node_gdf, connection_distance)
    
    # Calculate paths and scores
    if len(subway_connections) > 0:
        subway_path_scores = calculate_path_scores(
            G, subway_connections, parking_connections, max_walking_dist
        )
        subway_scores = calculate_location_scores(
            subway_path_scores, subway, parking_lots
        )
    else:
        subway_scores = {idx: {'weighted_score': 0.0} for idx in parking_lots.index}
    
    if len(population_connections) > 0:
        population_path_scores = calculate_path_scores(
            G, population_connections, parking_connections, max_walking_dist
        )
        housing_scores = calculate_location_scores(
            population_path_scores, population, parking_lots, weight_column='Units_CO'
        )
    else:
        housing_scores = {idx: {'weighted_score': 0.0} for idx in parking_lots.index}
    
    if len(bike_connections) > 0:
        bike_path_scores = calculate_path_scores(
            G, bike_connections, parking_connections, max_biking_dist
        )
        biking_scores = calculate_location_scores(
            bike_path_scores, bike_stations, parking_lots
        )
    else:
        biking_scores = {idx: {'weighted_score': 0.0} for idx in parking_lots.index}
    
    # Calculate traffic scores
    traffic_scores = calculate_traffic_scores(parking_lots, traffic)
    
    # Calculate POI proximity scores
    poi_scores = calculate_poi_scores(parking_lots, poi)
    
    # Combine all scores
    results = parking_lots.copy()
    
    # Add individual scores
    results['subway_score'] = [subway_scores.get(idx, {'weighted_score': 0.0})['weighted_score'] for idx in results.index]
    results['housing_score'] = [housing_scores.get(idx, {'weighted_score': 0.0})['weighted_score'] for idx in results.index]
    results['biking_score'] = [biking_scores.get(idx, {'weighted_score': 0.0})['weighted_score'] for idx in results.index]
    results['poi_score'] = [poi_scores.get(idx, 0.0) for idx in results.index]
    results['traffic_score'] = [traffic_scores.get(idx, 0.5) for idx in results.index]
    
    # Calculate weighted final score
    results['final_score'] = (
        subway_weight * results['subway_score'] +
        housing_weight * results['housing_score'] +
        biking_weight * results['biking_score'] +
        poi_weight * results['poi_score'] +
        traffic_weight * results['traffic_score']
    )
    
    # Normalize final scores
    max_score = results['final_score'].max()
    if max_score > 0:
        results['final_score'] = results['final_score'] / max_score
    
    return results

# Main content area with tabs
st.markdown('<p class="sub-header">Path-Based Multi-Criteria Analysis</p>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Map View", "Rankings", "Path Analysis", "Methodology"])

with tab1:
    # Build network and calculate scores
    with st.spinner("Building street network and calculating path-based scores..."):
        # Ensure all datasets have the same CRS
        for dataset in [pavement_data, subway, population, bike_stations, bike_lanes, traffic]:
            if dataset.crs != parking_lots.crs:
                dataset = dataset.to_crs(parking_lots.crs)
        
        # Create street network
        G, node_gdf, edge_gdf = create_street_network_with_intersections(pavement_data)
        
        if G is None:
            st.error("Failed to create street network. Please check pavement data.")
            st.stop()
        
        # Display network stats
        st.write(f"Street network created with {len(G.nodes)} nodes and {len(G.edges)} edges.")
        
        # Calculate scores
        # In tab1, update the function call:
        results = calculate_all_scores(
            G, node_gdf, parking_lots, subway, population, bike_stations, 
            traffic, poi, max_walking_dist, max_biking_dist
        )
        
        # Create a folium map
        center = context_area.to_crs(epsg=4326).unary_union.centroid
        m = folium.Map(
            location=[center.y, center.x],
            zoom_start=15,
            tiles="CartoDB positron"
        )

        # Add satellite imagery layer
        folium.TileLayer(
            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite',
            overlay=False
        ).add_to(m)

        # Add context area boundary
        folium.GeoJson(
            context_area.__geo_interface__,
            name="LIC South Expansion Area",
            style_function=lambda x: {'color': 'blue', 'weight': 2, 'fillOpacity': 0.1}
        ).add_to(m)

        # Add street network with quality coloring
        streets_layer = folium.FeatureGroup(name="Street Network by Quality").add_to(m)

        for idx, edge in edge_gdf.iterrows():
            # Color based on rating
            if 'good' in edge['rating']:
                color = 'green'
            elif 'fair' in edge['rating']:
                color = 'orange'
            else:
                color = 'red'
            
            # Add line to map
            folium.PolyLine(
                locations=[(p[1], p[0]) for p in edge['geometry'].coords],
                color=color,
                weight=3,
                opacity=0.7,
                popup=f"Rating: {edge['rating']}, Length: {edge['length']:.0f}m"
            ).add_to(streets_layer)

        # Create a layer for parking lots
        parking_layer = folium.FeatureGroup(name="Parking Lots (Score-Based)")

        # Add parking lots with score-based coloring
        for idx, row in results.iterrows():
            # Scale score to color (red to green)
            score = row['final_score']
            
            # Handle NaN values
            if pd.isna(score):
                color = '#AAAAAA'  # Gray color for NaN values
                score_str = 'N/A'
            else:
                color = f'#{int(255 * (1 - score)):02x}{int(255 * score):02x}00'
                score_str = f"{score:.2f}"
            
            # Format other scores with handling of NaN values
            subway_score_str = f"{row['subway_score']:.2f}" if not pd.isna(row['subway_score']) else "N/A"
            housing_score_str = f"{row['housing_score']:.2f}" if not pd.isna(row['housing_score']) else "N/A"
            biking_score_str = f"{row['biking_score']:.2f}" if not pd.isna(row['biking_score']) else "N/A"
            traffic_score_str = f"{row['traffic_score']:.2f}" if not pd.isna(row['traffic_score']) else "N/A"
            poi_score_str = f"{row['poi_score']:.2f}" if not pd.isna(row['poi_score']) else "N/A"
            
            # Create popup with scores
            # In the popup_html section:
            popup_html = f"""
            <div style="font-family: Arial; width: 200px;">
                <h4>Parking Lot {idx}</h4>
                <b>Overall Score:</b> {score_str}<br>
                <b>Subway Access:</b> {subway_score_str}<br>
                <b>Housing Access:</b> {housing_score_str}<br>
                <b>Biking Access:</b> {biking_score_str}<br>
                <b>POI Proximity:</b> {poi_score_str}<br>
                <b>Traffic Score:</b> {traffic_score_str}<br>
                <p style="font-size: 0.8em; margin-top: 5px;">
                    Higher scores are better for all metrics
                </p>
            </div>
            """
            
            try:
                # Add the parking lot polygon with transparent fill
                folium.GeoJson(
                    row.geometry.__geo_interface__,
                    name=f"Parking Lot {idx}",
                    style_function=lambda x, color=color: {
                        'fillColor': color,
                        'color': color,
                        'weight': 2,
                        'fillOpacity': 0.6
                    },
                    popup=folium.Popup(popup_html, max_width=300)
                ).add_to(parking_layer)
                
                # Add text label with score on top of polygon
                if hasattr(row.geometry, 'centroid'):
                    folium.Marker(
                        location=[row.geometry.centroid.y, row.geometry.centroid.x],
                        icon=folium.DivIcon(
                            icon_size=(50, 20),
                            icon_anchor=(25, 10),
                            html=f'<div style="font-size: 10pt; font-weight: bold; color: white; text-shadow: 1px 1px 2px black;">{score_str}</div>'
                        )
                    ).add_to(parking_layer)
            except Exception as e:
                st.write(f"Error adding parking lot {idx}: {e}")

        # Add the parking layer to the map after all lots are added
        parking_layer.add_to(m)

        # Add subway stations
        subway_layer = folium.FeatureGroup(name="Subway Stations")
        for idx, row in subway.iterrows():
            try:
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=5,
                    color='blue',
                    fill=True,
                    fill_color='blue',
                    fill_opacity=0.7,
                    popup=f"Subway Station: {row.get('name', 'Unknown')}"
                ).add_to(subway_layer)
            except:
                continue
        subway_layer.add_to(m)

        # Add bike stations
        bike_layer = folium.FeatureGroup(name="Citi Bike Stations")
        for idx, row in bike_stations.iterrows():
            try:
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=4,
                    color='green',
                    fill=True,
                    fill_color='green',
                    fill_opacity=0.7,
                    popup=f"Bike Station: {row.get('name', 'Unknown')}"
                ).add_to(bike_layer)
            except:
                continue
        bike_layer.add_to(m)

        # Add housing/population
        housing_layer = folium.FeatureGroup(name="Housing")
        for idx, row in population.iterrows():
            try:
                # Use Units_CO for size if available
                if 'Units_CO' in row and not pd.isna(row['Units_CO']):
                    size = min(20, max(5, int(row['Units_CO'] / 10)))
                    units = row['Units_CO']
                else:
                    size = 5
                    units = "Unknown"
                
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=size,
                    color='purple',
                    fill=True,
                    fill_color='purple',
                    fill_opacity=0.5,
                    popup=f"Housing: {units} units"
                ).add_to(housing_layer)
            except:
                continue
        housing_layer.add_to(m)
        
        # Add points of interest to the map
        poi_layer = folium.FeatureGroup(name="Points of Interest")
        for idx, row in poi.iterrows():
            try:
                name = row.get('NAME', f"POI {idx}")
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=6,
                    color='orange',
                    fill=True,
                    fill_color='orange',
                    fill_opacity=0.7,
                    popup=f"POI: {name}"
                ).add_to(poi_layer)
            except:
                continue
        poi_layer.add_to(m)

        # Add layer control
        folium.LayerControl().add_to(m)
    
        # Display the map
        st.markdown("### Interactive Map of Potential Pop-up Event Locations")
        st.markdown("""
        <div class="info-text">
            This map shows parking lots colored by their suitability score, along with the street network,
            subway stations, bike stations, and housing developments. The street network is colored by 
            pavement quality (green = good, orange = fair, red = poor).
        </div>
        """, unsafe_allow_html=True)
        
        folium_static(m, width=1000, height=600)
        
        st.markdown("""
        <div class="highlight">
            <strong>Map Legend:</strong><br>
            • Parking lots are colored on a scale from red (lowest score) to green (highest score)<br>
            • Blue circles represent subway stations<br>
            • Green circles represent Citi Bike stations<br>
            • Orange circles represent Points of Interest<br>
            • Purple circles represent housing developments (size indicates number of units)<br>
            • Streets are colored by quality: Green (good), Orange (fair), Red (poor)
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown("### Location Rankings")
    
    # Sort results by score
    sorted_results = results.sort_values(by='final_score', ascending=False).reset_index(drop=True)
    
    # Display top locations
    st.markdown("#### Top 5 Parking Lot Locations")
    
    # Prepare data for display
    display_cols = ['final_score', 'subway_score', 'housing_score', 'biking_score', 'poi_score', 'traffic_score']
    display_names = ['Overall Score', 'Subway Access', 'Housing Proximity', 'Biking Access', 'POI Proximity', 'Traffic Score']
    
    # Create a nicer table with rounded values
    top5_display = sorted_results[display_cols].head(5).round(2)
    top5_display.columns = display_names
    top5_display.index = [f"Location {i+1}" for i in range(len(top5_display))]
    
    st.dataframe(top5_display, use_container_width=True)
    
    # Create a bar chart comparing the top 5 locations
    import plotly.express as px
    
    st.markdown("#### Criteria Comparison for Top Locations")
    
    # Update plot data:
    plot_data = pd.DataFrame({
        'Location': [f"Location {i+1}" for i in range(min(5, len(sorted_results)))],
        'Subway': sorted_results['subway_score'].head(5).values,
        'Housing': sorted_results['housing_score'].head(5).values,
        'Biking': sorted_results['biking_score'].head(5).values,
        'POI': sorted_results['poi_score'].head(5).values,
        'Traffic': sorted_results['traffic_score'].head(5).values
    })

    # Melt the dataframe for plotting
    plot_data_melted = pd.melt(
        plot_data, 
        id_vars=['Location'], 
        value_vars=['Subway', 'Housing', 'Biking', 'POI', 'Traffic'],
        var_name='Criterion', 
        value_name='Score'
    )

    # Create a grouped bar chart
    fig = px.bar(
        plot_data_melted,
        x='Location',
        y='Score',
        color='Criterion',
        barmode='group',
        title='Comparison of Top 5 Locations by Individual Criteria',
        height=500
    )
    
    fig.update_layout(
        xaxis_title='',
        yaxis_title='Score (higher is better)',
        yaxis=dict(range=[0, 1.1]),
        legend_title='Criterion'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add stacked bar for weighted scores
    st.markdown("#### Weighted Contribution to Overall Score")
    
    weighted_data = pd.DataFrame({
        'Location': [f"Location {i+1}" for i in range(min(5, len(sorted_results)))],
        'Subway': sorted_results['subway_score'].head(5).values * subway_weight,
        'Housing': sorted_results['housing_score'].head(5).values * housing_weight,
        'Biking': sorted_results['biking_score'].head(5).values * biking_weight,
        'POI': sorted_results['poi_score'].head(5).values * poi_weight,
        'Traffic': sorted_results['traffic_score'].head(5).values * traffic_weight
    })

    # Update the melt for weighted data too
    weighted_data_melted = pd.melt(
        weighted_data, 
        id_vars=['Location'], 
        value_vars=['Subway', 'Housing', 'Biking', 'POI', 'Traffic'],
        var_name='Criterion', 
        value_name='Weighted Score'
    )
    
    # Create a stacked bar chart
    fig = px.bar(
        weighted_data_melted,
        x='Location',
        y='Weighted Score',
        color='Criterion',
        barmode='stack',
        title='Weighted Contribution to Overall Score',
        height=500
    )
    
    fig.update_layout(
        xaxis_title='',
        yaxis_title='Weighted Score',
        yaxis=dict(range=[0, 1.1]),
        legend_title='Criterion'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add full table with all locations
    st.markdown("#### All Locations Ranked")
    
    # Prepare full results table
    all_results = sorted_results[display_cols].round(2)
    all_results.columns = display_names
    all_results.index = [f"Location {i+1}" for i in range(len(all_results))]
    
    st.dataframe(all_results, use_container_width=True)
    
    # Add explanation
    st.markdown("""
    <div class="highlight">
        <strong>How to Interpret Rankings:</strong><br>
        The scores represent accessibility based on optimal paths through the street network, taking into account
        street quality. Higher scores indicate locations that are more accessible via high-quality streets from
        subway stations, housing developments, and bike stations, with lower traffic volumes.
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown("### Path Analysis")
    
    # Create a selectbox to choose a parking lot
    location_options = [f"Location {i+1} (Score: {row['final_score']:.2f})" 
                        for i, (_, row) in enumerate(sorted_results.iterrows())]
    
    selected_location_index = st.selectbox(
        "Select a parking lot to analyze paths:",
        range(len(location_options)),
        format_func=lambda i: location_options[i]
    )
    
    # Get the actual index in the original dataframe
    selected_lot_idx = sorted_results.index[selected_location_index]
    selected_lot = parking_lots.loc[selected_lot_idx]
    
    # Create tabs for different path types
    path_tab1, path_tab2, path_tab3 = st.tabs(["From Subway", "From Housing", "From Bike Stations"])
    
    with path_tab1:
        st.markdown("#### Paths from Subway Stations")
        
        # Create a map focused on the selected lot
        center = selected_lot.geometry.centroid
        m_subway = folium.Map(
            location=[center.y, center.x],
            zoom_start=16,
            tiles="CartoDB positron"
        )
        
        # Add the selected parking lot
        folium.GeoJson(
            selected_lot.geometry.__geo_interface__,
            name="Selected Parking Lot",
            style_function=lambda x: {'color': 'red', 'weight': 3, 'fillOpacity': 0.5}
        ).add_to(m_subway)
        
        # Add street network with quality coloring
        for idx, edge in edge_gdf.iterrows():
            # Color based on rating
            if 'good' in edge['rating']:
                color = 'green'
            elif 'fair' in edge['rating']:
                color = 'orange'
            else:
                color = 'red'
            
            # Add line to map
            folium.PolyLine(
                locations=[(p[1], p[0]) for p in edge['geometry'].coords],
                color=color,
                weight=2,
                opacity=0.7,
                popup=f"Rating: {edge['rating']}"
            ).add_to(m_subway)
        
        # Connect the parking lot to the network
        parking_connections = connect_points_to_network(
            gpd.GeoDataFrame([selected_lot], geometry='geometry'), 
            node_gdf, 
            max_distance=100
        )
        
        # Check if lot was connected
        if len(parking_connections) == 0:
            st.warning("Could not connect this parking lot to the street network.")
        else:
            # Connect subway stations to network
            subway_connections = connect_points_to_network(subway, node_gdf, max_distance=100)
            
            if len(subway_connections) == 0:
                st.warning("Could not connect any subway stations to the street network.")
            else:
                # Calculate paths from subway stations to the selected lot
                # Reverse connections for path calculation
                lot_to_subway_scores = calculate_path_scores(
                    G, parking_connections, subway_connections, max_walking_dist
                )
                
                # Add subway stations and paths
                for subway_idx, station in subway.iterrows():
                    if subway_idx in subway_connections:
                        # Add the station
                        folium.CircleMarker(
                            location=[station.geometry.y, station.geometry.x],
                            radius=5,
                            color='blue',
                            fill=True,
                            fill_color='blue',
                            fill_opacity=0.7,
                            popup=f"Subway Station: {station.get('name', 'Unknown')}"
                        ).add_to(m_subway)
                        
                        # Check if there's a path to this station
                        if (selected_lot_idx in parking_connections and 
                            subway_idx in lot_to_subway_scores.get(selected_lot_idx, {})):
                                
                            path_data = lot_to_subway_scores[selected_lot_idx][subway_idx]
                            
                            if path_data['score'] > 0 and len(path_data['path']) > 0:
                                # Get path nodes
                                path_nodes = path_data['path']
                                
                                # Get node locations
                                node_locations = []
                                for node_id in path_nodes:
                                    node_idx = node_gdf[node_gdf['node_id'] == node_id].index
                                    if len(node_idx) > 0:
                                        node = node_gdf.loc[node_idx[0]]
                                        node_locations.append((node.geometry.y, node.geometry.x))
                                
                                # Add path to map
                                folium.PolyLine(
                                    locations=node_locations,
                                    color='blue',
                                    weight=4,
                                    opacity=0.8,
                                    popup=f"Path to Station: {station.get('name', 'Unknown')}<br>Score: {path_data['score']:.2f}<br>Distance: {path_data['distance']:.0f}m"
                                ).add_to(m_subway)
                
                # Display the map
                st.write(f"Showing optimal paths from subway stations to Location {selected_location_index + 1}")
                folium_static(m_subway, width=800, height=500)
                
                # Add table with path scores
                st.markdown("#### Subway Station Accessibility Scores")
                
                # Create a table of subway station accessibility
                subway_path_data = []
                
                def get_path_quality_description(path_nodes, G):
                    """Get a description of path quality based on the nodes in the path"""
                    if not path_nodes or len(path_nodes) < 2:
                        return "Unknown"
                    
                    # Count the number of edges of each quality
                    good_count = 0
                    fair_count = 0
                    poor_count = 0
                    total_length = 0
                    
                    # Go through pairs of nodes
                    for i in range(len(path_nodes) - 1):
                        u = path_nodes[i]
                        v = path_nodes[i + 1]
                        
                        # Check if this edge exists in the graph
                        if G.has_edge(u, v):
                            edge_data = G.get_edge_data(u, v)
                            length = edge_data.get('length', 0)
                            rating = edge_data.get('rating', 'unknown')
                            
                            total_length += length
                            
                            if 'good' in rating:
                                good_count += length
                            elif 'fair' in rating:
                                fair_count += length
                            else:
                                poor_count += length
                    
                    # Calculate percentages
                    if total_length > 0:
                        good_pct = good_count / total_length * 100
                        fair_pct = fair_count / total_length * 100
                        poor_pct = poor_count / total_length * 100
                        
                        # Determine overall quality
                        if good_pct >= 60:
                            return f"Mostly Good ({good_pct:.0f}% good)"
                        elif good_pct + fair_pct >= 70:
                            return f"Mixed Good/Fair ({good_pct:.0f}% good, {fair_pct:.0f}% fair)"
                        elif fair_pct >= 60:
                            return f"Mostly Fair ({fair_pct:.0f}% fair)"
                        elif poor_pct >= 50:
                            return f"Poor Quality ({poor_pct:.0f}% poor)"
                        else:
                            return "Mixed Quality"
                    else:
                        return "Unknown"
                
                for subway_idx, station in subway.iterrows():
                    if (subway_idx in subway_connections and 
                        selected_lot_idx in parking_connections and 
                        subway_idx in lot_to_subway_scores.get(selected_lot_idx, {})):
                            
                        path_data = lot_to_subway_scores[selected_lot_idx][subway_idx]
                        
                        subway_path_data.append({
                            'Station': station.get('name', f'Station {subway_idx}'),
                            'Score': path_data['score'],
                            'Distance (m)': path_data['distance'],
                            'Path Quality': get_path_quality_description(path_data['path'], G)
                        })

                # For subway path data (and similar places)
                if subway_path_data:
                    subway_path_df = pd.DataFrame(subway_path_data)
                    subway_path_df = subway_path_df.sort_values('Score', ascending=False)
                    
                    # Format for display
                    subway_path_df['Score'] = subway_path_df['Score'].round(2)
                    
                    # Handle NaN and infinity values before converting to integer
                    subway_path_df['Distance (m)'] = subway_path_df['Distance (m)'].fillna(np.nan)  # Use pandas NA
                    subway_path_df['Distance (m)'] = subway_path_df['Distance (m)'].replace([np.inf, -np.inf], np.nan)  # Use pandas NA for infinity
                    
                    # Convert to int only where values are not NA
                    # This creates a numeric column with missing values rather than string 'N/A' values
                    subway_path_df['Distance (m)'] = subway_path_df['Distance (m)'].apply(
                        lambda x: int(round(x)) if pd.notna(x) else pd.NA
                    )
                    
                    # Display the dataframe - Streamlit will handle NA values appropriately
                    st.dataframe(subway_path_df, use_container_width=True)
                else:
                    st.write("No accessible subway stations found within the maximum walking distance.")
                    

    
    with path_tab2:
        st.markdown("#### Paths from Housing Developments")
        
        # Similar map and path visualization for housing
        center = selected_lot.geometry.centroid
        m_housing = folium.Map(
            location=[center.y, center.x],
            zoom_start=16,
            tiles="CartoDB positron"
        )
        
        # Add the selected parking lot
        folium.GeoJson(
            selected_lot.geometry.__geo_interface__,
            name="Selected Parking Lot",
            style_function=lambda x: {'color': 'red', 'weight': 3, 'fillOpacity': 0.5}
        ).add_to(m_housing)
        
        # Add street network with quality coloring
        for idx, edge in edge_gdf.iterrows():
            # Color based on rating
            if 'good' in edge['rating']:
                color = 'green'
            elif 'fair' in edge['rating']:
                color = 'orange'
            else:
                color = 'red'
            
            # Add line to map
            folium.PolyLine(
                locations=[(p[1], p[0]) for p in edge['geometry'].coords],
                color=color,
                weight=2,
                opacity=0.7,
                popup=f"Rating: {edge['rating']}"
            ).add_to(m_housing)
        
        # Connect the parking lot to the network
        parking_connections = connect_points_to_network(
            gpd.GeoDataFrame([selected_lot], geometry='geometry'), 
            node_gdf, 
            max_distance=100
        )
        
        # Check if lot was connected
        if len(parking_connections) == 0:
            st.warning("Could not connect this parking lot to the street network.")
        else:
            # Connect housing to network
            housing_connections = connect_points_to_network(population, node_gdf, max_distance=100)
            
            if len(housing_connections) == 0:
                st.warning("Could not connect any housing developments to the street network.")
            else:
                # Calculate paths from housing to the selected lot
                lot_to_housing_scores = calculate_path_scores(
                    G, parking_connections, housing_connections, max_walking_dist
                )
                
                # Add housing and paths
                for housing_idx, housing in population.iterrows():
                    if housing_idx in housing_connections:
                        # Get housing units
                        units = housing.get('Units_CO', 0)
                        if pd.isna(units):
                            units = 0
                        
                        # Adjust size based on units
                        size = min(20, max(5, int(units / 10)))
                        
                        # Add the housing development
                        folium.CircleMarker(
                            location=[housing.geometry.y, housing.geometry.x],
                            radius=size,
                            color='purple',
                            fill=True,
                            fill_color='purple',
                            fill_opacity=0.7,
                            popup=f"Housing: {units} units"
                        ).add_to(m_housing)
                        
                        # Check if there's a path to this housing
                        if (selected_lot_idx in parking_connections and 
                            housing_idx in lot_to_housing_scores.get(selected_lot_idx, {})):
                                
                            path_data = lot_to_housing_scores[selected_lot_idx][housing_idx]
                            
                            if path_data['score'] > 0 and len(path_data['path']) > 0:
                                # Get path nodes
                                path_nodes = path_data['path']
                                
                                # Get node locations
                                node_locations = []
                                for node_id in path_nodes:
                                    node_idx = node_gdf[node_gdf['node_id'] == node_id].index
                                    if len(node_idx) > 0:
                                        node = node_gdf.loc[node_idx[0]]
                                        node_locations.append((node.geometry.y, node.geometry.x))
                                
                                # Add path to map
                                folium.PolyLine(
                                    locations=node_locations,
                                    color='purple',
                                    weight=4,
                                    opacity=0.8,
                                    popup=f"Path to Housing: {units} units<br>Score: {path_data['score']:.2f}<br>Distance: {path_data['distance']:.0f}m"
                                ).add_to(m_housing)
                
                # Display the map
                st.write(f"Showing optimal paths from housing developments to Location {selected_location_index + 1}")
                folium_static(m_housing, width=800, height=500)
                
                # Add table with path scores
                st.markdown("#### Housing Development Accessibility Scores")
                
                # Create a table of housing accessibility
                housing_path_data = []
                
                for housing_idx, housing in population.iterrows():
                    if (housing_idx in housing_connections and 
                        selected_lot_idx in parking_connections and 
                        housing_idx in lot_to_housing_scores.get(selected_lot_idx, {})):
                            
                        path_data = lot_to_housing_scores[selected_lot_idx][housing_idx]
                        
                        units = housing.get('Units_CO', 0)
                        if pd.isna(units):
                            units = 0
                        
                        housing_path_data.append({
                            'Housing ID': housing_idx,
                            'Units': int(units),
                            'Score': path_data['score'],
                            'Distance (m)': path_data['distance'],
                            'Path Quality': get_path_quality_description(path_data['path'], G)
                        })
                
                # For housing path data
                if housing_path_data:
                    housing_path_df = pd.DataFrame(housing_path_data)
                    housing_path_df = housing_path_df.sort_values(['Units', 'Score'], ascending=False)
                    
                    # Format for display
                    housing_path_df['Score'] = housing_path_df['Score'].round(2)
                    
                    # Handle NaN and infinity values
                    housing_path_df['Distance (m)'] = housing_path_df['Distance (m)'].fillna(-1)
                    housing_path_df['Distance (m)'] = housing_path_df['Distance (m)'].replace([np.inf, -np.inf], -1)
                    housing_path_df['Distance (m)'] = housing_path_df['Distance (m)'].round(0).astype(int)
                    housing_path_df['Distance (m)'] = housing_path_df['Distance (m)'].replace(-1, "N/A")
                    
                    st.dataframe(housing_path_df, use_container_width=True)
                else:
                    st.write("No accessible housing developments found within the maximum walking distance.")
                    
    with path_tab3:
        
        st.markdown("#### Paths from Bike Stations")
        
        # Similar map and path visualization for bike stations
        center = selected_lot.geometry.centroid
        m_bike = folium.Map(
            location=[center.y, center.x],
            zoom_start=16,
            tiles="CartoDB positron"
        )
        
        # Add the selected parking lot
        folium.GeoJson(
            selected_lot.geometry.__geo_interface__,
            name="Selected Parking Lot",
            style_function=lambda x: {'color': 'red', 'weight': 3, 'fillOpacity': 0.5}
        ).add_to(m_bike)
        
        # Add street network with quality coloring
        for idx, edge in edge_gdf.iterrows():
            # Color based on rating
            if 'good' in edge['rating']:
                color = 'green'
            elif 'fair' in edge['rating']:
                color = 'orange'
            else:
                color = 'red'
            
            # Add line to map
            folium.PolyLine(
                locations=[(p[1], p[0]) for p in edge['geometry'].coords],
                color=color,
                weight=2,
                opacity=0.7,
                popup=f"Rating: {edge['rating']}"
            ).add_to(m_bike)
        
        # Add bike lanes
        folium.GeoJson(
            bike_lanes.__geo_interface__,
            name="Bike Lanes",
            style_function=lambda x: {'color': 'green', 'weight': 3, 'opacity': 0.7}
        ).add_to(m_bike)
        
        # Connect the parking lot to the network
        parking_connections = connect_points_to_network(
            gpd.GeoDataFrame([selected_lot], geometry='geometry'), 
            node_gdf, 
            max_distance=100
        )
        
        # Check if lot was connected
        if len(parking_connections) == 0:
            st.warning("Could not connect this parking lot to the street network.")
        else:
            # Connect bike stations to network
            bike_connections = connect_points_to_network(bike_stations, node_gdf, max_distance=100)
            
            if len(bike_connections) == 0:
                st.warning("Could not connect any bike stations to the street network.")
            else:
                # Calculate paths from bike stations to the selected lot
                lot_to_bike_scores = calculate_path_scores(
                    G, parking_connections, bike_connections, max_biking_dist
                )
                
                # Add bike stations and paths
                for bike_idx, station in bike_stations.iterrows():
                    if bike_idx in bike_connections:
                        # Add the station
                        folium.CircleMarker(
                            location=[station.geometry.y, station.geometry.x],
                            radius=5,
                            color='green',
                            fill=True,
                            fill_color='green',
                            fill_opacity=0.7,
                            popup=f"Bike Station: {station.get('name', 'Unknown')}"
                        ).add_to(m_bike)
                        
                        # Check if there's a path to this station
                        if (selected_lot_idx in parking_connections and 
                            bike_idx in lot_to_bike_scores.get(selected_lot_idx, {})):
                                
                            path_data = lot_to_bike_scores[selected_lot_idx][bike_idx]
                            
                            if path_data['score'] > 0 and len(path_data['path']) > 0:
                                # Get path nodes
                                path_nodes = path_data['path']
                                
                                # Get node locations
                                node_locations = []
                                for node_id in path_nodes:
                                    node_idx = node_gdf[node_gdf['node_id'] == node_id].index
                                    if len(node_idx) > 0:
                                        node = node_gdf.loc[node_idx[0]]
                                        node_locations.append((node.geometry.y, node.geometry.x))
                                
                                # Add path to map
                                folium.PolyLine(
                                    locations=node_locations,
                                    color='green',
                                    weight=4,
                                    opacity=0.8,
                                    popup=f"Path to Bike Station: {station.get('name', 'Unknown')}<br>Score: {path_data['score']:.2f}<br>Distance: {path_data['distance']:.0f}m"
                                ).add_to(m_bike)
                
                # Display the map
                st.write(f"Showing optimal paths from bike stations to Location {selected_location_index + 1}")
                folium_static(m_bike, width=800, height=500)
                
                # Add table with path scores
                st.markdown("#### Bike Station Accessibility Scores")
                
                # Create a table of bike station accessibility
                bike_path_data = []
                
                for bike_idx, station in bike_stations.iterrows():
                    if (bike_idx in bike_connections and 
                        selected_lot_idx in parking_connections and 
                        bike_idx in lot_to_bike_scores.get(selected_lot_idx, {})):
                            
                        path_data = lot_to_bike_scores[selected_lot_idx][bike_idx]
                        
                        bike_path_data.append({
                            'Station': station.get('name', f'Station {bike_idx}'),
                            'Score': path_data['score'],
                            'Distance (m)': path_data['distance'],
                            'Path Quality': get_path_quality_description(path_data['path'], G)
                        })
                
                # For bike station path data
                if bike_path_data:
                    bike_path_df = pd.DataFrame(bike_path_data)
                    bike_path_df = bike_path_df.sort_values('Score', ascending=False)
                    
                    # Format for display
                    bike_path_df['Score'] = bike_path_df['Score'].round(2)
                    
                    # Handle NaN and infinity values
                    bike_path_df['Distance (m)'] = bike_path_df['Distance (m)'].fillna(-1)
                    bike_path_df['Distance (m)'] = bike_path_df['Distance (m)'].replace([np.inf, -np.inf], -1)
                    bike_path_df['Distance (m)'] = bike_path_df['Distance (m)'].round(0).astype(int)
                    bike_path_df['Distance (m)'] = bike_path_df['Distance (m)'].replace(-1, "N/A")
                    
                    st.dataframe(bike_path_df, use_container_width=True)
                else:
                    st.write("No accessible bike stations found within the maximum biking distance.")

with tab4:
    st.markdown("### Methodology")
    
    st.markdown("""
    <div class="info-text">
        This tool uses a path-based multi-criteria decision analysis approach to evaluate potential 
        pop-up event locations. Unlike simple distance-based models, this approach simulates how people 
        would actually travel through the street network, taking into account street quality.
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("Street Network Model"):
        st.markdown("""
        #### Street Network Construction
        
        The model builds a graph representation of the street network where:
        
        - **Nodes** represent intersections and endpoints of street segments
        - **Edges** represent street segments
        - **Edge weights** are calculated based on both length and street quality:
            - Good quality streets: distance ÷ {good_street_factor}
            - Fair quality streets: distance ÷ {fair_street_factor}
            - Poor quality streets: distance × {poor_street_factor}
        
        This weighting scheme causes the path-finding algorithm to prefer high-quality streets,
        even if it means taking a slightly longer route.
        """.format(
            good_street_factor=good_street_factor,
            fair_street_factor=fair_street_factor,
            poor_street_factor=poor_street_factor
        ))
        
        # Visualize street quality impact
        st.markdown("#### Impact of Street Quality on Path Selection")
        
        # Create a simple example
        st.markdown("""
        For example, consider two possible routes between points A and B:
        
        - **Route 1**: 100m long, poor quality street
        - **Route 2**: 150m long, good quality street
        
        Using our current factors:
        - Route 1 effective distance: 100m × {0} = {1}m
        - Route 2 effective distance: 150m ÷ {2} = {3}m
        
        Since {3}m < {1}m, the algorithm would choose Route 2, the slightly longer but higher quality path.
        """.format(
            poor_street_factor,
            100 * poor_street_factor,
            good_street_factor,
            150 / good_street_factor,
            150 / good_street_factor,
            100 * poor_street_factor
        ))
    
    with st.expander("Path-Finding Algorithm"):
        st.markdown("""
        #### Dijkstra's Algorithm with Quality Weighting
        
        The model uses Dijkstra's algorithm to find optimal paths through the street network:
        
        1. Each origin point (subway station, housing development, bike station) is connected to the nearest node in the street network
        2. Each destination (parking lot) is also connected to the nearest node
        3. Dijkstra's algorithm finds the path with the lowest total weight
        4. The weight factors in both distance and street quality
        5. Paths exceeding the maximum distance threshold are discarded
        
        This approach simulates how people make real-world route choices, balancing distance with street quality.
        """)
    
    with st.expander("Scoring Methodology"):
        st.markdown("""
        #### Accessibility Scoring
        
        Each parking lot receives accessibility scores based on:
        
        1. **Subway Accessibility** (weight: {0:.0%})
            - Based on optimal paths from subway stations
            - Higher scores for shorter, higher-quality paths
        
        2. **Housing Proximity** (weight: {1:.0%})
            - Based on optimal paths from housing developments
            - Weighted by number of housing units
            - Higher scores for locations accessible to more residents via good streets
        
        3. **Biking Infrastructure** (weight: {2:.0%})
            - Based on optimal paths from bike stations
            - Higher scores for shorter, higher-quality paths
            - Based on optimal paths from bike stations
          - Higher scores for shorter, higher-quality paths
       
       4. **Traffic Flow** (weight: {3:.0%})
          - Based on traffic volume near each parking lot
          - Higher scores for lower traffic volumes
       
       The final score is a weighted sum of these individual scores, normalized to a 0-1 scale.
       """.format(
           subway_weight, 
           housing_weight, 
           biking_weight,
           traffic_weight
       ))
        
        # In the Scoring Methodology expander, update the markdown:
        st.markdown("""
        #### Accessibility Scoring

        Each parking lot receives accessibility scores based on:

        1. **Subway Accessibility** (weight: {0:.0%})
            - Based on optimal paths from subway stations
            - Higher scores for shorter, higher-quality paths

        2. **Housing Proximity** (weight: {1:.0%})
            - Based on optimal paths from housing developments
            - Weighted by number of housing units
            - Higher scores for locations accessible to more residents via good streets

        3. **Biking Infrastructure** (weight: {2:.0%})
            - Based on optimal paths from bike stations
            - Higher scores for shorter, higher-quality paths

        4. **POI Proximity** (weight: {3:.0%})
            - Based on proximity to points of interest
            - Higher scores for locations near more POIs
            - Weighted by distance (closer POIs count more)

        5. **Traffic Flow** (weight: {4:.0%})
            - Based on traffic volume near each parking lot
            - Higher scores for lower traffic volumes

        The final score is a weighted sum of these individual scores, normalized to a 0-1 scale.
        """.format(
        subway_weight, 
        housing_weight, 
        biking_weight,
        poi_weight,
        traffic_weight
        ))
   
    st.markdown("### Application for Pop-up Event Planning")

    st.markdown("""
    This tool provides valuable insights for planning pop-up events in the Long Island City BID:

    1. **Location Selection**: Identify the most accessible parking lots for converting to pop-up event spaces

    2. **Infrastructure Needs**: Understand which paths visitors are likely to take, helping plan for:
        - Temporary signage placement
        - Street cleaning priorities
        - Security positioning
        - Lighting improvements

    3. **Marketing Strategy**: Target promotion based on accessibility:
        - Focus subway advertising at stations with the best access to the event
        - Target housing developments with good paths to the venue
        - Promote bike access from specific Citi Bike stations

    4. **Schedule Optimization**: Coordinate event timing with traffic patterns and public transit schedules

    5. **Future Improvements**: Identify infrastructure improvements that would most benefit accessibility
    """)

# Footer
st.markdown("""
---
### About This Tool

This LIC BID Pop-up Event Location Selection Tool uses path planning algorithms to evaluate 
potential locations based on accessibility through the street network. The model incorporates 
street quality to simulate how people make real-world route choices.

**Data Sources:**
- NYC Planimetric Database (Parking Lots)
- Street Pavement Rating Data
- MTA Subway Stations
- NYC Housing Database
- Citi Bike Station Information
- NYC Bike Routes
- Automated Traffic Volume Counts

**References:**
- Batty, M. (2013). The New Science of Cities. MIT Press.
- Dijkstra, E. W. (1959). A note on two problems in connexion with graphs. Numerische Mathematik.
- Gehl, J. (2013). How to Study Public Life. Island Press.
""")
            