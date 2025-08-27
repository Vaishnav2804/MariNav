import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")
import datetime as dt
import os
import pprint
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import aisdb
import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from aisdb.database import sqlfcn
from aisdb.database.dbconn import PostgresDBConn
from aisdb.denoising_encoder import encode_greatcircledistance
from shapely.geometry import box
from tqdm import tqdm
from tqdm.notebook import tqdm
from collections import defaultdict

import cupy as cp
import h3
import networkx as nx
from numba import cuda


def process_interval(dbconn, start, end):
    """
    Extracts vessel movements, cleans the data, and ensures continuity
    """
    # Open a new connection with the database
    qry = aisdb.DBQuery(
        dbconn=dbconn,
        start=start,
        end=end,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        callback=aisdb.database.sqlfcn_callbacks.in_bbox_time_validmmsi,
    )

    # Decimate is for removing unnecessary points in the trajectory
    tracks = aisdb.track_gen.TrackGen(
        qry.gen_qry(fcn=sqlfcn.crawl_dynamic_static), decimate=False
    )

    # 1. Split the tracks based on the time between transmissions
    track_segments = aisdb.track_gen.split_timedelta(tracks, time_split)

    # 2. Filter out segments that are below the min-score threshold
    tracks_encoded = encode_greatcircledistance(
        track_segments, distance_threshold=distance_split, speed_threshold=speed_split
    )

    # 3. Interpolate the segments every five minutes to enforce continuity
    tracks_interpolated = aisdb.interp.interp_time(
        tracks_encoded, step=dt.timedelta(minutes=1)
    )

    # Returns a generator from a Python variable
    return list(tracks_interpolated)


def consecutive_deduplicate(id_list):
    """
    Sometimes a vessel stays in the same hexagon — we only want to log transitions between different hexagons. This removes consecutive duplicates.
    """
    result, prev_id = [], None
    for idx in id_list:
        if idx != prev_id:
            prev_id = str(idx)
            result.append(prev_id)
    return result

    # Counts vessel transitions between neighboring hexagons, building a "visit" map/dict.


def assign_visit(list_ids, ring_order=1):
    """
    We're counting how many times vessels move from hexagon `s` to neighboring hexagon `t`. This builds the movement network.
    """
    # Bulding a linked-list visit graph
    for i in range(0, len(list_ids) - 1):
        s, t = list_ids[i], list_ids[i + 1]  # source and target hexagon
        if t in h3.grid_ring(s, ring_order):  # find all neighboring hexagons around s.
            cell_visits[s][t] += (
                1  # record the number of times a vessel moves from hex s to hex t.
            )


def assign_visit_with_speed(
    list_ids, speeds, ship_type, shiptype_cell_visits_with_speed, ring_order=1
):
    """
    Count vessel transitions between neighboring hexagons and track speeds.
    """
    for i in range(0, len(list_ids) - 1):
        s, t = list_ids[i], list_ids[i + 1]
        if t in h3.grid_ring(s, ring_order):
            # Get the speed at the transition point (use speed from source position)
            speed = speeds[i] if i < len(speeds) else 0

            # Skip if speed is NaN
            if pd.isna(speed):
                speed = 0

            # Store both count and speed data
            if t not in shiptype_cell_visits_with_speed[ship_type][s]:
                shiptype_cell_visits_with_speed[ship_type][s][t] = {
                    "count": 0,
                    "speeds": [],
                }

            shiptype_cell_visits_with_speed[ship_type][s][t]["count"] += 1
            shiptype_cell_visits_with_speed[ship_type][s][t]["speeds"].append(speed)


def create_directed_graph(cell_visits):
    """
    This constructs a directed graph, where nodes are hexes and edges represent vessel flows.
    """
    G = nx.DiGraph()
    for src, targets in cell_visits.items():
        for tgt, weight in targets.items():
            if weight > 0:
                if not G.has_node(src):
                    lat, lon = h3.cell_to_latlng(src)
                    G.add_node(
                        src,
                        weight=0,
                        lat=lat,
                        lon=lon,
                        incoming_weight=0,
                        outgoing_weight=0,
                    )

                if not G.has_node(tgt):
                    lat, lon = h3.cell_to_latlng(tgt)
                    G.add_node(
                        tgt,
                        weight=0,
                        lat=lat,
                        lon=lon,
                        incoming_weight=0,
                        outgoing_weight=0,
                    )

                G.nodes[src]["weight"] += weight
                G.nodes[tgt]["weight"] += weight

                G.nodes[src]["outgoing_weight"] += weight
                G.nodes[tgt]["incoming_weight"] += weight

                G.add_edge(src, tgt, weight=weight)
    return G


def get_season_dates(year):
    return {  # TODO: Check this dates - they vary; how to handle?
        "spring": (dt.datetime(year, 3, 20), dt.datetime(year, 6, 20)),
        "summer": (dt.datetime(year, 6, 21), dt.datetime(year, 9, 22)),
        "autumn": (dt.datetime(year, 9, 23), dt.datetime(year, 12, 20)),
        "winter": (dt.datetime(year, 12, 21), dt.datetime(year + 1, 3, 19)),
    }


# Initialize a directed graph
G = nx.DiGraph()


# Convert cell visits to CuPy arrays for parallel processing
def preprocess_cell_visits(cell_visits):
    src_list, tgt_list, weight_list = [], [], []
    for src, targets in cell_visits.items():
        for tgt, weight in targets.items():
            if weight > 0:
                src_list.append(src)
                tgt_list.append(tgt)
                weight_list.append(weight)
    return cp.array(src_list), cp.array(tgt_list), cp.array(weight_list)


# GPU kernel for updating node attributes
@cuda.jit
def update_nodes(src, tgt, weight, lat_src, lon_src, lat_tgt, lon_tgt, node_data):
    idx = cuda.grid(1)
    if idx < src.size:
        node_data[idx, 0] = src[idx]
        node_data[idx, 1] = tgt[idx]
        node_data[idx, 2] = weight[idx]
        node_data[idx, 3] = lat_src[idx]
        node_data[idx, 4] = lon_src[idx]
        node_data[idx, 5] = lat_tgt[idx]
        node_data[idx, 6] = lon_tgt[idx]


# Main graph creation function
def create_directed_graph_gpu(cell_visits):
    src, tgt, weight = preprocess_cell_visits(cell_visits)

    # Get lat/lon for nodes
    lat_src = cp.array([h3.h3_to_geo(str(s))[0] for s in src])
    lon_src = cp.array([h3.h3_to_geo(str(s))[1] for s in src])
    lat_tgt = cp.array([h3.h3_to_geo(str(t))[0] for t in tgt])
    lon_tgt = cp.array([h3.h3_to_geo(str(t))[1] for t in tgt])

    # Prepare node data array
    node_data = cp.zeros((src.size, 7))

    # Configure GPU grid size
    threadsperblock = 256
    blockspergrid = (src.size + (threadsperblock - 1)) // threadsperblock

    # Launch GPU kernel
    update_nodes[blockspergrid, threadsperblock](
        src, tgt, weight, lat_src, lon_src, lat_tgt, lon_tgt, node_data
    )

    # Transfer node data back to CPU
    node_data = node_data.get()

    # Populate the NetworkX graph
    for i in range(len(node_data)):
        s, t, w, lat_s, lon_s, lat_t, lon_t = node_data[i]

        s, t = str(int(s)), str(int(t))

        if not G.has_node(s):
            G.add_node(
                s, weight=0, lat=lat_s, lon=lon_s, incoming_weight=0, outgoing_weight=0
            )
        if not G.has_node(t):
            G.add_node(
                t, weight=0, lat=lat_t, lon=lon_t, incoming_weight=0, outgoing_weight=0
            )

        G.nodes[s]["weight"] += w
        G.nodes[t]["weight"] += w

        G.nodes[s]["outgoing_weight"] += w
        G.nodes[t]["incoming_weight"] += w

        G.add_edge(s, t, weight=w)

    return G


def create_directed_graph_with_speed(cell_visits_with_speed):
    """
    Create directed graph with speed information on edges.
    """
    G = nx.DiGraph()
    for src, targets in cell_visits_with_speed.items():
        for tgt, data in targets.items():
            count = data["count"]
            speeds = data["speeds"]

            if count > 0:
                # Calculate average speed for this edge
                avg_speed = float(np.mean(speeds)) if speeds else 0.0
                speed_std = float(np.std(speeds)) if len(speeds) > 1 else 0.0

                # Add nodes if they don't exist
                if not G.has_node(src):
                    lat, lon = h3.cell_to_latlng(src)
                    G.add_node(
                        src,
                        weight=0,
                        lat=float(lat),
                        lon=float(lon),
                        incoming_weight=0,
                        outgoing_weight=0,
                    )

                if not G.has_node(tgt):
                    lat, lon = h3.cell_to_latlng(tgt)
                    G.add_node(
                        tgt,
                        weight=0,
                        lat=float(lat),
                        lon=float(lon),
                        incoming_weight=0,
                        outgoing_weight=0,
                    )

                # Update node weights
                G.nodes[src]["weight"] += count
                G.nodes[tgt]["weight"] += count
                G.nodes[src]["outgoing_weight"] += count
                G.nodes[tgt]["incoming_weight"] += count

                # Add edge with only simple data types (remove speed_list)
                G.add_edge(
                    src,
                    tgt,
                    weight=int(count),
                    avg_speed=avg_speed,
                    speed_std=speed_std,
                    speed_count=len(speeds),
                )
    return G


def compute_outgoing_speed_knots_for_track(df_track: pd.DataFrame) -> pd.DataFrame:
    # Sort by time to ensure correct segment ordering
    df = df_track.sort_values("time").reset_index(drop=True)

    lon = df["lon"].to_numpy(dtype=float)
    lat = df["lat"].to_numpy(dtype=float)
    t = df["time"].to_numpy(dtype=float)

    track = dict(
        mmsi=int(df["mmsi"].iloc[0])
        if "mmsi" in df.columns and pd.notna(df["mmsi"].iloc[0])
        else 0,
        lon=lon,
        lat=lat,
        time=t,
        dynamic=set(["lon", "lat", "time"]),
        static=set(["mmsi"]) if "mmsi" in df.columns else set(),
    )

    # seg_speeds length is N-1
    seg_speeds = aisdb.gis.delta_knots(track)

    # Outgoing alignment: write into rows 0..N-2; last row has no outgoing segment
    speed_out = np.full(len(df), np.nan, dtype=float)
    if len(seg_speeds) > 0:
        speed_out[:-1] = seg_speeds

    df["speed_knots_outgoing"] = speed_out

    return df


if __name__ == "__main__":
    gdf_hexagons = gpd.read_file("outputs/res-6/Hexagons_r6.shp")  # Load the shapefile
    gdf_hexagons = gdf_hexagons.to_crs(epsg=4326)  # Consistent CRS projection
    bounding_box = gdf_hexagons.total_bounds  # Extract the bounding box
    xmin, ymin, xmax, ymax = bounding_box  # Split the bounding box

    dbconn = PostgresDBConn(
        hostaddr="127.0.0.1",  # Replace with the PostgreSQL address
        port=5555,  # Replace with the PostgreSQL running port
        user="<>",  # Replace with the PostgreSQL username
        password="<>",  # Replace with your password
        dbname="aisviz",  # Replace with your database name
    )

    for year in range(2024, 2025):
        for month in range(1, 13):
            start_date = dt.datetime(year, month, 1)
            end_date = (start_date + dt.timedelta(days=32)).replace(
                day=1
            ) - dt.timedelta(seconds=1)

    def nested_defaultdict():
        return defaultdict(dict)

    def triple_nested_defaultdict():
        return defaultdict(nested_defaultdict)

    for year in range(2024, 2025):
        for month in range(1, 13):
            start_date = dt.datetime(year, month, 1)
            end_date = (start_date + dt.timedelta(days=32)).replace(
                day=1
            ) - dt.timedelta(seconds=1)
            print(
                f"Processing {start_date.strftime('%B')} of {year} from {start_date} to {end_date}"
            )

            # Preprocessing parameters
            time_split = dt.timedelta(hours=3)
            distance_split = 10000  # meters
            speed_split = 40  # knots

            # We’ll store separate visit maps for each ship type
            shiptype_cell_visits_with_speed = defaultdict(
                lambda: defaultdict(lambda: defaultdict(dict))
            )
            g2h3_vec = np.vectorize(h3.latlng_to_cell)
            current_start = start_date

            with tqdm(
                total=((end_date - start_date).days // 7) + 1, desc="Weeks", unit="week"
            ) as pbar_days:
                while current_start < end_date:
                    current_end = current_start + dt.timedelta(days=7)
                    if current_end > end_date:
                        current_end = end_date

                    vessel_list = process_interval(dbconn, current_start, current_end)
                    for vessel in tqdm(
                        vessel_list, total=len(vessel_list), desc="Vessels", leave=False
                    ):
                        raw_type = vessel.get("ship_type_txt", "UNKNOWN") or "UNKNOWN"
                        raw_type = raw_type.upper()

                        if raw_type in {"TANKER", "CARGO"}:
                            ship_type = "CARGO_TANKER"
                        else:
                            ship_type = raw_type

                        if ship_type not in {"CARGO_TANKER"}:
                            continue

                        # Convert vessel data to DataFrame for speed calculation
                        vessel_df = pd.DataFrame(
                            {
                                "lon": vessel["lon"],
                                "lat": vessel["lat"],
                                "time": vessel["time"],
                                "mmsi": [vessel.get("mmsi", 0)] * len(vessel["lon"]),
                            }
                        )

                        # Calculate outgoing speeds using your custom function
                        vessel_df_with_speed = compute_outgoing_speed_knots_for_track(
                            vessel_df
                        )

                        # Get the calculated speeds
                        speeds = vessel_df_with_speed["speed_knots_outgoing"].values

                        h3_ids = g2h3_vec(vessel["lat"], vessel["lon"], 6)
                        h3_ids = consecutive_deduplicate(h3_ids)

                        # Use the new function with speed tracking
                        assign_visit_with_speed(
                            h3_ids, speeds, ship_type, shiptype_cell_visits_with_speed
                        )

                    current_start = current_end
                    pbar_days.update(1)

            year_str = str(start_date.year)
            output_dir = os.path.join("graphs_with_speed", "ship_types", year_str)
            os.makedirs(output_dir, exist_ok=True)

            # Save one graph per ship type
            # Save one graph per ship type
            for ship_type, cell_visits in shiptype_cell_visits_with_speed.items():
                G = create_directed_graph_with_speed(cell_visits)  # Use new function
                safe_type = ship_type.replace(" ", "_").lower()
                month_str = start_date.strftime("%B").lower()
                file_name = f"GULF_VISITS_{safe_type}_{month_str}_{year}.gexf"
                file_path = os.path.join(output_dir, file_name)
                nx.write_gexf(G, file_path)
