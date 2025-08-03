import csv
from collections import defaultdict
from datetime import datetime

def load_full_wind_map(csv_path: str) -> dict[str, dict[datetime, tuple[float, float]]]:
    """
    Loads wind data from a CSV file into a nested dictionary structure.

    Args:
        csv_path (str): The path to the CSV file containing wind data.

    Returns:
        dict: A dictionary where keys are H3 cell IDs and values are
              dictionaries mapping timestamps to (u_wind, v_wind) tuples.
    """
    wind_map: defaultdict[str, dict[datetime, tuple[float, float]]] = defaultdict(dict)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            h3_cell = row["h3_cell"]
            timestamp = datetime.fromisoformat(row["timestamp"])
            u = float(row["u"])
            v = float(row["v"])
            wind_map[h3_cell][timestamp] = (u, v)

    return dict(wind_map)



