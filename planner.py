"""Core logic for the Martian Map: one 'brain' that works out how long it takes
to walk from home base to every spot on the map, avoiding steep ground."""
import heapq
import json
import math

import numpy as np

MARS_RADIUS_M = 3396190.0


def load_grid(path="jezero_grid.json"):
    with open(path) as f:
        d = json.load(f)
    d["elev"] = np.array(d["elevation_m"], dtype=float)
    d["slope"] = np.array(d["slope_deg"], dtype=float)
    return d


def latlon_to_rc(grid, lat, lon):
    b = grid["bbox"]
    r = (b["lat_max"] - lat) / (b["lat_max"] - b["lat_min"]) * grid["rows"]
    c = (lon - b["lon_min"]) / (b["lon_max"] - b["lon_min"]) * grid["cols"]
    r, c = int(r), int(c)
    if not (0 <= r < grid["rows"] and 0 <= c < grid["cols"]):
        raise ValueError("That spot is outside the map.")
    return r, c


def rc_to_latlon(grid, r, c):
    b = grid["bbox"]
    lat = b["lat_max"] - (r + 0.5) / grid["rows"] * (b["lat_max"] - b["lat_min"])
    lon = b["lon_min"] + (c + 0.5) / grid["cols"] * (b["lon_max"] - b["lon_min"])
    return lat, lon


def travel_times(grid, home_rc, speed_kmh=3.0, max_slope=20.0, max_hours=8.0):
    """Dijkstra from home: shortest TIME (hours) to reach every cell.
    Steeper ground = slower; ground steeper than max_slope = impassable.
    Returns (hours_grid, came_from) -- came_from lets us rebuild any route."""
    rows, cols = grid["rows"], grid["cols"]
    slope = grid["slope"]
    cell_m = grid["cell_size_m"]
    b = grid["bbox"]
    lats = b["lat_max"] - (np.arange(rows) + 0.5) / rows * (b["lat_max"] - b["lat_min"])
    ew_scale = np.cos(np.radians(lats))  # east-west cells get narrower toward the poles

    def speed_factor(s):
        return max(0.15, 1.0 - s / max_slope)

    inf = float("inf")
    hours = np.full((rows, cols), inf)
    came_from = np.full((rows, cols, 2), -1, dtype=int)
    hr, hc = home_rc
    hours[hr, hc] = 0.0
    heap = [(0.0, hr, hc)]
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while heap:
        t, r, c = heapq.heappop(heap)
        if t > hours[r, c]:
            continue
        if t > max_hours:
            break
        for dr, dc in nbrs:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if slope[nr, nc] >= max_slope:
                continue
            dy = abs(dr) * cell_m
            dx = abs(dc) * cell_m * ew_scale[r]
            dist_km = math.hypot(dx, dy) / 1000.0
            f = speed_factor((slope[r, c] + slope[nr, nc]) / 2.0)
            nt = t + dist_km / (speed_kmh * f)
            if nt < hours[nr, nc]:
                hours[nr, nc] = nt
                came_from[nr, nc] = (r, c)
                heapq.heappush(heap, (nt, nr, nc))
    return hours, came_from


def route_to(came_from, home_rc, target_rc):
    path = [tuple(target_rc)]
    cur = tuple(target_rc)
    while cur != tuple(home_rc):
        prev = tuple(came_from[cur])
        if prev[0] < 0:
            return None  # not reachable
        path.append(prev)
        cur = prev
    return path[::-1]


def hillshade(elev, cell_m, azimuth=315, altitude=45):
    az, alt = math.radians(azimuth), math.radians(altitude)
    gy, gx = np.gradient(elev, cell_m)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(shade, 0, 1)
