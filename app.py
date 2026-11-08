"""Martian Map -- EVA planner for Jezero Crater. Run with:  streamlit run app.py"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from planner import hillshade, latlon_to_rc, load_grid, rc_to_latlon, route_to, travel_times

st.set_page_config(page_title="Martian Map", layout="wide")
st.title("Martian Map: plan a safe Marswalk")
st.caption("Real NASA elevation data (MOLA/HRSC) for Jezero Crater. "
           "Pick a start and a destination, and we plan the route and check the astronaut can get back.")


@st.cache_data
def get_grid():
    g = load_grid("jezero_grid.json")
    g["shade"] = hillshade(g["elev"], g["cell_size_m"])
    return g


@st.cache_data
def get_times(home_rc, speed, max_slope, max_hours):
    g = get_grid()
    return travel_times(g, home_rc, speed, max_slope, max_hours)


g = get_grid()
b = g["bbox"]

with st.sidebar:
    st.header("Mission settings")
    st.caption("These numbers are adjustable assumptions, not official NASA values.")
    speed = st.number_input("Walking speed on flat ground (km/h)", 0.5, 20.0, 3.0, 0.5)
    hours = st.number_input("Total time outside (hours)", 1.0, 24.0, 8.0, 0.5)
    reserve = st.number_input("Safety reserve (hours)", 0.0, 4.0, 1.0, 0.5)
    max_slope = st.number_input("Steepest slope we allow (degrees)", 5.0, 40.0, 20.0, 1.0)
    st.header("Where")
    home_lat = st.number_input("Base latitude", b["lat_min"], b["lat_max"], 18.4446, 0.01, format="%.4f")
    home_lon = st.number_input("Base longitude", b["lon_min"], b["lon_max"], 77.4509, 0.01, format="%.4f")
    tgt_lat = st.number_input("Destination latitude", b["lat_min"], b["lat_max"], 18.50, 0.01, format="%.4f")
    tgt_lon = st.number_input("Destination longitude", b["lon_min"], b["lon_max"], 77.38, 0.01, format="%.4f")

budget = hours - reserve
home = latlon_to_rc(g, home_lat, home_lon)
target = latlon_to_rc(g, tgt_lat, tgt_lon)
t_grid, came = get_times(home, speed, max_slope, budget)

# Safe zone = places where there-and-back fits inside the time budget
safe = (2 * t_grid) <= budget
path = route_to(came, home, target)

# ---- verdict ----
if path is None or not np.isfinite(t_grid[target]):
    st.error("No safe route to this destination within the time budget (blocked by steep ground or too far).")
else:
    one_way = t_grid[target]
    round_trip = 2 * one_way
    pts = np.array(path)
    max_s = g["slope"][pts[:, 0], pts[:, 1]].max()
    if round_trip <= budget:
        st.success(f"GO. One way {one_way:.1f} h, round trip {round_trip:.1f} h. "
                   f"{hours - round_trip:.1f} h left over (including the {reserve:.1f} h reserve). "
                   f"Steepest slope on route: {max_s:.0f} degrees.")
    else:
        st.warning(f"NO-GO. Round trip would take {round_trip:.1f} h, but only {budget:.1f} h are available "
                   f"after the safety reserve. The astronaut could not safely return.")

# ---- map ----
lats = [rc_to_latlon(g, r, 0)[0] for r in range(g["rows"])]
lons = [rc_to_latlon(g, 0, c)[1] for c in range(g["cols"])]
fig = go.Figure()
fig.add_trace(go.Heatmap(z=g["shade"], x=lons, y=lats, colorscale="Greys", showscale=False,
                         hoverinfo="skip"))
fig.add_trace(go.Heatmap(z=np.where(g["slope"] >= max_slope, 1.0, np.nan), x=lons, y=lats,
                         colorscale=[[0, "red"], [1, "red"]], opacity=0.55, showscale=False,
                         hoverinfo="skip"))
fig.add_trace(go.Heatmap(z=np.where(safe, 1.0, np.nan), x=lons, y=lats,
                         colorscale=[[0, "lime"], [1, "lime"]], opacity=0.25, showscale=False,
                         hoverinfo="skip"))
if path:
    ll = [rc_to_latlon(g, r, c) for r, c in path]
    fig.add_trace(go.Scatter(x=[p[1] for p in ll], y=[p[0] for p in ll], mode="lines",
                             line=dict(color="gold", width=4), name="Route"))
fig.add_trace(go.Scatter(x=[home_lon], y=[home_lat], mode="markers+text", text=["Base"],
                         textposition="top center", marker=dict(size=12, color="cyan"), name="Base"))
fig.add_trace(go.Scatter(x=[tgt_lon], y=[tgt_lat], mode="markers+text", text=["Destination"],
                         textposition="top center", marker=dict(size=12, color="orange"), name="Destination"))
fig.update_layout(height=700, xaxis_title="Longitude (E)", yaxis_title="Latitude (N)",
                  yaxis=dict(scaleanchor="x"), margin=dict(l=10, r=10, t=10, b=10))
try:
    st.plotly_chart(fig, width="stretch")
except TypeError:  # older Streamlit versions
    st.plotly_chart(fig, use_container_width=True)
st.markdown("**Green** = places the astronaut can reach AND return from in time. "
            "**Red** = too steep to walk. **Yellow line** = planned route.")
