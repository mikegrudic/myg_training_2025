"""Add out-and-back side trips to summits to a route you like, keeping the route itself.

  python spurify.py ROUTE.gpx --extra 3          # allow up to 3 more miles
  python spurify.py ROUTE.gpx --budget 24        # total distance, miles

The route is matched onto OpenStreetMap trails; the solver may then add trail run out and back from it, turning
around only at named peaks (and wherever the route already turns around), to maximize gain (3DEP, 50 m smoothing).
Writes ROUTE_spurred.gpx and prints the side trips it added.
"""
import argparse, glob, os, re, sys
import networkx as nx
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import max_gain_route as mg

FT = mg.M_TO_FT


def read_gpx(path):
    t = open(path).read()
    p = re.findall(r'<(?:trkpt|rtept)[^>]*lat="([-\d.]+)"[^>]*lon="([-\d.]+)"', t)
    p = p or [(a, b) for b, a in re.findall(r'<(?:trkpt|rtept)[^>]*lon="([-\d.]+)"[^>]*lat="([-\d.]+)"', t)]
    return np.array(p, float)


def along(pts):
    lat0 = np.radians(pts[:, 0].mean())
    return np.r_[0, np.cumsum(np.hypot(np.diff(pts[:, 1]) * 111320 * np.cos(lat0), np.diff(pts[:, 0]) * 110540))]


def match(edges, pts, tol):
    """Times the track runs each edge: the junctions it passes within ``tol`` (m), in order, joined by the shortest
    trail between them, so the result is always one continuous walk."""
    lat0 = np.radians(pts[:, 0].mean())
    G = nx.MultiGraph()
    xy = {}
    for k, e in enumerate(edges):
        G.add_edge(e["u"], e["v"], key=k, w=e["length"])
        for n, i in ((e["u"], 0), (e["v"], -1)):
            xy[n] = (e["lon"][i] * 111320 * np.cos(lat0), e["lat"][i] * 110540)
    nodes = list(xy)
    nxy = np.array([xy[n] for n in nodes])
    d = along(pts)
    s = np.arange(0, d[-1], 5.0)
    trk = np.c_[np.interp(s, d, pts[:, 1]) * 111320 * np.cos(lat0), np.interp(s, d, pts[:, 0]) * 110540]
    seq = []
    for p in trk:
        dd = np.hypot(*(nxy - p).T)
        i = int(dd.argmin())
        if dd[i] <= tol and (not seq or seq[-1] != nodes[i]):
            seq.append(nodes[i])
    counts = np.zeros(len(edges), int)
    for a, b in zip(seq, seq[1:]):
        path = nx.shortest_path(G, a, b, weight="w")
        for u, v in zip(path, path[1:]):
            counts[min(G[u][v], key=lambda k: G[u][v][k]["w"])] += 1
    if (counts > 2).any():
        print(f"Warning: the route runs {(counts > 2).sum()} trail sections more than twice; counted as twice.")
    return np.minimum(counts, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gpx")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--extra", type=float, help="miles that may be added")
    g.add_argument("--budget", type=float, help="total miles")
    ap.add_argument("--out", help="output GPX (default ROUTE_spurred.gpx)")
    ap.add_argument("--time", type=float, default=60, help="solver seconds (default 60)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--match-m", type=float, default=15, help="how close the track must follow a trail (m)")
    ap.add_argument("--summit-m", type=float, default=60, help="how close a trail must pass a peak (m)")
    args = ap.parse_args()

    pts = read_gpx(args.gpx)
    L0 = along(pts)[-1]
    budget = args.budget * mg.MI_TO_M if args.budget else L0 + args.extra * mg.MI_TO_M
    reach = max(budget - L0, 0) / 2 + 300  # side trips go out and back
    closed_loop = mg._haversine(*pts[0], *pts[-1]) < 100
    centers = pts[np.linspace(0, len(pts) - 1, max(2, int(L0 / max(reach, 500)) + 2)).astype(int)]

    print(f"Route {L0 / mg.MI_TO_M:.2f} mi; budget {budget / mg.MI_TO_M:.2f} mi; fetching trails and peaks...")
    osm = mg.fetch_osm([tuple(c) for c in centers], reach + 200, True)
    q = "[out:json][timeout:180];(" + "".join(
        f'node["natural"="peak"]["name"](around:{reach + 200:.0f},{la:.6f},{lo:.6f});' for la, lo in centers) + ");out;"
    peaks = {x["id"]: (x["lat"], x["lon"], x["tags"]["name"]) for x in mg._overpass(q, "peaks")["elements"]}
    anchors = [tuple(pts[0]), tuple(pts[-1])] + [(la, lo) for la, lo, _ in peaks.values()]
    closures = mg.load_closures(sorted(glob.glob(os.path.join(os.path.dirname(mg.__file__), "closures", "*.json"))))
    raw, ids = mg.build_graph(osm, True, None, anchors, closed=closures)
    at = {}
    for e in raw:
        at[e["u"]], at[e["v"]] = e["latlon"][0], e["latlon"][-1]
    summit = {}
    for n, (la, lo, name) in zip(ids[2:], peaks.values()):
        if n in at and mg._haversine(*at[n], la, lo) <= args.summit_m:
            summit[n] = name
    start, end = ids[0], (ids[0] if closed_loop else ids[1])
    edges = mg.contract([dict(e) for e in raw], {start, end} | set(summit))
    mg.add_elevation(edges, 50.0, "3dep")
    edges = mg.subdivide(edges, None)  # per-edge climb, no splitting

    base = match(edges, pts, args.match_m)
    L_base = sum(e["length"] * c for e, c in zip(edges, base))
    if abs(L_base - L0) > 0.05 * L0:
        print(f"Warning: the route matched {L_base / mg.MI_TO_M:.2f} mi of trail for a {L0 / mg.MI_TO_M:.2f} mi track; "
              "try a larger --match-m.")
    # The route's own turnarounds stay allowed; anything new turns around only at a summit.
    G = nx.MultiGraph()
    for e, c in zip(edges, base):
        if c:
            G.add_edge(e["u"], e["v"])
    tips = {n for n in G if G.degree(n) == 1}
    for e, c in zip(edges, base):
        if c:
            e["require"] = int(c)
        else:
            e["spur_only"] = True
    topo = dict(loops=None, spurs=None, start=None, reuse=True)
    m, s, t, proven = mg.solve(edges, [start], None if closed_loop else [end], budget, topo, 0, args.time, args.workers,
                               False, hint=base, no_turnarounds=True, turnaround_ok=set(summit) | tips)

    gain = lambda counts: float(np.sum(np.maximum(0, np.diff(mg.assemble(edges, counts, start, end)["z"])))) * FT
    route = mg.assemble(edges, np.asarray(m), start, end)
    g0, g1 = gain(base), float(np.sum(np.maximum(0, np.diff(route["z"])))) * FT
    out = args.out or re.sub(r"\.gpx$", "", args.gpx) + "_spurred.gpx"
    mg.write_gpx(out, route, os.path.basename(out)[:-4])

    # Side trips: the added edges (all run out and back), grouped by where they leave the route.
    added = [k for k in range(len(edges)) if m[k] and not base[k]]
    H = nx.Graph()
    for k in added:
        H.add_edge(edges[k]["u"], edges[k]["v"], k=k)
    on_route = set(G)
    d_base = along(pts)
    lat0 = np.radians(pts[:, 0].mean())
    print(f"\n{'Leaves route':>12s}  {'Out and back':>12s}  {'Gain':>8s}  {'ft/mi':>6s}  Summits")
    for comp in sorted(nx.connected_components(H), key=lambda c: -len(c)):
        ks = [H[u][v]["k"] for u, v in H.subgraph(comp).edges]
        length = 2 * sum(edges[k]["length"] for k in ks)
        g = sum(edges[k]["var"] for k in ks) * FT  # out and back: every metre up or down is climbed once
        junction = next((n for n in comp if n in on_route), None)
        mi = float("nan")
        if junction is not None:
            la, lo = at[junction]
            mi = d_base[np.argmin(np.hypot((pts[:, 0] - la) * 110540, (pts[:, 1] - lo) * 111320 * np.cos(lat0)))] / mg.MI_TO_M
        names = ", ".join(summit[n] for n in comp if n in summit) or "(connector, no summit)"
        print(f"{mi:9.2f} mi  {length / mg.MI_TO_M:9.2f} mi  {g:6,.0f} ft  {g / (length / mg.MI_TO_M):6,.0f}  {names}")
    if not added:
        print("  (none: no summit side trip fits the budget)")
    print(f"\nBefore: {g0:,.0f} ft over {L_base / mg.MI_TO_M:.2f} mi.  After: {g1:,.0f} ft over "
          f"{route['dist'][-1] / mg.MI_TO_M:.2f} mi ({g1 - g0:+,.0f} ft){'; optimal' if proven else ''}.  Wrote {out}")


if __name__ == "__main__":
    main()
