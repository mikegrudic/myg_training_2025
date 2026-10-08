"""Per cell, the best route valid under the current road-walk rules, from the given run folders (first = preferred
on ties). Routes over the closed 9D stretch at Breakneck or an abandoned way are dropped."""
import glob, json, os, re, shutil, sys
import numpy as np
CLOSURES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "closures")
flags = {a for a in sys.argv[1:] if a.startswith("--")}
LOWEST = "--lowest" in flags  # keep the flattest valid route per cell instead
ROAD_RUNS = "--road-runs" in flags  # also reject the road-run exclusions (Manitou campus, Foundry Preserve climbs)
out, *srcs = [a for a in sys.argv[1:] if not a.startswith("--")]
pairs = json.load(open(os.path.join(CLOSURES, "breakneck_9d_2026-10.json")))["closed_segments"]
osm_nodes = json.load(open(os.path.join(os.path.dirname(__file__), "breakneck_9d_nodes.json")))
mids = np.array([[(osm_nodes[str(a)][0] + osm_nodes[str(b)][0]) / 2, (osm_nodes[str(a)][1] + osm_nodes[str(b)][1]) / 2] for a, b in pairs])
# Abandoned ways (an OSM end_date in the past), anywhere.
ab_pairs = json.load(open(os.path.join(CLOSURES, "abandoned_ways_2026-10.json")))["closed_segments"]
ab_nodes = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "abandoned_nodes.json")))
ab_mids = np.array([[(ab_nodes[str(a)][0] + ab_nodes[str(b)][0]) / 2, (ab_nodes[str(a)][1] + ab_nodes[str(b)][1]) / 2] for a, b in ab_pairs])

SCHOOL = np.array([[41.4192, -73.9423], [41.41929, -73.94202]])  # Manitou School's drive off 9D (road runs)
# The steep climbs out of the Foundry Preserve (road runs).
SCHOOL = np.vstack([SCHOOL, np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "foundry_climbs.npy"))])
# The private drive from the top of Moffatt Road to Healy Road (road runs).
SCHOOL = np.vstack([SCHOOL, [[41.42404, -73.93610], [41.42549, -73.93495], [41.42752, -73.93611]]])


def uses_closed(gpx):
    pts = np.array(re.findall(r'lat="([-\d.]+)" lon="([-\d.]+)"', open(gpx).read()), float)
    if ROAD_RUNS and (np.hypot((pts[None, :, 0] - SCHOOL[:, None, 0]) * 110540, (pts[None, :, 1] - SCHOOL[:, None, 1]) * 83000).min(1) < 8).any():
        return True
    near = (np.abs(pts[:, None, 0] - ab_mids[None, :, 0]) < 0.001) & (np.abs(pts[:, None, 1] - ab_mids[None, :, 1]) < 0.0013)
    if near.any():
        d = np.hypot((pts[None, :, 0] - ab_mids[:, None, 0]) * 110540, (pts[None, :, 1] - ab_mids[:, None, 1]) * 80000).min(1)
        if (d < 8).sum() >= 2:
            return True
    if not ((np.abs(pts[:, 0] - 41.443) < 0.01) & (np.abs(pts[:, 1] + 73.977) < 0.01)).any():
        return False
    d = np.hypot((pts[None, :, 0] - mids[:, None, 0]) * 110540, (pts[None, :, 1] - mids[:, None, 1]) * 83000).min(1)
    return (d < 8).sum() >= 3

EXCLUDED_END_ROADS = {"Mount Washington Auto Road", "Breakneck Road"}  # one-way routes may not end on these
sites = sorted({p.split("/")[-3] for s in srcs for p in glob.glob(f"{s}/*/cells/*.json")})
dropped = 0
for site in sites:
    os.makedirs(f"{out}/{site}/cells", exist_ok=True)
    os.makedirs(f"{out}/{site}/gpx", exist_ok=True)
    net = next((s for s in srcs if os.path.exists(f"{s}/{site}/network_p2p.json")), None)
    for f in ("network.json", "network_p2p.json"):
        if net:
            shutil.copy(f"{net}/{site}/{f}", f"{out}/{site}/{f}")
    keys = sorted({os.path.basename(p) for s in srcs for p in glob.glob(f"{s}/{site}/cells/*.json")})
    for k in keys:
        best = None
        for s in srcs:
            p = f"{s}/{site}/cells/{k}"
            if not os.path.exists(p):
                continue
            c = json.load(open(p))
            if "none" not in c and (uses_closed(f"{s}/{site}/{c['gpx']}")
                                    or c.get("end", "").split(" at ")[-1] in EXCLUDED_END_ROADS):
                dropped += 1
                continue
            g = -1 if "none" in c else c["gain_ft"]
            if LOWEST and g >= 0:
                g = -g  # flatter is better
            elif LOWEST:
                g = -float("inf")
            if best is None or g > best[0]:
                best = (g, s, c)
        if best is None:
            continue
        g, s, c = best
        json.dump(c, open(f"{out}/{site}/cells/{k}", "w"))
        if "none" not in c:
            shutil.copy(f"{s}/{site}/{c['gpx']}", f"{out}/{site}/{c['gpx']}")
print(f"{len(sites)} sites merged; {dropped} routes over closed or excluded segments dropped")
