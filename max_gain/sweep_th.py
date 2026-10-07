"""Max-gain sweeps over trailheads x shapes x distance budgets, writing page cells and GPX.

  python sweep_th.py prepare                  # fetch OSM/DEM for every trailhead, one at a time
  python sweep_th.py run JOBS WORKERS PROCS   # JOBS: slug:shape:budgetset,... run by PROCS processes
"""
import collections, contextlib, glob, io, json, math, os, re, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.environ.get("MAX_GAIN_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import networkx as nx, numpy as np, max_gain_route as mg

OUT = os.environ.get("MAX_GAIN_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "sites"))
ROAD_TIME_FRAC = 0.10  # roads may take up to this share of a route's grade-adjusted time
CLOSED = mg.load_closures(sorted(glob.glob(os.path.join(os.path.dirname(mg.__file__), "closures", "*.json"))))
TRAILHEADS = {
    "washburn": ("Washburn Trailhead", "Cold Spring, NY", (41.42698, -73.96568)),
    "prediger": ("Prediger Road Trailhead", "Catskills, Hunter, NY", (42.13410, -74.10425)),
    "woodland-valley": ("Woodland Valley Campground", "Catskills, Phoenicia, NY", (42.03545, -74.35961)),
    "anthony-wayne": ("Anthony Wayne Recreation Area", "Harriman State Park, NY", (41.29636, -74.02785)),
    "bear-mountain": ("Bear Mountain Inn parking", "Bear Mountain State Park, NY", (41.31193, -73.98874)),
    "harding-road": ("Harding Road Trailhead", "Catskills, Palenville, NY", (42.17632, -74.03041)),
    "windham-escarpment": ("Escarpment Trail, Windham", "Catskills, Windham, NY", (42.31286, -74.18985)),
    "north-south-lake": ("North-South Lake Campground", "Catskills, Haines Falls, NY", (42.20057, -74.04509)),
    "barnum-road": ("Barnum Road Trailhead", "Catskills, Maplecrest, NY", (42.26398, -74.17667)),
    "moon-haw": ("Moon Haw Road Trailhead", "Catskills, West Shokan, NY", (41.98436, -74.32758)),
    "beacon-casino": ("Mount Beacon, Casino Trail", "Beacon, NY", (41.49353, -73.95998)),
    "wilkinson-east": ("Wilkinson Memorial Trail, east end", "Fishkill, NY", (41.48855, -73.91136)),
    "nelsonville": ("Nelsonville Trailhead", "Nelsonville, NY", (41.42347, -73.95237)),
    "mikes-house": ("Mike's house", "Cold Spring, NY", (41.41644, -73.95609)),
    "central-park": ("Central Park, base of Harlem Hill", "New York, NY", (40.79934, -73.95550)),
    "central-park-south": ("Central Park, 7th Avenue entrance", "New York, NY", (40.76703, -73.97900)),
    "prospect-park": ("Prospect Park, Grand Army Plaza", "Brooklyn, NY", (40.67180, -73.97060)),
    "inwood-215": ("215 St station (1), Inwood Hill and Fort Tryon parks", "New York, NY", (40.86957, -73.91519)),
    "fort-tryon-190": ("190 St station (A), Fort Tryon and Inwood Hill parks", "New York, NY", (40.85890, -73.93399)),
    "dyckman-a": ("Dyckman St station (A), between Fort Tryon and Inwood Hill parks", "New York, NY", (40.86528, -73.92780)),
    "daniel-webster": ("Daniel Webster trailhead, Dolly Copp", "White Mountains, Gorham, NH", (44.32596, -71.22000)),
    "pinkham-notch": ("Pinkham Notch Visitor Center", "White Mountains, NH", (44.25646, -71.25334)),
    "crawford-notch": ("Crawford Notch Depot (Highland Center)", "White Mountains, NH", (44.21787, -71.41128)),
    "lincoln-woods": ("Lincoln Woods trailhead", "White Mountains, Lincoln, NH", (44.06354, -71.58810)),
    "valley-way": ("Valley Way trailhead, Appalachia", "White Mountains, Randolph, NH", (44.37152, -71.28933)),
    "big-cypress-oasis": ("Oasis Visitor Center, Big Cypress", "Ochopee, FL", (25.85618, -81.03380)),
    "stowe": ("Stowe, Mansfield Base Lodge", "Stowe, VT", (44.52903, -72.78443)),
    "underhill": ("Underhill State Park", "Underhill, VT", (44.52759, -72.84278)),
    "giant-ridge": ("Giant Mountain Ridge Trail", "Adirondacks, Keene, NY", (44.13816, -73.74371)),
    "cold-spring-bandstand": ("Cold Spring waterfront bandstand", "Cold Spring, NY", (41.41602, -73.96122)),
    "fishkill-ridge": ("Fishkill Ridge Trailhead", "Beacon, NY", (41.50780, -73.92846)),
    "storm-king": ("Storm King, Highlands Trail parking", "Cornwall, NY", (41.43976, -74.00688)),
    "adk-loj": ("Adirondak Loj (High Peaks Information Center)", "Adirondacks, Lake Placid, NY", (44.18287, -73.96307)),
    "garden": ("The Garden trailhead", "Adirondacks, Keene Valley, NY", (44.18904, -73.81595)),
    "rooster-comb": ("Rooster Comb trailhead", "Adirondacks, Keene Valley, NY", (44.18550, -73.78671)),
    "hubbard-lodge": ("Hubbard Lodge", "Hudson Highlands State Park, NY", (41.44343, -73.91483)),
}
RACES = [(3.107, "5K"), (6.214, "10K"), (13.109, "Half"), (26.219, "Marathon"), (31.069, "50K")]
# Whole miles, except those next to a race distance (3 by the 5K, 6 by the 10K, 13 by the half, 26 by the marathon,
# 31 by the 50K).
BUDGETS = {"ints": [(float(d), str(d)) for d in range(4, 31) if d not in (6, 13, 26)], "races": RACES}
BUDGETS["all"] = sorted(BUDGETS["ints"] + RACES)
DMAX, MIN_LOOP, MIN_LOOP_FRAC, MIN_END = 31, 1.0 * mg.MI_TO_M, 0.25, 1.0 * mg.MI_TO_M
TIME_LIMIT = float(os.environ.get("SWEEP_TIME_LIMIT", 120))  # s per cell
SPUR_MAX = 1.0 * mg.MI_TO_M  # free access path from the lot to the nearest loop
WARM_LIMIT = 30  # s for the trail-only warm start when no earlier route can seed the solver


# Road walks are capped per row at under 10% of its nominal distance; graphs are built for these tiers (miles).
ROAD_TIERS_MI = [0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.1]
# Trails-only edition: roads may only be crossed, i.e. road links up to this length between two trail ends.
CROSSINGS_ONLY = os.environ.get("SWEEP_ROADS") == "crossings"
CROSSING_MI = 50.0 / mg.MI_TO_M
# Road-run edition: public roads, paved or not (no trails, driveways or parking aisles; major roads only crossed),
# plus these ways, which count as road-run routes: Cold Spring Cemetery lanes, the West Point Foundry Preserve
# paths and the Main Street railroad tunnel.
ROAD_RUNS = os.environ.get("SWEEP_ROADS") == "roads"
MINIMIZE = os.environ.get("SWEEP_OBJECTIVE") == "min"  # the route climbing the least, at least MIN_FILL of the row
MIN_FILL = 0.98
ROAD_RUN_EXTRA_WAYS = {
    295688401, 295688402, 295688403, 295688405, 295688407, 295688408, 295688409, 295688410, 295688411, 329378033,
    1165285748, 1287157241,  # cemeteries
    24170448, 97522214, 298020412, 298020413, 298020414, 298020415, 298020416, 298020417, 298020418, 298020419,
    298020420, 298020421, 298020422, 339748534, 339748538, 339748540, 339748541, 339748543, 339748544, 339748545,
    339748547, 339748548, 339748549, 339748551, 339748555, 339748556, 339748558, 339748559, 432386609, 432386612,
    432386616, 1287164047, 1287164048, 1287164049, 1287164050, 1287164051, 1287164052, 1287164053, 1287216454,
    1287216455, 1287216456, 1287216457, 1287216458, 1287216459, 1287216460, 1287216461, 1287216462, 1287216463,
    1287216464, 1287216465, 1287232106, 1287232107, 1287232109, 1287232110, 1287232111, 1287232112,  # foundry
    1287153001, 1287153000, 1287153031, 1287153032,  # the pedestrian tunnel under the tracks at the foot of Main Street
}


def sidewalk_links(edges, max_m):
    """Straight connectors from dead ends of the extra road-run ways to the nearest road junction within ``max_m``,
    standing in for the sidewalks (dropped from the graph) that join them to the street."""
    pos, deg = {}, collections.Counter()
    for e in edges:
        pos[e["u"]], pos[e["v"]] = e["latlon"][0], e["latlon"][-1]
        deg[e["u"]] += 1
        deg[e["v"]] += 1
    road = sorted({n for e in edges if e["road"] for n in (e["u"], e["v"])})
    xy = np.array([pos[n] for n in road])
    links = []
    for e in edges:
        if e.get("way") not in ROAD_RUN_EXTRA_WAYS:
            continue
        for n in (e["u"], e["v"]):
            if deg[n] == 1:
                d = mg._haversine(*pos[n], xy[:, 0], xy[:, 1])
                i = int(np.argmin(d))
                if d[i] <= max_m:
                    ll = np.array([pos[n], pos[road[i]]])
                    links.append(dict(u=n, v=road[i], latlon=ll, names=np.full(2, "sidewalk", dtype=object),
                                      roadpt=np.zeros(2, bool), length=max(float(d[i]), 0.2), road=False, walk=True))
    return links


# Road-run starts close enough to seed each other (a route there plus the walk to it and back).
ROAD_RUN_NEIGHBORS = {"cold-spring-bandstand": ["mikes-house"], "mikes-house": ["cold-spring-bandstand"]}


ROAD_RUN_EXCLUDED_WAYS = {
    329378036, 329378037, 795381392, 329378046,  # Manitou School's drive off 9D (private)
    298020420, 298020413, 298020414, 298020421,  # steep climbs out of the Foundry Preserve (west, east)
    192418583, 192418575,  # unnamed drive from the top of Moffatt Road to Healy Road (private land)
}
ROAD_RUN_EXCLUDED_EDGES = {  # OSM node pairs: the steep top of the north climb out of the Foundry Preserve
    frozenset((3019334326, 3019341652)), frozenset((3019334322, 3019334326)),
}


def road_run_ok(e):
    """Whether a road-run route may use this edge: road runs must suit a broad group."""
    if e.get("way") in ROAD_RUN_EXCLUDED_WAYS or frozenset((e["u"], e["v"])) in ROAD_RUN_EXCLUDED_EDGES:
        return False
    if e.get("way") in ROAD_RUN_EXTRA_WAYS:
        return True
    return e["road"] and e["walk"] and e.get("service") not in ("driveway", "parking_aisle", "drive-through")


def road_tier(budget_mi):
    """Largest road-walk tier within 10% of the row's distance."""
    if CROSSINGS_ONLY:
        return CROSSING_MI
    return max([t for t in ROAD_TIERS_MI if t <= 0.1 * budget_mi + 1e-9], default=ROAD_TIERS_MI[0])


# Sites confined to an area: only ways inside it (plus a margin for its edge streets), roads and paths alike with no
# road-walk limit, including car-free drives tagged pedestrian/private; sunken transverses excluded.
SITE_AREAS = {
    "central-park": dict(center=(40.7829, -73.9654), radius_mi=2.6, margin_m=40.0, periphery_m=30.0, exclude_names=("Transverse",),
                         corners=[(40.7644, -73.9730), (40.7681, -73.9819), (40.8003, -73.9580), (40.7968, -73.9496)]),
    "central-park-south": None,  # set below: the same area as central-park
    # Fort Tryon and Inwood Hill parks (OSM outlines, simplified) with Isham Park, the streets between them (within
    # the margin) and the station approaches; the rest of the street grid is left out.
    "uptown-parks": dict(center=(40.8665, -73.9265), radius_mi=1.2, margin_m=40.0, periphery_m=30.0,
                         exclude_names=("Henry Hudson Parkway",), polys=[
        [(40.86526, -73.92808), (40.86463, -73.92922), (40.86369, -73.93012), (40.86204, -73.93022), (40.85948, -73.93144), (40.85963, -73.93232), (40.85729, -73.93355), (40.85744, -73.93402), (40.85834, -73.9337), (40.85854, -73.9344), (40.85916, -73.93404), (40.85908, -73.93379), (40.85931, -73.93368), (40.85961, -73.93381), (40.85971, -73.93413), (40.85962, -73.93447), (40.85946, -73.93454), (40.85944, -73.93497), (40.85902, -73.93555), (40.85716, -73.93684), (40.85753, -73.93768), (40.85904, -73.93644), (40.8597, -73.9354), (40.86006, -73.93511), (40.86126, -73.9348), (40.86314, -73.93365), (40.86505, -73.9329), (40.86662, -73.93192), (40.86678, -73.93157), (40.86677, -73.93113), (40.86578, -73.92986), (40.86561, -73.92935), (40.86549, -73.92823)],
        [(40.87425, -73.92063), (40.87385, -73.91998), (40.87339, -73.91996), (40.87301, -73.92045), (40.87301, -73.92093), (40.87317, -73.92149), (40.87361, -73.92208), (40.87426, -73.92255), (40.87512, -73.92269), (40.8759, -73.9224), (40.87662, -73.92238), (40.87686, -73.9225), (40.87726, -73.92325), (40.87745, -73.92472), (40.8774, -73.92506), (40.87708, -73.92563), (40.87707, -73.92613), (40.87718, -73.92642), (40.87733, -73.92646), (40.87751, -73.92628), (40.87761, -73.92651), (40.8768, -73.92739), (40.87659, -73.92806), (40.87554, -73.9292), (40.87115, -73.93192), (40.86985, -73.93221), (40.86689, -73.92867), (40.86843, -73.92645), (40.86862, -73.926), (40.86906, -73.92396), (40.86961, -73.92232), (40.86932, -73.92205), (40.87105, -73.91937), (40.87213, -73.91977), (40.8731, -73.91847), (40.87337, -73.919), (40.87367, -73.91928), (40.87416, -73.91921), (40.87466, -73.91888), (40.87548, -73.92089), (40.87511, -73.92119), (40.8744, -73.92069), (40.87428, -73.92066), (40.8742, -73.92088)],
        [(40.8709, -73.91922), (40.86987, -73.92071), (40.86898, -73.92015), (40.8691, -73.91977), (40.86841, -73.9193), (40.86917, -73.91744), (40.86971, -73.91802), (40.86996, -73.91745), (40.8706, -73.91823), (40.87036, -73.91859)],
        [(40.8686, -73.9178), (40.8686, -73.9146), (40.8703, -73.9146), (40.8703, -73.9178)],  # to the 215 St station
        [(40.8583, -73.9348), (40.8583, -73.9322), (40.8596, -73.9322), (40.8596, -73.9348)],  # the 190 St entrances
    ]),
    "prospect-park": dict(center=(40.6620, -73.9690), radius_mi=1.2, margin_m=40.0, periphery_m=60.0, exclude_names=(),
                          corners=[(40.6735, -73.9699), (40.6627, -73.9617), (40.6552, -73.9616), (40.6512, -73.9719),
                                   (40.6612, -73.9799)]),
}
SITE_AREAS["central-park-south"] = SITE_AREAS["central-park"]
SITE_AREAS["inwood-215"] = SITE_AREAS["fort-tryon-190"] = SITE_AREAS["dyckman-a"] = SITE_AREAS.pop("uptown-parks")
# Area sites whose one-way routes may also end at another start (e.g. the other subway station).
UPTOWN_STATIONS = ['inwood-215', 'fort-tryon-190', 'dyckman-a']
AREA_EXTRA_ENDS = {s: [t for t in UPTOWN_STATIONS if t != s] for s in UPTOWN_STATIONS}
# Sites whose loops (from any seed directory) also seed this site's, when they pass its start.
SHARED_LOOP_SITES = {"central-park": ["central-park-south"], "central-park-south": ["central-park"],
                     **{s: [t for t in UPTOWN_STATIONS if t != s] for s in UPTOWN_STATIONS}}
BASE_TRAIL_HIGHWAYS = list(mg.TRAIL_HIGHWAYS)


def area_edge(lat, lon, area):
    """Whether a point is inside the area's polygon, and its distance (m) from the polygon's edge."""
    xy = lambda a, b: np.array([(b - area["center"][1]) * 111320 * math.cos(math.radians(area["center"][0])),
                                (a - area["center"][0]) * 110540])
    p, inside, d = xy(lat, lon), False, math.inf
    for corners in area.get("polys", [area.get("corners")]):  # a union of polygons, or one
        poly, odd = [xy(*c) for c in corners], False
        for a, b in zip(poly, poly[1:] + poly[:1]):
            if (a[1] > p[1]) != (b[1] > p[1]) and p[0] < a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]):
                odd = not odd
            t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1)
            d = min(d, float(np.linalg.norm(a + t * (b - a) - p)))
        inside |= odd
    return inside, d


def in_area(lat, lon, area):
    """Whether a point is inside the area's polygon or within its margin of the edge."""
    inside, d = area_edge(lat, lon, area)
    return inside or d <= area["margin_m"]


def on_periphery(lat, lon, area):
    """Whether a point is at the area's edge: where a one-way route through it may end."""
    return area_edge(lat, lon, area)[1] <= area["margin_m"] + area["periphery_m"]


# Sites whose road walks may also follow primary roads (state/US routes); elsewhere major roads can only be crossed.
PRIMARY_ROAD_SITES = {"harding-road"}
MAJOR_ROADS = ["primary", "primary_link", "trunk", "trunk_link"]
# Areas (lat, lon, radius m) where a route may not end: road access that isn't a trailhead.
EXCLUDED_ENDS = [(44.27060, -71.30330, 500.0)]  # Mount Washington summit facilities
EXCLUDED_END_ROADS = {"Mount Washington Auto Road"}  # nor anywhere along these roads
# Per-site closures (OSM node pairs): trail access that a site's routes may not use.
SITE_CLOSED = {"mikes-house": {frozenset((983378796, 1129156784))}}  # footway off Marion Avenue, Nelsonville

# Starts on a street: each route starts at one of the listed trailheads, reached by the shortest walk from the start
# (counted in distance; no other street walking), and follows the trails-only rules. Their routes also seed these.
ROAD_START_SITES = {"mikes-house": ["nelsonville", "washburn", "wilkinson-east"]}
BASE_ROAD_HIGHWAYS = list(mg.ROAD_HIGHWAYS)


PISTE_SITES = {"stowe", "underhill"}  # sites whose routes may use ski runs (Vermont)


def with_pistes(osm, point, radius_m):
    """``osm`` plus the ski runs (downhill and nordic) near ``point`` that aren't mapped as paths, as paths: on foot
    they're fair game."""
    q = (f'[out:json][timeout:300];way["piste:type"~"^(downhill|nordic)$"][!"highway"]'
         f'(around:{radius_m:.0f},{point[0]:.6f},{point[1]:.6f});(._;>;);out body;')
    extra = mg._overpass(q, "pistes")["elements"]
    have = {(e["type"], e["id"]) for e in osm["elements"]}
    add = []
    for e in extra:
        if (e["type"], e["id"]) in have:
            continue
        if e["type"] == "way":
            if e["nodes"][0] == e["nodes"][-1]:  # a run's outline, not a line to follow
                continue
            if "glade" in e.get("tags", {}).get("name", "").lower():  # tree skiing: no path in summer
                continue
            t = dict(e.get("tags", {}))
            t["highway"] = "path"
            t.setdefault("name", t.get("piste:name", "ski run"))
            e = dict(e, tags=t)
        add.append(e)
    return dict(osm, elements=osm["elements"] + add)


def raw_graph(slug, p2p):
    """Trails and roads (closures removed), split at junctions, before road pruning."""
    th = TRAILHEADS[slug][2]
    mg.TRAIL_HIGHWAYS[:] = BASE_TRAIL_HIGHWAYS + (["pedestrian"] if slug in SITE_AREAS else [])
    if slug in SITE_AREAS:
        return area_graph(slug, p2p)
    mg.ROAD_HIGHWAYS[:] = BASE_ROAD_HIGHWAYS + MAJOR_ROADS  # major roads are fetched for their crossings
    with contextlib.redirect_stdout(io.StringIO()):
        if p2p:
            osm = mg.fetch_osm([th], DMAX * mg.MI_TO_M, True)
            if slug in PISTE_SITES:
                osm = with_pistes(osm, th, DMAX * mg.MI_TO_M)
            heads = mg.road_trailheads(osm, [th], DMAX * mg.MI_TO_M, False, None, CLOSED | SITE_CLOSED.get(slug, set()))
            heads = {n: h for n, h in heads.items() if mg._haversine(h[0], h[1], *th) >= MIN_END
                     and all(mg._haversine(h[0], h[1], a, b) > r for a, b, r in EXCLUDED_ENDS)
                     and h[2].split(" at ")[-1] not in EXCLUDED_END_ROADS}
        else:
            osm = mg.fetch_osm([th], DMAX * mg.MI_TO_M / 2, True)
            if slug in PISTE_SITES:
                osm = with_pistes(osm, th, DMAX * mg.MI_TO_M / 2)
            heads = {}
        anchors = [th] + ([] if ROAD_RUNS else [TRAILHEADS[t][2] for t in ROAD_START_SITES.get(slug, [])])
        raw, ids = mg.build_graph(osm, True, None, anchors, extra_ids=heads, closed=CLOSED | SITE_CLOSED.get(slug, set()),
                                  snap_roads=(0,) if slug in ROAD_START_SITES or ROAD_RUNS else ())
    walk_major = ("primary", "primary_link") if slug in PRIMARY_ROAD_SITES else ()
    for e in raw:
        e["walk"] = not e["road"] or e["highway"] not in MAJOR_ROADS or e["highway"] in walk_major
    return raw, ids, heads


def area_graph(slug, p2p):
    """raw_graph for a SITE_AREAS site: every usable way in the area, all of it walkable."""
    area, th = SITE_AREAS[slug], TRAILHEADS[slug][2]
    mg.ROAD_HIGHWAYS[:] = BASE_ROAD_HIGHWAYS + MAJOR_ROADS
    with contextlib.redirect_stdout(io.StringIO()):
        osm = mg.fetch_osm([area["center"]], area["radius_mi"] * mg.MI_TO_M, True)
    nodes = {e["id"]: (e["lat"], e["lon"]) for e in osm["elements"] if e["type"] == "node"}
    keep = []
    for e in osm["elements"]:
        if e["type"] != "way":
            continue
        t = dict(e.get("tags", {}))
        if any(x in t.get("name", "") for x in area["exclude_names"]) or t.get("foot") == "no":
            continue
        if not any(n in nodes and in_area(*nodes[n], area) for n in e["nodes"]):
            continue
        if t.get("access") == "private" or t.get("motor_vehicle") == "private":  # car-free drives: open on foot
            t["foot"] = "yes"
        keep.append(dict(e, tags=t))
    sub = dict(osm, elements=[e for e in osm["elements"] if e["type"] == "node"] + keep)
    with contextlib.redirect_stdout(io.StringIO()):
        heads = {}
        if p2p:
            heads = mg.road_trailheads(sub, [area["center"]], area["radius_mi"] * mg.MI_TO_M, False, None, CLOSED)
            heads = {n: h for n, h in heads.items() if in_area(h[0], h[1], area) and on_periphery(h[0], h[1], area)
                     and mg._haversine(h[0], h[1], *th) >= MIN_END}
            in_ways = {n for w in keep for n in w["nodes"] if n in nodes}
            for other in AREA_EXTRA_ENDS.get(slug, []):  # e.g. the other station
                la, lo = TRAILHEADS[other][2]
                n = min(in_ways, key=lambda k: mg._haversine(*nodes[k], la, lo))
                heads[n] = (*nodes[n], TRAILHEADS[other][0].split(",")[0])
        raw, ids = mg.build_graph(sub, True, None, [th], extra_ids=heads, closed=CLOSED)
    raw = [e for e in raw if in_area(*e["latlon"][len(e["latlon"]) // 2], area)]
    for e in raw:
        e["walk"], e["free"] = True, True
    return raw, ids, heads


HUB_M = 400.0  # walkable roads entirely this close to a trail start belong to its trailhead (lots, campground roads)


def hub_roads(raw, start):
    """Edges (ids) of the walkable roads within HUB_M of ``start``: the trailhead's own lots and access roads."""
    at = next((e["latlon"][0] for e in raw if e["u"] == start), next((e["latlon"][-1] for e in raw if e["v"] == start), None))
    if at is None:
        return set()
    return {id(e) for e in raw if e["road"] and e["walk"]
            and all(mg._haversine(la, lo, *at) <= HUB_M for la, lo in e["latlon"])}


def street_walks(raw, start, targets):
    """Edges (ids) of the shortest walk on walkable roads and trails from ``start`` to each target node."""
    G = nx.Graph()
    for k, e in enumerate(raw):
        if e["walk"] and G.get_edge_data(e["u"], e["v"], {}).get("w", math.inf) > e["length"]:
            G.add_edge(e["u"], e["v"], w=e["length"], k=k)
    if start not in G:
        return set()
    _, path = nx.single_source_dijkstra(G, start, weight="w")
    return {id(raw[G[u][v]["k"]]) for t in targets if t in path for u, v in zip(path[t], path[t][1:])}


def tier_graph(raw, ids, heads, tier_mi):
    """The graph for one road-walk tier: road links up to ``tier_mi``, pruned, contracted, with elevation."""
    with contextlib.redirect_stdout(io.StringIO()):
        if raw and raw[0].get("free"):  # an area site: everything in it
            base = mg.contract(mg.prune([dict(e) for e in raw], DMAX * mg.MI_TO_M, ids, list(heads) or None),
                               set(ids) | set(heads))
            mg.add_elevation(base, 50.0, "3dep")
            return base
        if ROAD_RUNS:
            base = [dict(e) for e in raw if road_run_ok(e)]
            base += sidewalk_links(base, 30.0)
            base = mg.contract(mg.prune(base, DMAX * mg.MI_TO_M, ids, None), set(ids))
            mg.add_elevation(base, 50.0, "3dep")
            return base
        # Road walks follow walkable roads; any road (major ones included) can be crossed.
        keep = {id(e) for e in mg.road_connectors(raw, CROSSING_MI * mg.MI_TO_M)}
        if tier_mi > CROSSING_MI:
            keep |= {id(e) for e in mg.road_connectors([e for e in raw if e["walk"]], tier_mi * mg.MI_TO_M)}
        if len(ids) > 1:  # a start on a street, with the walks to its trailheads
            keep |= street_walks(raw, ids[0], ids[1:])
        else:
            keep |= hub_roads(raw, ids[0])
        base = [dict(e) for e in raw if id(e) in keep]  # copies: add_elevation writes to edges
        base = mg.contract(mg.prune(base, DMAX * mg.MI_TO_M, ids, list(heads) or None), set(ids) | set(heads))
        mg.add_elevation(base, 50.0, "3dep")
    return base


def graph(slug, p2p):
    """The largest tier's graph (map background)."""
    raw, ids, heads = raw_graph(slug, p2p)
    crossings = CROSSINGS_ONLY or slug in ROAD_START_SITES
    return tier_graph(raw, ids, heads, CROSSING_MI if crossings else ROAD_TIERS_MI[-1]), ids, heads


def thin(a, n):
    k = max(1, len(a) // n)
    return np.concatenate([a[::k], a[-1:]])


def ekey(e):
    """Edge identity that survives rebuilding the graph for another road tier."""
    if "lat" not in e:  # a virtual link edge
        return (e["u"], e["v"], round(e["length"]))
    k = len(e["lat"]) // 2  # parallel edges can share ends and length
    return (e["u"], e["v"], round(e["length"]), round(float(e["lat"][k]), 5), round(float(e["lon"][k]), 5))


SEED_DIRS = [d for d in os.environ.get("SWEEP_SEED_DIRS", "").split(":") if d]  # earlier runs' outputs
SEED_TOL_M = 0.5  # seed tracks and graphs share OSM vertices, so a node passed lies on the track


def seed_routes(slug, name, D):
    """Saved routes for this row: of this shape, and for a loop with spurs also its special cases (loops, lollipops)."""
    names = [name] + (["loop", "lollipop"] if name == "loop-spurs" else
                      ["loop"] if name == "lollipop" and slug in SITE_AREAS else [])  # another start's loop: a lollipop here
    return sorted((r for n in names for r in seed_routes_of(slug, n, D)), reverse=True)


def seed_routes_of(slug, name, D):
    """Saved routes of this shape within ``D`` miles, best first, as (gain, gpx, site, miles): this site's (this
    run's and earlier runs'), and for a start on a street also those of its trailheads."""
    found = {}
    others = ROAD_RUN_NEIGHBORS.get(slug, []) if ROAD_RUNS else ROAD_START_SITES.get(slug, [])
    for site in [slug] + others + (SHARED_LOOP_SITES.get(slug, []) if name == "loop" else []):
        for d in [OUT] + SEED_DIRS:
            for p in glob.glob(f"{d}/{site}/cells/{name}_*.json"):
                c = json.load(open(p))
                if "none" in c or c["dist_mi"] > D + 1e-6:
                    continue
                gpx = f"{d}/{site}/{c['gpx']}"
                if os.path.exists(gpx) and c["gain_ft"] > found.get(gpx, (-1,))[0]:
                    found[gpx] = (c["gain_ft"], gpx, site, c["dist_mi"])
                    if "edges" in c:
                        SEED_EDGES[gpx] = c["edges"]
    return sorted(found.values(), reverse=True)


SEED_EDGES = {}  # saved route (gpx path) -> its solver edge counts, as [*ekey, count]


def exact_counts(gpx, sub):
    """A saved route's edge counts on ``sub`` from its stored solver solution, or None if any edge is missing."""
    at = {ekey(x): i for i, x in enumerate(sub)}
    cnt = np.zeros(len(sub), int)
    for *k, c in SEED_EDGES.get(gpx, []):
        i = at.get(tuple(tuple(x) if isinstance(x, list) else x for x in k))  # JSON turned split nodes' ids into lists
        if i is None:
            return None
        cnt[i] = c
    return cnt if gpx in SEED_EDGES else None


def rebase_seed(cnt, sub, start, shape):
    """Counts for another start's route moved to ``start`` as an out-and-back (the shortest walk to the route, then
    along it to its climbier end) or a lollipop (the walk as the stem, to the route's closed part as the loop),
    each walk run out and back. None if the route has no such part."""
    on = [k for k in range(len(sub)) if cnt[k]]
    if not on:
        return None
    G = nx.MultiGraph()
    for k, x in enumerate(sub):
        G.add_edge(x["u"], x["v"], key=k, w=x["length"])
    if start not in G:
        return None
    R = nx.MultiGraph()
    for k in on:
        R.add_edge(sub[k]["u"], sub[k]["v"], key=k)
    leaves = [n for n in R if R.degree(n) == 1]
    if shape == "lollipop":
        if nx.is_tree(R) and len(leaves) == 2 and all(cnt[k] == 1 for k in on):  # a loop the map match left open
            try:
                q = nx.shortest_path(G, *leaves, weight="w")
            except nx.NetworkXNoPath:
                return None
            for u, v in zip(q, q[1:]):
                k = min(G[u][v], key=lambda k: G[u][v][k]["w"])
                R.add_edge(u, v, key=k)
                on.append(k)
        R = nx.k_core(nx.Graph(R), 2)  # drop stems and spurs
        if not R:
            return None
    elif not (nx.is_tree(R) and len(leaves) == 2):
        return None
    dist, path = nx.single_source_dijkstra(G, start, weight="w")
    j = min(R.nodes, key=lambda n: dist.get(n, math.inf))
    if j not in path:
        return None
    out = np.zeros(len(sub), int)
    for u, v in zip(path[j], path[j][1:]):
        out[min(G[u][v], key=lambda k: G[u][v][k]["w"])] = 2
    if shape == "lollipop":
        for k in on:
            if R.has_edge(sub[k]["u"], sub[k]["v"]):
                out[k] = max(out[k], 1)
        return out
    best = None
    for leaf in (n for n in R if R.degree(n) == 1):
        q = nx.shortest_path(R, j, leaf)
        ks = [next(iter(R[u][v])) for u, v in zip(q, q[1:])]
        var = sum(sub[k]["var"] for k in ks)
        if best is None or var > best[1]:
            best = (ks, var)
    for k in best[0]:
        out[k] = 2
    return out


def starts_at(gpx, sub, node, tol_m=10.0):
    """Whether a saved route starts at ``node``."""
    at = next(((x["lat"][0], x["lon"][0]) for x in sub if x["u"] == node and "lat" in x),
              next(((x["lat"][-1], x["lon"][-1]) for x in sub if x["v"] == node and "lat" in x), None))
    m = re.search(r'<trkpt lat="([-\d.]+)" lon="([-\d.]+)"', open(gpx).read())
    return at is not None and m is not None and mg._haversine(float(m[1]), float(m[2]), *at) <= tol_m


def seed_counts_via(gpx, sub, start, mult):
    """Solver traversal counts for another start's saved route on ``sub``, plus the shortest walk from
    ``start`` to it, out and back."""
    cnt = route_counts(gpx, sub)
    G = nx.MultiGraph()
    for k, x in enumerate(sub):
        G.add_edge(x["u"], x["v"], key=k, w=x["length"])
    on = {n for k, x in enumerate(sub) if cnt[k] for n in (x["u"], x["v"])}
    if start not in G or not on:
        return None
    dist, path = nx.single_source_dijkstra(G, start, weight="w")
    end = min(on, key=lambda n: dist.get(n, math.inf))
    if end not in path:
        return None
    for u, v in zip(path[end], path[end][1:]):
        cnt[min(G[u][v], key=lambda k: G[u][v][k]["w"])] += mult
    return np.clip(cnt, 0, 2)


def cycle_seed(edges, node, lo, hi, tries=60):
    """Indices into ``edges`` of a simple loop through ``node`` between ``lo`` and ``hi`` (m) long, built from two
    node-disjoint shortest paths to a turnaround: the climbiest of up to ``tries`` turnarounds. For graphs where
    the solver finds no loop in time. None if there's none."""
    G = nx.Graph()
    for k, x in enumerate(edges):
        if x["u"] != x["v"] and G.get_edge_data(x["u"], x["v"], {}).get("w", math.inf) > x["length"]:
            G.add_edge(x["u"], x["v"], w=x["length"], k=k)
    if node not in G:
        return None
    dist, path = nx.single_source_dijkstra(G, node, cutoff=hi / 2, weight="w")
    far = sorted((n for n in dist if dist[n] >= lo / 4), key=dist.get)
    best = None
    for X in far[:: max(1, len(far) // tries)]:
        p1 = path[X]
        H = nx.restricted_view(G, p1[1:-1], [(p1[0], p1[1])] if len(p1) == 2 else [])
        try:
            l2, p2 = nx.single_source_dijkstra(H, node, X, cutoff=hi - dist[X], weight="w")
        except nx.NetworkXNoPath:
            continue
        if not lo <= dist[X] + l2 <= hi:
            continue
        ks = [G[u][v]["k"] for q in (p1, p2) for u, v in zip(q, q[1:])]
        var = sum(edges[k]["var"] for k in ks)
        if best is None or var > best[1]:
            best = (ks, var)
    return best and best[0]


def two_loop_seed(sub, start, budget, shape, tries=(0.5, 0.6, 0.4)):
    """Counts for a figure-8 (two loops through ``start`` sharing only it) or a dumbbell (a loop through ``start``
    and a bar, run twice, to a second loop), built from constructed loops: the climbiest that fits ``budget``."""
    lo = max(MIN_LOOP, MIN_LOOP_FRAC * budget)
    best = None
    for frac in tries:
        k1 = cycle_seed(sub, start, lo, budget * frac)
        if k1 is None:
            continue
        on1 = {n for k in k1 for n in (sub[k]["u"], sub[k]["v"])} - {start}
        rest = [k for k, x in enumerate(sub) if x["u"] not in on1 and x["v"] not in on1 and k not in k1]
        left = budget - sum(sub[k]["length"] for k in k1)
        cnt = np.zeros(len(sub), int)
        cnt[k1] = 1
        if shape == "figure-8":  # the loops cross at a node of the first, the start or another
            k2 = None
            for X in [start] + sorted(on1, key=lambda n: -sum(n in (x["u"], x["v"]) for x in sub))[:12]:
                restX = [k for k, x in enumerate(sub)
                         if k not in k1 and not ({x["u"], x["v"]} & (on1 | {start}) - {X})]
                k2 = cycle_seed([sub[k] for k in restX], X, lo, left)
                if k2 is not None:
                    k2 = [restX[k] for k in k2]
                    break
            if k2 is None:
                continue
            cnt[k2] = 1
        else:
            G = nx.Graph()
            for k in rest:
                x = sub[k]
                if G.get_edge_data(x["u"], x["v"], {}).get("w", math.inf) > x["length"]:
                    G.add_edge(x["u"], x["v"], w=x["length"], k=k)
            if start not in G:
                continue
            dist, path = nx.single_source_dijkstra(G, start, cutoff=(left - lo) / 2, weight="w")
            found = None
            for J in sorted((n for n in dist if n != start and G.degree(n) >= 3), key=dist.get)[:8]:
                bar = path[J]
                off = set(bar[:-1])
                rest2 = [k for k in rest if sub[k]["u"] not in off and sub[k]["v"] not in off]
                k2 = cycle_seed([sub[k] for k in rest2], J, lo, left - 2 * dist[J])
                if k2 is not None:
                    found = ([G[u][v]["k"] for u, v in zip(bar, bar[1:])], [rest2[k] for k in k2])
                    break
            if found is None:
                continue
            cnt[found[0]] = 2
            cnt[found[1]] = 1
        var = sum(x["var"] * c for x, c in zip(sub, cnt))
        if best is None or var > best[1]:
            best = (cnt, var)
    return best and best[0]


def path_seed(sub, start, ends, budget):
    """Counts for a one-way route: the climbiest of the shortest paths from ``start`` to each end that fits."""
    G = nx.Graph()
    for k, x in enumerate(sub):
        if G.get_edge_data(x["u"], x["v"], {}).get("w", math.inf) > x["length"]:
            G.add_edge(x["u"], x["v"], w=x["length"], k=k)
    if start not in G:
        return None
    dist, path = nx.single_source_dijkstra(G, start, cutoff=budget, weight="w")
    best = None
    for end in (e for e in ends if e in dist and e != start):
        ks = [G[u][v]["k"] for u, v in zip(path[end], path[end][1:])]
        var = sum(sub[k]["var"] for k in ks)
        if best is None or var > best[1]:
            best = (ks, var)
    if best is None:
        return None
    cnt = np.zeros(len(sub), int)
    cnt[best[0]] = 1
    return cnt


def lollipop_seed(sub, start, budget, workers, stem_max=800.0, tries=8, limit=10.0, min_length=0.0):
    """A lollipop to start the solver from: for each of the nearest junctions within ``stem_max`` (m), the
    shortest path there as the stem plus the best loop through it found in ``limit`` s on the graph without the
    stem. With ``min_length`` the flattest such lollipop at least that long instead. Returns (counts, gain in ft)
    for the best, or None."""
    G = nx.MultiGraph()
    for k, x in enumerate(sub):
        G.add_edge(x["u"], x["v"], key=k, w=x["length"])
    dist, path = nx.single_source_dijkstra(G, start, cutoff=stem_max, weight="w")
    ends = sorted((n for n in dist if n != start and G.degree(n) >= 3), key=dist.get)[:tries]
    at = {id(x): i for i, x in enumerate(sub)}
    best = None
    for end in ends:
        stem_nodes = set(path[end][:-1])
        stem = [min(G[u][v], key=lambda k: G[u][v][k]["w"]) for u, v in zip(path[end], path[end][1:])]
        rest = [x for x in sub if x["u"] not in stem_nodes and x["v"] not in stem_nodes]
        min_loop = max(MIN_LOOP, MIN_LOOP_FRAC * budget)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                m, *_ = mg.solve(rest, [end], None, budget - 2 * dist[end], mg.TOPOLOGIES["loop"], min_loop, limit,
                                 workers, False, minimize=min_length > 0,
                                 min_length=max(min_length - 2 * dist[end], 0.0))
        except SystemExit:
            ks = None if min_length else cycle_seed(rest, end, min_loop, budget - 2 * dist[end])
            if ks is None:
                continue
            m = np.zeros(len(rest), int)
            m[ks] = 1
        cnt = np.zeros(len(sub), int)
        for x, c in zip(rest, m):
            cnt[at[id(x)]] = c
        for k in stem:
            cnt[k] = 2
        gain = sum(x["var"] * c for x, c in zip(sub, cnt)) / 2 * mg.M_TO_FT
        if best is None or (gain < best[1] if min_length else gain > best[1]):
            best = (cnt, gain)
    return best


def seed_counts(gpx, sub, spur, walk, mult):
    """Solver traversal counts for a saved route on ``sub``, less the edges added after solving: the access path
    ``spur`` and the street walk ``walk`` (None when the route doesn't take that walk)."""
    cnt = route_counts(gpx, sub)
    at = {id(x): i for i, x in enumerate(sub)}
    if any(id(x) in at and cnt[at[id(x)]] < mult for x in walk):
        return None
    for x in spur + walk:
        if id(x) in at:
            cnt[at[id(x)]] -= mult
    return np.clip(cnt, 0, 2)


def route_counts(gpx, edges):
    """Traversal count per edge of a saved route, map-matched onto edges (which it wasn't necessarily solved
    on): the graph nodes the track passes within SEED_TOL_M, in track order, joined by the edge between each
    consecutive pair whose length best matches the track (or the shortest path, where a node was missed)."""
    pts = np.array(re.findall(r'lat="([-\d.]+)" lon="([-\d.]+)"', open(gpx).read()), float)
    sc = np.array([110540.0, 111320.0 * math.cos(math.radians(pts[:, 0].mean()))])
    xy = pts * sc
    along = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    cell = 20.0
    grid = {}
    for i, (p, q) in enumerate(zip(xy[:-1], xy[1:])):
        lo, hi = np.floor(np.minimum(p, q) / cell).astype(int), np.floor(np.maximum(p, q) / cell).astype(int)
        for gx in range(lo[0], hi[0] + 1):
            for gy in range(lo[1], hi[1] + 1):
                grid.setdefault((gx, gy), []).append(i)
    node_xy, G = {}, nx.MultiGraph()
    for k, e in enumerate(edges):
        if "lat" not in e:  # a stand-in edge without geometry
            continue
        node_xy[e["u"]] = np.array([e["lat"][0], e["lon"][0]]) * sc
        node_xy[e["v"]] = np.array([e["lat"][-1], e["lon"][-1]]) * sc
        G.add_edge(e["u"], e["v"], key=k, w=e["length"])
    visits = []  # (distance along the track, node)
    for n, m in node_xy.items():
        hits = []
        for i in grid.get(tuple(np.floor(m / cell).astype(int)), ()):
            p, q = xy[i], xy[i + 1]
            t = np.clip(np.dot(m - p, q - p) / max(np.dot(q - p, q - p), 1e-9), 0, 1)
            d = np.linalg.norm(p + t * (q - p) - m)
            if d < SEED_TOL_M:
                hits.append((along[i] + t * (along[i + 1] - along[i]), d))
        hits.sort()
        run = []
        for h in hits + [(math.inf, 0)]:  # one visit per pass: the closest point of each run of hits
            if run and h[0] - run[-1][0] > 2 * SEED_TOL_M:
                visits.append((min(run, key=lambda r: r[1])[0], n))
                run = []
            run.append(h)
    visits.sort()
    seq = [n for _, n in visits]
    pos = [x for x, _ in visits]
    counts = np.zeros(len(edges), int)
    for (n1, x1), (n2, x2) in zip(zip(seq, pos), zip(seq[1:], pos[1:])):
        if n1 == n2:
            continue
        if G.has_edge(n1, n2):
            k = min(G[n1][n2], key=lambda k: abs(G[n1][n2][k]["w"] - (x2 - x1)))
            counts[k] += 1
            continue
        try:
            path = nx.shortest_path(G, n1, n2, weight="w")
        except nx.NetworkXNoPath:
            continue
        for u, v in zip(path, path[1:]):
            counts[min(G[u][v], key=lambda k: G[u][v][k]["w"])] += 1
    return counts


def job(spec):
    slug, name, bset = spec.split(":")
    workers = int(os.environ["SWEEP_WORKERS"])
    p2p = name == "traverse"
    topo = mg.TOPOLOGIES[name]
    # Flattest road loops: Main Street may be run on both sides, with other streets between (no turning back on
    # it); nothing else twice.
    main_st_twice = ROAD_RUNS and MINIMIZE and name == "loop"
    if main_st_twice:
        topo = mg.TOPOLOGIES["loop-spurs"]
    raw, ids, heads = raw_graph(slug, p2p)
    ends = list(heads) if p2p else None
    tiers = {}
    mult = 1 if p2p else 2

    def tier(t):
        """Per-tier graph, start options and loop bound (built on first use)."""
        if t not in tiers:
            edges = mg.subdivide(tier_graph(raw, ids, heads, t), None if topo["spurs"] == 0 else 500)
            # Each start option is a trail node the route starts from, the edges run to it out (and back) on top of
            # what the solver picks (the access path, and from a street start the walk to the trailhead), and their
            # distance. The access path stays in the solver's graph so side branches along it remain reachable; the
            # street walk doesn't. Neither counts toward the shape.
            options = {}
            if ROAD_RUNS:  # starts on the street itself; nothing added
                starts_at = []
                options[ids[0]] = dict(core=ids[0], site=slug, walk=[], spur=[], cost=0.0)
            elif slug in ROAD_START_SITES:
                G = nx.Graph()
                for k, x in enumerate(edges):
                    if G.get_edge_data(x["u"], x["v"], {}).get("w", math.inf) > x["length"]:
                        G.add_edge(x["u"], x["v"], w=x["length"], k=k)
                _, path = nx.single_source_dijkstra(G, ids[0], weight="w")
                starts_at = [(site, node, [edges[G[u][v]["k"]] for u, v in zip(path[node], path[node][1:])])
                             for site, node in zip(ROAD_START_SITES[slug], ids[1:]) if node in path]
                for _, _, walk in starts_at:
                    for x in walk:
                        if x["roadpt"].any():
                            x["frozen"] = True
            else:
                starts_at = [(slug, ids[0], [])]
            for site, node, walk in starts_at:
                spur, core = mg.access_spur(edges, node, SPUR_MAX)
                spur = [edges[k] for k in spur]
                cost = mult * sum(x["length"] for x in walk + spur)
                if core not in options or cost < options[core]["cost"]:
                    options[core] = dict(core=core, site=site, walk=walk, spur=spur, cost=cost)
            options = list(options.values())
            if slug in ROAD_START_SITES and not p2p and not ROAD_RUNS:
                # Out via one trailhead and home via another: one edge standing for both walks, run once.
                z_at = {}
                for x in edges:
                    z_at[x["u"]], z_at[x["v"]] = x["z"][0], x["z"][-1]
                same = list(options)
                for i, oa in enumerate(same):
                    for ob in same[i + 1:]:
                        walks = oa["walk"] + oa["spur"] + ob["walk"] + ob["spur"]
                        link = dict(u=oa["core"], v=ob["core"], length=sum(x["length"] for x in walks),
                                    var=sum(x["var"] for x in walks), z=np.array([z_at[oa["core"]], z_at[ob["core"]]]),
                                    roadpt=np.zeros(1, bool), names=np.array(["street"], dtype=object))
                        edges.append(link)
                        options.append(dict(core=oa["core"], site=f"{oa['site']}+{ob['site']}", walk=oa["walk"] + ob["walk"],
                                            spur=oa["spur"] + ob["spur"], cost=0.0, link=link, pair=(oa, ob)))
            for i, o in enumerate(options):
                if o.get("link") is not None:
                    o["link"]["link"] = i
            # Shapes that start on a loop can't be shorter than the shortest loop through the start.
            min_len = (min(mg.shortest_loop_through(edges, o["core"]) + o["cost"] for o in options)
                       if topo["start"] == "loop" else 0.0)
            tiers[t] = (edges, options, min_len)
        return tiers[t]
    os.makedirs(f"{OUT}/{slug}/cells", exist_ok=True)
    os.makedirs(f"{OUT}/{slug}/gpx", exist_ok=True)
    prev, prev_gain, prev_tier = {}, -1.0, None
    budgets = ([b for b in BUDGETS["all"] if b[1] in bset[len("labels="):].split("+")]
               if bset.startswith("labels=") else BUDGETS[bset])
    for D, label in budgets:
        if os.environ.get("SKIP_EXISTING") and os.path.exists(f"{OUT}/{slug}/cells/{name}_{label}.json"):
            prev, prev_gain = {}, -1.0  # the next budget seeds or warm-starts instead of chaining from this one
            continue
        # A start on a street gets the walk to its trailheads and otherwise the trails-only rules.
        t = "roads" if ROAD_RUNS else "area" if slug in SITE_AREAS else CROSSING_MI if slug in ROAD_START_SITES else road_tier(D)
        if t != prev_tier:  # a new graph: the previous route's edges don't carry over, so seed from saved routes
            prev, prev_gain, prev_tier = {}, -1.0, t
        solve_edges, options, min_len = tier(t)
        budget = D * mg.MI_TO_M
        starts, costs = [o["core"] for o in options], [o["cost"] for o in options]
        cell = dict(topology=name, budget_mi=D, label=label)
        if budget < min_len:
            cell["none"] = (f"INFEASIBLE: the shortest loop through the trailhead is {min_len / mg.MI_TO_M:.1f} mi"
                            if min_len < math.inf else "INFEASIBLE: no loop passes through the trailhead")
            json.dump(cell, open(f"{OUT}/{slug}/cells/{name}_{label}.json", "w"))
            print(f"{slug:15s} {name:14s} {label:>8s}: none (no loop through the start fits)", flush=True)
            continue
        if budget <= min(costs):
            cell["none"] = "INFEASIBLE: budget used up by the access path"
            json.dump(cell, open(f"{OUT}/{slug}/cells/{name}_{label}.json", "w"))
            continue
        cell = dict(topology=name, budget_mi=D, label=label)
        t0 = time.time()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                sub = mg.prune(solve_edges, budget, starts, ends)
                if main_st_twice:
                    sub = [dict(x, once=not any("Main Street" in str(n) for n in x["names"])) for x in sub]
                # A two-trailhead option whose walk edge was pruned (too long for the budget) is dropped with it.
                kept = {id(x) for x in sub}
                opts = [o for o in options if o.get("link") is None or id(o["link"]) in kept]
                for i, o in enumerate(opts):
                    if o.get("link") is not None:
                        o["link"]["link"] = i
                starts, costs = [o["core"] for o in opts], [o["cost"] for o in opts]
                # Seed from the best saved route that fits, when it beats the previous budget's route. Road-run
                # lollipops also start from a stem plus loop built directly, which the search struggles to find itself.
                if name == "lollipop" and (ROAD_RUNS or slug in SITE_AREAS):
                    found = lollipop_seed(sub, opts[0]["core"], budget - opts[0]["cost"], workers,
                                          min_length=MIN_FILL * D * mg.MI_TO_M if MINIMIZE else 0.0)
                    if found and (MINIMIZE or found[1] > prev_gain):
                        prev, prev_gain = {ekey(sub[i]): int(found[0][i]) for i in range(len(sub))}, found[1]
                        print(f"seed lollipop from stem + loop ({found[1]:.0f} ft) for {label}", file=sys.stderr)
                # The flattest edition starts from the flattest saved route that still fills the row.
                seeds = seed_routes(slug, name, D)
                seeds = ([] if prev else sorted(seeds)[:6]) if MINIMIZE else seeds[:6]
                taken = False
                for seed_gain, gpx, site, _ in seeds:
                    if not MINIMIZE and seed_gain <= prev_gain:
                        break
                    for o in opts:
                        exact = False
                        if ROAD_RUNS and site != slug:
                            cnt = seed_counts_via(gpx, sub, o["core"], mult)
                        elif slug in SITE_AREAS and f"_{name}_" not in os.path.basename(gpx) and (starts_at(gpx, sub, o["core"]) or starts_at(gpx, sub, ids[0])):
                            cnt = None  # this start's route of another shape
                        elif slug in SITE_AREAS and not (starts_at(gpx, sub, o["core"]) or starts_at(gpx, sub, ids[0])):
                            # Another start's route: a loop through this start as it is; any other shape (but a
                            # traverse) with the walk from this start added, out and back.
                            if name == "loop":
                                cnt = seed_counts(gpx, sub, [], [], mult)
                                if cnt is not None and not any(c and o["core"] in (x["u"], x["v"]) for x, c in zip(sub, cnt)):
                                    cnt = None
                            elif name in ("out-and-back", "lollipop"):
                                cnt = rebase_seed(route_counts(gpx, sub), sub, o["core"], name)
                            else:
                                cnt = None
                        elif (site != slug and o["site"] != site) or o.get("link") is not None:
                            continue
                        else:
                            cnt = exact_counts(gpx, sub) if site == slug and not o.get("walk") else None
                            exact = cnt is not None
                            if not exact:
                                cnt = seed_counts(gpx, sub, o["spur"], o["walk"] if site == slug else [], mult)
                        length = None if cnt is None else sum(x["length"] * c for x, c in zip(sub, cnt)) + o["cost"]
                        # A route re-mapped from its track can pick up a metre or two; the solver repairs that.
                        slack = 1.0 if exact else 1.005
                        if length is not None and length <= budget * slack and (not MINIMIZE or length >= MIN_FILL * budget):
                            prev = {ekey(sub[i]): int(cnt[i]) for i in range(len(sub))}
                            print(f"seed {os.path.basename(gpx)} ({seed_gain:.0f} ft) for {label}", file=sys.stderr)
                            taken = True
                            break
                    if taken:
                        break
                if not prev and slug in SITE_AREAS and name in ("figure-8", "dumbbell", "traverse"):
                    cnt = (path_seed(sub, opts[0]["core"], ends, budget) if p2p
                           else two_loop_seed(sub, opts[0]["core"], budget, name))
                    if cnt is not None:
                        prev = {ekey(sub[i]): int(cnt[i]) for i in range(len(sub))}
                        print(f"seed {name} constructed for {label}", file=sys.stderr)
                if not prev and name == "loop" and slug in SITE_AREAS:
                    ks = cycle_seed(sub, opts[0]["core"], max(MIN_LOOP, MIN_LOOP_FRAC * budget), budget - opts[0]["cost"])
                    if ks is not None:
                        prev = {ekey(sub[k]): 1 for k in ks}
                        print(f"seed loop constructed for {label}", file=sys.stderr)
                if not prev and not ROAD_RUNS:
                    # Warm start: a trail-only route is always valid with roads allowed, and much quicker to find.
                    trail = [e for e in sub if not e["roadpt"].any()]
                    on_trail = {n for e in trail for n in (e["u"], e["v"])}
                    try:
                        if not set(starts) <= on_trail:  # a start on its trailhead's roads
                            raise SystemExit
                        mt, *_ = mg.solve(trail, starts, ends, budget, topo, MIN_LOOP, WARM_LIMIT, workers, False,
                                          min_loop_frac=MIN_LOOP_FRAC, start_cost=costs)
                        prev = {ekey(trail[i]): mt[i] for i in range(len(trail))}
                    except SystemExit:
                        pass
                hint = np.array([prev.get(ekey(e), 0) for e in sub])
                m, s, e, proven = mg.solve(sub, starts, ends, budget, topo, MIN_LOOP, TIME_LIMIT, workers, False,
                                           hint=hint if prev else None, min_loop_frac=MIN_LOOP_FRAC,
                                           road_time_frac=(None if slug in ROAD_START_SITES or slug in SITE_AREAS or ROAD_RUNS
                                                           else ROAD_TIME_FRAC),
                                           start_cost=costs, minimize=MINIMIZE,
                                           min_length=MIN_FILL * D * mg.MI_TO_M if MINIMIZE else 0.0,
                                           no_turnarounds=main_st_twice)
                pos = {id(x): i for i, x in enumerate(sub)}
                o = next((o for o in opts if o.get("link") is not None and m[pos[id(o["link"])]]), None) or next(
                    o for o in opts if o["core"] == s and o.get("link") is None)
                full_edges, full_m = list(sub), list(m)
                if o.get("link") is not None:
                    full_m[pos[id(o["link"])]] = 0  # replaced by the walks it stands for, each run once
                for x in o["spur"] + o["walk"]:
                    if id(x) not in pos:
                        pos[id(x)] = len(full_edges)
                        full_edges.append(x)
                        full_m.append(0)
                    full_m[pos[id(x)]] += 1 if o.get("link") is not None else mult
                route = mg.assemble(full_edges, np.array(full_m), ids[0], e if p2p else ids[0])
        except SystemExit as err:
            cell["none"] = str(err)
            json.dump(cell, open(f"{OUT}/{slug}/cells/{name}_{label}.json", "w"))
            print(f"{slug:15s} {name:14s} {label:>8s}: none ({'not found in time' if 'UNKNOWN' in str(err) else 'infeasible'})", flush=True)
            continue
        prev = {ekey(sub[i]): m[i] for i in range(len(sub))}
        shape, det = mg.classify(sub, m, s, e, MIN_LOOP, MIN_LOOP_FRAC)
        access = sum(x["length"] for x in o["spur"] + o["walk"]) / mg.MI_TO_M
        if o.get("link") is not None:
            a_, b_ = (TRAILHEADS[x["site"]][0] for x in o["pair"])
            det += f"; out via {a_}, home via {b_} ({access:.2f} mi of streets and access paths)"
            via = ""
        else:
            via = f" from the house via {TRAILHEADS[o['site']][0]}" if o["walk"] else " access path"
            if access and not p2p:
                det += f"; plus {access:.2f} mi{via} each way"
        dist = route["dist"][-1] / mg.MI_TO_M
        gain = float(np.sum(np.maximum(0, np.diff(route["z"]))) * mg.M_TO_FT)
        base_name = f"{slug}_{name}_{label}"
        title = f"{TRAILHEADS[slug][0]} {shape} {dist:.1f} mi"
        if p2p:
            lat, lon, end_label = heads[e]
            cell.update(end=end_label, end_latlon=[round(lat, 5), round(lon, 5)])
            det = f"Ends at {end_label}"
            title = f"{TRAILHEADS[slug][0]} to {end_label} {dist:.1f} mi"
        if p2p and access:
            det += f" (after {access:.2f} mi{via})"
        mg.write_gpx(f"{OUT}/{slug}/gpx/{base_name}.gpx", route, title)
        idx = thin(np.arange(len(route["lat"])), 400)
        pidx = thin(np.arange(len(route["lat"])), 160)
        prev_gain = gain
        cell.update(
            shape=shape, details=det, dist_mi=round(dist, 2), gain_ft=round(gain),
            gain_raw_ft=round(float(np.sum(np.maximum(0, np.diff(route["z_raw"]))) * mg.M_TO_FT)),
            proven=bool(proven), solve_s=round(time.time() - t0, 1), gpx=f"gpx/{base_name}.gpx", file=f"{base_name}.gpx",
            line=[[round(float(route["lat"][i]), 5), round(float(route["lon"][i]), 5)] for i in idx],
            profile=[[round(float(route["dist"][i] / mg.MI_TO_M), 3), round(float(route["z_raw"][i] * mg.M_TO_FT))] for i in pidx],
            high_ft=round(float(route["z_raw"].max() * mg.M_TO_FT)),
            legs=[[n, round(l / mg.MI_TO_M, 2)] for n, l in route["legs"]],
            edges=[[*ekey(sub[i]), int(m[i])] for i in range(len(sub)) if m[i]],
        )
        json.dump(cell, open(f"{OUT}/{slug}/cells/{name}_{label}.json", "w"))
        flag = "" if shape == name else f"  SHAPE MISMATCH: {shape}"
        print(f"{slug:15s} {name:14s} {label:>8s}: {gain:6,.0f} ft ({dist:.2f} mi, {'optimal' if proven else 'best found'}){flag}", flush=True)
    return spec


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        for slug, (title, place, th) in TRAILHEADS.items():
            if len(sys.argv) > 2 and slug not in sys.argv[2].split(","):
                continue
            for p2p in (False, True):
                t0 = time.time()
                base, ids, heads = graph(slug, p2p)
                net = [[[round(la, 5), round(lo, 5)] for la, lo in zip(e["lat"][::4].tolist() + [e["lat"][-1]], e["lon"][::4].tolist() + [e["lon"][-1]])] for e in base]
                os.makedirs(f"{OUT}/{slug}", exist_ok=True)
                json.dump(dict(trailhead=th, title=title, place=place, edges=net), open(f"{OUT}/{slug}/network{'_p2p' if p2p else ''}.json", "w"))
                print(f"{slug:15s} {'p2p' if p2p else 'closed'}: {len(base)} edges, {len(heads)} end trailheads ({time.time() - t0:.0f} s)", flush=True)
    else:
        jobs, workers, procs = sys.argv[2].split(","), sys.argv[3], int(sys.argv[4])
        os.environ["SWEEP_WORKERS"] = workers
        with Pool(procs) as pool:
            for spec in pool.imap_unordered(job, jobs):
                print(f"### done {spec}", flush=True)
