"""Assemble the multi-trailhead page (index.html) and per-trailhead full tracks (routes/<slug>.json and .js)."""
import glob, json, os, re, sys
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
SITES = os.environ.get("SITES_DIR", "sites")
PAGE = os.environ.get("PAGE_DIR", ".")  # where index.html and routes/ go
LEDE = 'For whole-mile distance budgets up to 31 miles and the exact race distances (5K, 10K, half marathon, marathon, 50K), standing in for the nearest whole mile, the route that climbs the most in five closed shapes and one-way to another trailhead. Each map shows the route over the surrounding trail network, with the start as a circle and a one-way finish as a square. Select a route for its profile and turn-by-turn trail list.'  # the template's intro, replaced per edition
TRAILS = os.environ.get("EDITION") == "trails"  # trails-only edition: roads may only be crossed
ROADS = os.environ.get("EDITION") in ("roads", "roads-min")  # road runs: paved roads only, loops and lollipops
FLAT = os.environ.get("EDITION") == "roads-min"  # the road runs that climb the least
sys.path.insert(0, ".")
sys.path.insert(0, "..")
from codec import encode_ints, encode_pairs, simplify
from sweep_th import BUDGETS, RACES, TRAILHEADS

ORDER = ["mikes-house", "cold-spring-bandstand", "washburn", "nelsonville", "beacon-casino", "wilkinson-east", "fishkill-ridge", "hubbard-lodge", "storm-king", "bear-mountain", "anthony-wayne", "north-south-lake", "harding-road",
         "prediger", "windham-escarpment", "barnum-road", "woodland-valley", "moon-haw", "adk-loj", "garden", "rooster-comb", "giant-ridge", "daniel-webster", "pinkham-notch", "crawford-notch", "lincoln-woods", "valley-way", "stowe", "underhill", "central-park", "central-park-south", "prospect-park", "inwood-215", "dyckman-a", "fort-tryon-190"]
SHORT = {"mikes-house": "Mike's house", "cold-spring-bandstand": "Bandstand", "washburn": "Washburn", "nelsonville": "Nelsonville", "beacon-casino": "Mount Beacon", "wilkinson-east": "Wilkinson East", "hubbard-lodge": "Hubbard Lodge", "storm-king": "Storm King", "fishkill-ridge": "Fishkill Ridge", "adk-loj": "Adirondak Loj", "garden": "The Garden", "rooster-comb": "Rooster Comb", "giant-ridge": "Giant Ridge Trail", "daniel-webster": "Daniel Webster", "pinkham-notch": "Pinkham Notch", "crawford-notch": "Crawford Notch", "lincoln-woods": "Lincoln Woods", "valley-way": "Valley Way", "stowe": "Stowe", "underhill": "Underhill", "central-park": "Harlem Hill", "central-park-south": "7th Ave", "inwood-215": "215 St", "fort-tryon-190": "190 St", "dyckman-a": "Dyckman St", "prospect-park": "Prospect Park", "bear-mountain": "Bear Mountain",
         "anthony-wayne": "Anthony Wayne", "north-south-lake": "North-South Lake", "harding-road": "Harding Road",
         "prediger": "Prediger Road", "windham-escarpment": "Windham Escarpment", "barnum-road": "Barnum Road",
         "woodland-valley": "Woodland Valley", "moon-haw": "Moon Haw"}
EXCLUDE = set(filter(None, os.environ.get("EXCLUDE_SITES", "").split(",")))  # e.g. private starts
ORDER = [s for s in ORDER if s not in EXCLUDE and glob.glob(f"{SITES}/{s}/cells/*.json")]
race_labels = {label for _, label in RACES}
ROWS = sorted(BUDGETS["all"] + [(31.0, "31")])  # 31 mi until the 50K replaces it
budget_labels = {label for _, label in ROWS}
# Mountain sites list the peaks each route goes over (passes within PEAK_M of the summit) from their region's
# peakbagging list: the Catskill 3500 Club's 35, the Adirondack 46 and the White Mountain 4000-footers.
CATSKILL_35 = ["Slide", "Hunter", "Black Dome", "Thomas Cole", "Blackhead", "Westkill|West Kill", "Graham", "Doubletop",
               "Cornell", "Table", "Peekamoose", "Plateau", "Sugarloaf", "Wittenberg", "Southwest Hunter|Leavitt", "Lone",
               "Balsam Lake", "Panther", "Big Indian", "Friday", "Rusk", "Kaaterskill High", "Twin", "Balsam Cap", "Fir",
               "North Dome", "Eagle", "Balsam", "Bearpen", "Indian Head", "Sherrill", "Vly", "Windham High", "Halcott",
               "Rocky"]
ADK_46 = ["Marcy", "Algonquin", "Haystack", "Skylight", "Whiteface", "Dix", "Gray", "Iroquois", "Basin", "Gothics",
          "Colden", "Giant", "Nippletop", "Santanoni", "Redfield", "Wright", "Saddleback", "Panther", "Tabletop",
          "Rocky Peak Ridge", "Macomb", "Armstrong", "Hough", "Seward", "Marshall", "Allen", "Big Slide", "Esther",
          "Upper Wolfjaw", "Lower Wolfjaw", "Street", "Phelps", "Donaldson", "Seymour", "Sawteeth", "Cascade",
          "South Dix|Carson", "Porter", "Colvin", "Emmons", "Dial", "East Dix|Grace", "Blake", "Cliff", "Nye",
          "Couchsachraga"]
NH_48 = ["Washington", "Adams", "Jefferson", "Monroe", "Madison", "Lafayette", "Lincoln", "South Twin", "Carter Dome",
         "Moosilauke", "North Twin", "Eisenhower", "Carrigain", "Bond", "Middle Carter", "West Bond", "Garfield",
         "Liberty", "South Carter", "Wildcat|Wildcat A", "Hancock", "South Kinsman", "Field", "Osceola", "Flume",
         "South Hancock", "Pierce|Clinton", "North Kinsman", "Willey", "Bondcliff", "Zealand", "North Tripyramid",
         "Cabot", "East Osceola", "Middle Tripyramid", "Cannon", "Hale", "Jackson", "Tom", "Wildcat D", "Moriah",
         "Passaconaway", "Owl's Head", "Galehead", "Whiteface", "Waumbek", "Isolation", "Tecumseh"]
# Region per site: its list and a floor (m) on a tagged summit elevation, which rejects lower namesakes.
PEAK_LISTS = {"catskills": (CATSKILL_35, 1000.0), "adk": (ADK_46, 1100.0), "whites": (NH_48, 1150.0)}
PEAK_REGION = {**dict.fromkeys(["north-south-lake", "harding-road", "prediger", "windham-escarpment", "barnum-road",
                                "woodland-valley", "moon-haw"], "catskills"),
               **dict.fromkeys(["adk-loj", "garden", "rooster-comb", "giant-ridge"], "adk"),
               **dict.fromkeys(["daniel-webster", "pinkham-notch", "crawford-notch", "lincoln-woods", "valley-way"], "whites")}
PEAK_M = 60.0


def norm_peak(name):
    words = re.sub(r"[^a-z ]", "", name.lower().replace("-", " ")).split()
    return " ".join(w for w in words if w not in ("mount", "mt", "mountain", "peak"))


def site_peaks(net, region):
    """The region's listed peaks over the site's network, as [(name, lat, lon)], the highest of any namesakes."""
    import max_gain_route as mg
    names, floor = PEAK_LISTS[region]
    alias = {norm_peak(a): n.split("|")[0] for n in names for a in n.split("|")}
    ll = np.array([p for e in net for p in e])
    (s, w), (n, e) = ll.min(0) - 0.01, ll.max(0) + 0.01
    q = f'[out:json][timeout:120];node["natural"="peak"]["name"]({s:.4f},{w:.4f},{n:.4f},{e:.4f});out;'
    best = {}
    for x in mg._overpass(q, "peaks")["elements"]:
        key = alias.get(norm_peak(x["tags"]["name"]))
        try:
            ele = float(x["tags"].get("ele", "nan").split()[0])
        except ValueError:
            ele = float("nan")
        if key and not ele < floor and ele == ele and ele > best.get(key, (-1,))[0]:
            best[key] = (ele, x["tags"]["name"], x["lat"], x["lon"])
        elif key and ele != ele and key not in best:
            best[key] = (-1, x["tags"]["name"], x["lat"], x["lon"])
    return [(nm, la, lo) for _, nm, la, lo in best.values()]


def peaks_over(pts, peaks):
    """Names of the peaks within PEAK_M of the track, in the order first reached."""
    if not peaks:
        return []
    lat0 = np.radians(pts[:, 0].mean())
    xy = np.c_[pts[:, 1] * 111320 * np.cos(lat0), pts[:, 0] * 110540]
    hits = []
    for name, la, lo in peaks:
        d = np.hypot(xy[:, 0] - lo * 111320 * np.cos(lat0), xy[:, 1] - la * 110540)
        if d.min() <= PEAK_M:
            hits.append((int(np.argmax(d <= PEAK_M)), name))
    return [name for _, name in sorted(hits)]


def thin_line(ll, tol_m):
    """A map polyline (lat, lon) simplified to within ``tol_m``: the page draws these small."""
    ll = np.asarray(ll, float)
    if len(ll) < 3:
        return ll
    xy = np.c_[ll[:, 1] * 111320 * np.cos(np.radians(ll[:, 0].mean())), ll[:, 0] * 110540]
    return ll[simplify(np.c_[xy, np.zeros(len(xy))], tol_m)]


def merge_legs(legs):
    """The turn list with consecutive legs on the same way merged."""
    out = []
    for name, mi in legs:
        if out and out[-1][0] == name:
            out[-1][1] = round(out[-1][1] + mi, 2)
        else:
            out.append([name, mi])
    return out


os.makedirs(f"{PAGE}/routes", exist_ok=True)
sites = []
for slug in ORDER:
    title, place, th = TRAILHEADS[slug]
    net = json.load(open(f"{SITES}/{slug}/network_p2p.json"))["edges"]
    out_cells, routes, zmax = [], {}, 0
    peaks = site_peaks(net, PEAK_REGION[slug]) if slug in PEAK_REGION else []
    for p in sorted(glob.glob(f"{SITES}/{slug}/cells/*.json")):
        c = json.load(open(p))
        if c["label"] not in budget_labels:  # a dropped row
            continue
        key = f"{c['topology']}_{c['label']}"
        o = dict(t=c["topology"], b=c["label"], mi=c["budget_mi"], key=key)
        if "none" in c:
            o["none"] = "timeout" if "UNKNOWN" in c["none"] else True
            out_cells.append(o)
            continue
        text = open(f"{SITES}/{slug}/{c['gpx']}").read()
        pts = np.array(re.findall(r'lat="([-\d.]+)" lon="([-\d.]+)"><ele>([-\d.]+)', text), float)
        lat0 = np.radians(pts[:, 0].mean())
        idx = simplify(np.c_[pts[:, 1] * 111320 * np.cos(lat0), pts[:, 0] * 110540, 2 * pts[:, 2]], 0.5)
        # Space separates the two streams: polyline characters are all in ASCII 63-126.
        routes[key] = encode_pairs(pts[idx, :2], 1e5) + " " + encode_ints(np.round(pts[idx, 2] * 10))
        zmax = max(zmax, max(z for _, z in c["profile"]))
        o.update(
            dist=c["dist_mi"], gain=c["gain_ft"], raw=c["gain_raw_ft"], proven=c["proven"], details=c["details"],
            high=c["high_ft"], legs=merge_legs(c["legs"]), solve=c["solve_s"], file=c["file"],
            gpxname=re.search(r"<name>(.*?)</name>", text).group(1),
            line=encode_pairs(thin_line(c["line"], 6.0), 1e5),
            profile=encode_pairs([(d, z / 100) for d, z in c["profile"]], 100),
        )
        if peaks:
            o["peaks"] = peaks_over(pts, peaks)
        if "end_latlon" in c:
            o["endll"] = c["end_latlon"]
        out_cells.append(o)
    # A route that fits a shorter row fits every longer one, so a longer row never shows less gain: where the solver
    # did worse (or found nothing in time), carry the best shorter route forward.
    order = {label: i for i, (_, label) in enumerate(ROWS)}
    by_key = {o["key"]: i for i, o in enumerate(out_cells)}
    for t in ({o["t"] for o in out_cells} if not FLAT else ()):
        best = None
        for o in sorted((o for o in out_cells if o["t"] == t), key=lambda o: order[o["b"]]):
            if best and ("none" in o or o["gain"] < best["gain"]):
                c = {k: v for k, v in best.items() if k not in ("b", "mi", "key")}
                c.update(b=o["b"], mi=o["mi"], key=o["key"], proven=False,
                         details=f"{best['details']}; same route as the {best['b']} mi row")
                out_cells[by_key[o["key"]]] = c
                routes[o["key"]] = routes[best["key"]]
            elif "none" not in o:
                best = o
    payload = json.dumps(routes, separators=(",", ":"))
    open(f"{PAGE}/routes/{slug}.json", "w").write(payload)
    open(f"{PAGE}/routes/{slug}.js", "w").write(f"window.__ROUTES__=window.__ROUTES__||{{}};window.__ROUTES__[{json.dumps(slug)}]={payload};\n")
    sites.append(dict(slug=slug, title=title, short=SHORT[slug], place=place, trailhead=list(th), roads=0 if TRAILS or ROADS else 10,
                      zmax=zmax, net=" ".join(encode_pairs(thin_line(e, 10.0), 1e5) for e in net if len(e) > 1), cells=out_cells))
    print(f"{slug:16s} {sum('none' not in c for c in out_cells):4d} routes, {sum('none' in c for c in out_cells):3d} none, "
          f"tracks {len(payload) / 1e6:.1f} MB")
labels = {c["b"] for st in sites for c in st["cells"]}
data = dict(budgets=[dict(mi=mi, label=label, race=label in race_labels) for mi, label in ROWS if label in labels],
            sites=sites)
# Columns: the shapes that have routes (road runs: loops and lollipops).
data["shapes"] = (["loop", "lollipop"] if ROADS else
                  sorted({c["t"] for st in sites for c in st["cells"]}))
if ROADS:
    data["minimize"] = FLAT
    for st in sites:
        st["network_note"] = "OpenStreetMap public roads (paved or not), plus the Cold Spring Cemetery lanes, the West Point Foundry Preserve paths and the Main Street railroad tunnel"
html = open("template2.html").read().replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
if FLAT:
    html = html.replace("<title>Max-Gain Trail Routes</title>", "<title>Flattest Road Routes</title>")
    html = html.replace(LEDE, "For every distance from 5K to 10K, the road loop and lollipop that climb the least from each start, "
                        "at least 98% of the distance. A loop may run both sides of Main Street, with other streets in between (no "
                        "turning back on it); nothing else is repeated. Each map shows the route over the surrounding road network, with the start "
                        "as a circle. Select a route for its profile and turn-by-turn street list.")
elif ROADS:
    html = html.replace("<title>Max-Gain Trail Routes</title>", "<title>Max-Gain Road Routes</title>")
    html = html.replace(LEDE, "For every distance from 5K to 10K, the road loop and lollipop that climb the most from each start. "
                        "Each map shows the route over the surrounding road network, with the start as a circle. Select a route "
                        "for its profile and turn-by-turn street list.")
    html = html.replace("Roads can link trails, but road time (by the Strava grade-adjusted-pace model) is capped at 10% of the route's.",
                        "Road runs: public roads, paved or not (no trails, driveways or highway shoulders), plus the Cold Spring Cemetery lanes, the West Point Foundry Preserve paths and the Main Street railroad tunnel.")
if TRAILS:
    html = html.replace("<title>Max-Gain Trail Routes</title>", "<title>Max-Gain Trail Routes, Trails Only</title>")
    html = html.replace("Roads can link trails, but road time (by the Strava grade-adjusted-pace model) is capped at 10% of the route's.",
                        "Trails only: a route may cross a road (a link of at most 50 m between two trail ends) but never follow one.")
open(f"{PAGE}/index.html", "w").write(html)
print(f"index.html {len(html) / 1e6:.2f} MB; tracks total {sum(os.path.getsize(p) for p in glob.glob(f'{PAGE}/routes/*.json')) / 1e6:.1f} MB")
