# Vertmaxxing: max-gain route finder

Give it a starting point, a distance and a route shape. It finds the route with the most climbing on the
OpenStreetMap trail network, and writes a GPX file.

```
python max_gain_route.py --start 41.42698,-73.96568 --distance 15 --topology lollipop
```

Elevation comes from USGS 3DEP, so it works in the US only. The solver uses OR-Tools CP-SAT; the model is
described at the top of `max_gain_route.py`.

## Setup

Python 3:

```
pip install numpy scipy networkx requests pillow ortools matplotlib
```

Run the commands from the repository root. Trail and elevation downloads are cached in `cache/max_gain_route/`,
so a second run near the same start is much faster.

## Vertmaxx a route

1. **Get the start as `lat,lon`.** In Google Maps, right-click the trailhead and click the coordinates to copy
   them. Use the parking lot or trailhead; the route starts from the nearest trail.
2. **Pick a distance** in miles. This is a maximum; the route can come in shorter if the extra distance would add
   no climbing.
3. **Pick a shape:**

   | `--topology`   | Route |
   |----------------|-------|
   | `loop`         | One loop, nothing repeated |
   | `out-and-back` | Out and back along the same trail |
   | `lollipop`     | Out on a stem, around a loop, back down the stem (the default) |
   | `figure-8`     | Two loops through one crossing point, nothing repeated |
   | `dumbbell`     | A loop, a repeated connector, a second loop, then back |
   | `traverse`     | Point to point; needs `--end LAT,LON` or `--end-trailheads` |
   | `any`          | Whatever climbs most, as long as no trail is run more than twice |

4. **Run it:**

   ```
   python max_gain_route.py --start 44.21787,-71.41128 --distance 20 --topology lollipop \
       --closures closures/abandoned_ways_2026-10.json -o crawford.gpx --plot crawford.png
   ```

   It prints the shape, the distance, the gain and a turn-by-turn list of trails, then writes the GPX and, with
   `--plot`, a map and elevation profile. This one finds about 8,300 ft in 20 mi, on the Crawford Path, Dry River and
   Webster Cliff trails.

### Useful options

| Option | What it does |
|--------|--------------|
| `-o FILE.gpx` | Where to write the route (default `max_gain_route.gpx`) |
| `--plot FILE.png` | Also draw a map and elevation profile |
| `--roads` | Allow roads as well as trails |
| `--road-connectors M` | Stay on trails, but allow road stretches up to M meters that join two trails, such as a walk along a highway between trailheads |
| `--time-limit S` | Seconds to search (default 120). Give long routes, or `figure-8`, `dumbbell` and `any`, 600 or more |
| `--end LAT,LON` | Finish here (with `--topology traverse`) |
| `--end-trailheads` | Finish at whichever trailhead gives the most gain (with `--topology traverse`) |
| `--closures FILE` | Avoid closed trail segments listed in FILE; repeat for each file in `closures/` |
| `--max-sac N` | Skip trails rated harder than SAC grade TN (1-6) |
| `--start` again | Give several starts; the solver uses whichever is best |

`python max_gain_route.py --help` lists the rest.

### Reading the result

- **Gain** is computed from elevation smoothed over about 50 m, which removes noise from the elevation data.
  It reads about 9% lower than CalTopo. The figure in parentheses is the gain without smoothing.
- **The search usually runs until the time limit** rather than proving its route is the best possible. A longer
  `--time-limit` sometimes finds more.
- **If it prints "requested X but the route is a Y"**, the best route it found has a simpler shape, for example
  a lollipop whose loop shrank away. Try another shape, or a different distance.

## Add summit side trips to a route

To add out-and-back side trips to summits to a route you already have (from this tool, CalTopo or a watch),
use `spurify.py`. It keeps your route and adds trips that turn around only at named peaks:

```
python max_gain/spurify.py my_route.gpx --extra 3        # up to 3 more miles
python max_gain/spurify.py my_route.gpx --budget 30      # or a total distance
```

It writes `my_route_spurred.gpx` and prints a table of the side trips added, with their length, gain and
summits. If it warns that the matched length is off, which can happen with a noisy watch track, raise
`--match-m`.

## When it fails

| Message | What to do |
|---------|------------|
| `All Overpass servers failed` | The OpenStreetMap servers are busy. Try again in a few minutes |
| HTTP 504 from the USGS elevation service | Add `--dem terrarium` to use AWS terrain tiles instead |
| `No feasible route found` | No route of that shape fits the distance on trails alone. A loop often needs a short road link between two trailheads: add `--road-connectors 400`. Otherwise try more miles or another shape |
| A start snapped hundreds of meters away | The point isn't near a mapped trail. Move it onto the trail |
