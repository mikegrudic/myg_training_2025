"""A self-contained copy of the trail pages for a public web host: road walks allowed (roads/) and trails only
(trails/), switchable from either page, without private starts. Usage: python make_site.py OUT_DIR"""
import glob, os, shutil, subprocess, sys

OUT = os.path.abspath(sys.argv[1])
HERE = os.path.dirname(os.path.abspath(__file__))
EDITIONS = [("roads", "Road walks allowed", dict(SITES_DIR="sites_merged")),
            ("trails", "Trails only", dict(SITES_DIR="sites_trails", EDITION="trails"))]
TOGGLE_CSS = """.edition { display: inline-flex; border: 1px solid var(--rule); border-radius: 999px; overflow: hidden; margin: 0 0 12px; font-size: 14px; }
.edition a { padding: 5px 14px; color: var(--muted); text-decoration: none; }
.edition a:hover { color: var(--ink); }
.edition a.on { background: var(--ink); color: var(--paper); }
"""

shutil.rmtree(OUT, ignore_errors=True)
for name, _, env in EDITIONS:
    page = os.path.join(OUT, name)
    subprocess.run([sys.executable, "build2.py"], cwd=HERE, check=True, stdout=subprocess.DEVNULL,
                   env=dict(os.environ, PAGE_DIR=page, EXCLUDE_SITES="mikes-house", **env))
    for p in glob.glob(os.path.join(page, "routes", "*.js")):  # only for pages opened from disk
        os.remove(p)
    html = open(os.path.join(page, "index.html")).read()
    links = "".join(f'<a href="../{n}/"{" class=\"on\" aria-current=\"page\"" if n == name else ""}>{label}</a>'
                    for n, label, _ in EDITIONS)
    toggle = f'<nav class="edition" aria-label="Edition">{links}</nav>\n      '
    anchor = '<p class="eyebrow" id="place"></p>'
    assert html.count(anchor) == 1 and html.count("</style>") >= 1
    html = html.replace(anchor, toggle + anchor)
    html = html.replace("</style>", TOGGLE_CSS + "</style>", 1)
    # Keep the selected trailhead when switching editions.
    keep = ('<script>document.querySelectorAll(".edition a").forEach(a => a.addEventListener("click", () => '
            '{ a.href = a.getAttribute("href").split("#")[0] + location.hash; }));</script>\n</body>')
    html = html.replace("</body>", keep, 1) if "</body>" in html else html + keep.replace("\n</body>", "")
    html = html.replace(" On claude.ai, GPX files download inside .zip archives because the viewer only saves "
                        "certain file types.", "")
    # No medals.
    assert 'const MEDALS = { 1: "gold", 2: "silver", 3: "bronze" };' in html
    html = html.replace('const MEDALS = { 1: "gold", 2: "silver", 3: "bronze" };', "const MEDALS = {};")
    html = html.replace("Gold, silver and bronze outlines mark the three shapes with the most gain at each distance; "
                        "ties share a medal. ", "")
    open(os.path.join(page, "index.html"), "w").write(html)
open(os.path.join(OUT, "index.html"), "w").write(
    '<!doctype html><meta charset="utf-8"><title>Max-Gain Trail Routes</title>'
    '<meta http-equiv="refresh" content="0; url=roads/"><a href="roads/">Max-Gain Trail Routes</a>\n')
for name, _, _ in EDITIONS:
    page = os.path.join(OUT, name)
    size = sum(os.path.getsize(p) for p in glob.glob(os.path.join(page, "**"), recursive=True) if os.path.isfile(p))
    print(f"{name}: index {os.path.getsize(os.path.join(page, 'index.html')) / 1e6:.1f} MB, total {size / 1e6:.1f} MB, "
          f"{len(glob.glob(os.path.join(page, 'routes', '*.json')))} trailheads")
