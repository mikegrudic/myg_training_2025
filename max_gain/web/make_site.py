"""A self-contained copy of the trail pages for a public web host: road walks allowed (roads/) and trails only
(trails/), switchable from either page, without private starts. Usage: python make_site.py OUT_DIR"""
import glob, os, re, shutil, subprocess, sys

OUT = os.path.abspath(sys.argv[1])
HERE = os.path.dirname(os.path.abspath(__file__))
EDITIONS = [("roads", "Road walks allowed", dict(SITES_DIR="sites_merged")),
            ("trails", "Trails only", dict(SITES_DIR="sites_trails_merged", EDITION="trails"))]
TITLE = "Combinatorial vertmaxxing"
ABSTRACT = ("Do it for the vert. These are routes generated to have the highest known vertical gain for a given "
            "distance from a chosen starting point, with various choices for the topology of the route. Some of these "
            "are proven optimal, while some may have room for improvement.")
ABSTRACT2 = "These vert figures are smoothed to 50 m, and tend to be conservative compared to what might show up on your watch — do not underestimate them. For comparison, our vert measure gives 4,600 ft (1,400 m) for the Escarpment Trail Run, 8,000 ft (2,440 m) for the Devil's Path, 8,800 ft (2,680 m) for the Great Range Traverse, and 8,100 ft (2,470 m) for the Presidential Traverse."
HEAD_CSS = """.site-head { padding: 8px 0 18px; margin-bottom: 14px; border-bottom: 1px solid var(--rule); }
.site-head h1 { font-size: clamp(34px, 6vw, 56px); margin: 0 0 8px; }
.site-head p { max-width: 70ch; margin: 0 0 8px; font-size: 17px; color: var(--muted); }
"""
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
    html = html.replace("</style>", TOGGLE_CSS + HEAD_CSS + "</style>", 1)
    head = f'<header class="site-head"><h1>{TITLE}</h1><p>{ABSTRACT}</p><p>{ABSTRACT2}</p></header>\n  '
    assert html.count('<div class="wrap">\n') == 1
    html = html.replace('<div class="wrap">\n', '<div class="wrap">\n  ' + head, 1)
    html = re.sub(r"<title>.*?</title>", f"<title>{TITLE}{' (trails only)' if name == 'trails' else ''}</title>", html, count=1)
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
    if not html.lower().startswith("<!doctype"):
        html = "<!doctype html>\n" + html
    open(os.path.join(page, "index.html"), "w").write(html)
open(os.path.join(OUT, "index.html"), "w").write(
    '<!doctype html><meta charset="utf-8"><title>Max-Gain Trail Routes</title>'
    '<meta http-equiv="refresh" content="0; url=roads/"><a href="roads/">Max-Gain Trail Routes</a>\n')
for name, _, _ in EDITIONS:
    page = os.path.join(OUT, name)
    size = sum(os.path.getsize(p) for p in glob.glob(os.path.join(page, "**"), recursive=True) if os.path.isfile(p))
    print(f"{name}: index {os.path.getsize(os.path.join(page, 'index.html')) / 1e6:.1f} MB, total {size / 1e6:.1f} MB, "
          f"{len(glob.glob(os.path.join(page, 'routes', '*.json')))} trailheads")
