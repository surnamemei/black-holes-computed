"""Build the GitHub Pages site in docs/ from the two projects' page sources.

The pages were first published as Claude artifacts, whose host wraps each file in an HTML
skeleton. This script adds that skeleton (head, metadata, minimal reset), inlines each page's
data, and turns the artifact cross-links into relative links.

    python3 tools/make_pages.py [--base https://USER.github.io/REPO/]
"""
import argparse
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS = os.path.join(ROOT, "docs")

ARTIFACT_LINKS = {
    "https://claude.ai/artifact/6znP3XzNmc2aKoUFPPKpv7": "../photograph/",
    "https://claude.ai/artifact/Gn8FFSzYWkigibgjpBMvJS": "../hear/",
}
RESET = ("<style>body{margin:0}img{max-width:100%}[hidden]{display:none!important}"
         ":root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}</style>")


def standalone(src, description, image, base):
    for a, b in ARTIFACT_LINKS.items():
        src = src.replace(a, b)
    src = src.replace(' target="_blank" rel="noopener">How to', '>How to')  # sibling pages open in place
    head, sep, body = src.partition('<div class="wrap">')
    assert sep, "page source must contain <div class=\"wrap\">"
    title = head.split("<title>")[1].split("</title>")[0]
    body = body.replace("</footer>", '  <span><a href="../">All pages</a></span>\n  </footer>', 1)
    meta = [
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">',
        f'<meta name="description" content="{description}">',
        f'<meta property="og:title" content="{title}">',
        f'<meta property="og:description" content="{description}">',
        '<meta property="og:type" content="website">',
    ]
    if base:
        meta.append(f'<meta property="og:image" content="{base}{image}">')
        meta.append('<meta name="twitter:card" content="summary_large_image">')
    return ("<!doctype html>\n<html lang=\"en\">\n<head>\n" + "\n".join(meta) + "\n" + RESET + "\n" + head.strip()
            + "\n</head>\n<body>\n" + sep + body + "\n</body>\n</html>\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="", help="public site URL ending in /, for share-preview images")
    a = ap.parse_args()
    # How to Photograph a Black Hole
    src = open(os.path.join(ROOT, "film", "web", "src", "index.src.html")).read()
    src = src.replace("/*DATA*/", open(os.path.join(ROOT, "film", "web", "data.js")).read())
    out = standalone(src, "A fully computed short film on how the Event Horizon Telescope photographed a black hole, "
                          "with a live Kerr ray tracer and a build-your-own Earth-sized telescope.",
                     "photograph/poster.jpg", a.base)
    open(os.path.join(DOCS, "photograph", "index.html"), "w").write(out)
    # How to Hear Two Black Holes Collide
    src = open(os.path.join(ROOT, "gw150914", "web", "src", "index.src.html")).read()
    d = json.load(open(os.path.join(ROOT, "gw150914", "results", "page_data.json")))
    d["chisq"] = json.load(open(os.path.join(ROOT, "gw150914", "results", "chisq.json")))
    src = src.replace("/*DATA*/null", json.dumps(d, separators=(",", ":")))
    out = standalone(src, "GW150914 found again in LIGO's raw public strain with a from-scratch matched-filter search: "
                          "listen to the chirp, explore the template bank, and see why it isn't noise.",
                     "hear/cover.jpg", a.base)
    open(os.path.join(DOCS, "hear", "index.html"), "w").write(out)
    # landing page
    repo = ""
    if a.base:  # https://USER.github.io/REPO/  ->  https://github.com/USER/REPO
        user = a.base.split("//")[1].split(".")[0]
        name = a.base.rstrip("/").rsplit("/", 1)[1]
        repo = f"https://github.com/{user}/{name}"
    land = open(os.path.join(ROOT, "tools", "landing.html")).read().replace("{BASE}", a.base).replace("{REPO}", repo)
    open(os.path.join(DOCS, "index.html"), "w").write(land)
    for p in ("index.html", "photograph/index.html", "hear/index.html"):
        print(f"docs/{p}: {os.path.getsize(os.path.join(DOCS, p)) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
