"""Bannière du README (1280x420) et visuel LinkedIn (1200x627)."""
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT = Path(__file__).resolve().parent.parent
FORMATS = {
    "banner.png": dict(w=1280, h=420, pad=56, top=44, tw=700, h1=46, p=16, sr=-330, st=60, sw=780, kb=34),
    "linkedin.png": dict(w=1200, h=627, pad=60, top=70, tw=600, h1=60, p=19, sr=-360, st=130, sw=820, kb=56),
}
with sync_playwright() as p:
    br = p.chromium.launch()
    for name, f in FORMATS.items():
        pg = br.new_page(viewport={"width": f["w"], "height": f["h"]}, device_scale_factor=2)
        pg.goto((ROOT / "tools" / "banner.html").as_uri())
        style = ";".join(f"--{k}:{v}px" for k, v in f.items())
        pg.evaluate(f"document.querySelector('.b').setAttribute('style', '{style}')")
        pg.wait_for_timeout(700)
        pg.locator(".b").screenshot(path=str(ROOT / "docs" / name))
    br.close()
print("ok")
