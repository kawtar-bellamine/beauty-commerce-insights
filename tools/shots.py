"""Screenshots of every dashboard tab (needs the dashboard running on port 8621)."""
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
NAMES = ["map", "forecast", "experience", "personas", "quality"]
with sync_playwright() as p:
    br = p.chromium.launch()
    pg = br.new_page(viewport={"width": 1400, "height": 1900}, device_scale_factor=2)
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto("http://localhost:8621")
    pg.wait_for_selector(".kpis", timeout=180000)
    pg.wait_for_timeout(4000)
    tabs = pg.locator("[role=tab]")
    for i, n in enumerate(NAMES):
        tabs.nth(i).click()
        pg.wait_for_timeout(2500)
        bad = pg.locator("[data-testid=stException]").count()
        pg.screenshot(path=str(OUT / f"{n}.png"), full_page=True)
        print(n, "exceptions:", bad)
    tabs.nth(0).click(); pg.wait_for_timeout(2000)
    pg.screenshot(path=str(OUT / "hero.png"), clip={"x": 0, "y": 0, "width": 1400, "height": 760})
    print("js errors:", errs)
    br.close()
