"""Animated tour of the dashboard for the README (needs the dashboard running on port 8621)."""
from io import BytesIO
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "docs" / "tour.gif"
frames, durations = [], []


def snap(pg, ms, scroll=None):
    if scroll is not None:
        pg.mouse.wheel(0, scroll)
        pg.wait_for_timeout(600)
    img = Image.open(BytesIO(pg.screenshot())).convert("RGB").resize((1000, 640), Image.LANCZOS)
    frames.append(img)
    durations.append(ms)


with sync_playwright() as p:
    br = p.chromium.launch()
    pg = br.new_page(viewport={"width": 1400, "height": 896})
    pg.goto("http://localhost:8621")
    pg.wait_for_selector(".kpis", timeout=180000)
    pg.wait_for_timeout(3500)
    snap(pg, 2200)
    pg.mouse.move(700, 600)
    snap(pg, 1800, scroll=520)
    pg.get_by_text("Late deliveries (%)").click(); pg.wait_for_timeout(1500); snap(pg, 1800)
    pg.get_by_text("Average review", exact=True).last.click(); pg.wait_for_timeout(1500); snap(pg, 1800)
    tabs = pg.locator("[role=tab]")
    tabs.nth(1).click(); pg.wait_for_timeout(2000); snap(pg, 2000)
    tabs.nth(2).click(); pg.wait_for_timeout(2000); snap(pg, 1500)
    slider = pg.locator("[data-testid=stSlider]:visible input").first
    slider.scroll_into_view_if_needed(); pg.wait_for_timeout(500)
    slider.focus()
    for _ in range(5):
        pg.keyboard.press("ArrowRight"); pg.wait_for_timeout(900); snap(pg, 700)
    snap(pg, 1600, scroll=420)
    tabs.nth(3).click(); pg.wait_for_timeout(2000)
    pg.mouse.wheel(0, -2000); pg.wait_for_timeout(500); snap(pg, 1400, scroll=480)
    snap(pg, 1800, scroll=380)
    sl = pg.locator("[data-testid=stSlider]:visible input").last
    sl.scroll_into_view_if_needed(); pg.wait_for_timeout(500)
    sl.focus()
    for _ in range(4):
        pg.keyboard.press("ArrowLeft"); pg.wait_for_timeout(900); snap(pg, 700)
    snap(pg, 1800)
    br.close()

frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
print(OUT, OUT.stat().st_size // 1024, "KB", len(frames), "frames")
