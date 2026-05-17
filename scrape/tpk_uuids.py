import asyncio
import json
from playwright.async_api import async_playwright

URL = "https://www.tpktraffic.com/LoadMaps?lat=26.471503&long=-80.282163&zoom=5"
OUTPUT_FILE = "ptk_traffic_cameras.json"


async def main():
    results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        await page.goto(URL, wait_until="networkidle")
        await page.wait_for_selector(
            "div.marker.mapboxgl-marker.mapboxgl-marker-anchor-center"
        )
        markers = await page.query_selector_all(
            "div.marker.mapboxgl-marker.mapboxgl-marker-anchor-center"
        )

        print(f"Found {len(markers)} markers")
        
        page.set_default_timeout(1000)

        for index in range(len(markers)):
            try:
                marker = markers[index]
                await marker.evaluate("node => node.click()")

                iframe_element = await page.wait_for_selector("iframe")
                frame = await iframe_element.content_frame()

                if frame is None:
                    raise Exception("Could not access iframe")

                await frame.wait_for_selector("#lblCamera")
                new_camera_text = (
                    await frame.locator("#lblCamera").text_content()
                ).strip()

                # skip duplicates
                if results and results[-1]["camera"] == new_camera_text:
                    print(f"[{index + 1}] duplicate marker skipped")
                    continue

                camera_text = new_camera_text
                
                print(f"[{index + 1}] {camera_text}")

                results.append({
                    "index": index + 1,
                    "camera": camera_text.strip()
                })

                close = await page.query_selector(".mapboxgl-popup-close-button")
                if close is not None:
                    await close.evaluate("node => node.click()")

            except Exception as e:
                print(f"Error on marker {index + 1}: {e}")

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nSaved results to {OUTPUT_FILE}")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())