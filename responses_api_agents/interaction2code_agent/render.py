# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render generated HTML inside a disposable sandbox, without reference annotations."""

import argparse
import json
import re
import time
from pathlib import Path

from selenium import webdriver
from selenium.common.exceptions import UnexpectedAlertPresentException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service


def render(*, html: Path, output: Path, width: int, height: int) -> dict:
    """Capture upstream-style full-page screenshots and click interact1, interact2, ..."""
    output.mkdir(parents=True, exist_ok=True)
    options = Options()
    options.add_argument("-headless")
    options.set_preference("browser.shell.checkDefaultBrowser", False)
    driver = webdriver.Firefox(service=Service(executable_path="/usr/local/bin/geckodriver"), options=options)
    report = {"screenshots": [], "click_errors": [], "width": width, "height": height}
    try:
        driver.set_page_load_timeout(30)
        driver.set_script_timeout(10)
        # Firefox's window size includes chrome; correct the content viewport explicitly.
        driver.set_window_size(width, max(200, height))
        outer = driver.execute_script(
            "return [window.outerWidth-window.innerWidth, window.outerHeight-window.innerHeight]"
        )
        driver.set_window_size(width + outer[0], max(200, height) + outer[1])
        driver.get(html.resolve().as_uri())
        time.sleep(1)
        driver.save_full_page_screenshot(str(output / "0_source.png"))
        report["screenshots"].append("0_source.png")
        elements = driver.find_elements(By.XPATH, "//*")
        count = sum(bool(re.match(r"^interact", element.get_attribute("id") or "")) for element in elements)
        for index in range(1, count + 1):
            try:
                element = driver.find_element(By.ID, f"interact{index}")
                driver.execute_script("arguments[0].scrollIntoView();", element)
                ActionChains(driver).click(element).perform()
                location = element.location
                time.sleep(0.1)
                name = f"{index}_{int(location['x'])}_{int(location['y'])}_click.png"
                driver.save_full_page_screenshot(str(output / name))
                report["screenshots"].append(name)
                time.sleep(0.1)
                # Preserve the upstream second click, including its reset behavior.
                ActionChains(driver).click(element).perform()
            except Exception as error:
                report["click_errors"].append({"element": f"interact{index}", "error": str(error)[:1000]})
        report["status"] = "ok"
    finally:
        driver.quit()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    args = parser.parse_args()
    try:
        result = render(html=args.html, output=args.output, width=args.width, height=args.height)
    except UnexpectedAlertPresentException as error:
        result = {"status": "task_error", "error": str(error), "failure_kind": "unexpected_alert"}
    except Exception as error:
        result = {"status": "browser_error", "error": str(error)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "render.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
