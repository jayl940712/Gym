#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive PLAYWRIGHT_BROWSERS_PATH=/opt/playwright
apt-get update
apt-get install -y --no-install-recommends firefox-esr chromium ca-certificates curl git ripgrep fonts-dejavu fonts-liberation
python -m pip install --no-cache-dir selenium==4.35.0 Pillow==11.3.0 playwright==1.55.0
python -m playwright install --with-deps --only-shell chromium
# Debian Chromium supports the cluster ARM nodes; route Playwright through it.
for shell in /opt/playwright/chromium_headless_shell-*/chrome-linux/headless_shell; do
    rm -f "$shell"
    printf '#!/bin/sh\nexec /usr/bin/chromium "$@"\n' > "$shell"
    chmod +x "$shell"
done
case "$(uname -m)" in aarch64) gecko_arch=aarch64;; x86_64) gecko_arch=64;; *) exit 1;; esac
curl -fsSL "https://github.com/mozilla/geckodriver/releases/download/v0.36.0/geckodriver-v0.36.0-linux-${gecko_arch}.tar.gz" | tar -xz -C /usr/local/bin
curl -fsSL https://opencode.ai/install -o /tmp/install-opencode
VERSION=1.17.11 bash /tmp/install-opencode
mv /root/.opencode/bin/opencode /usr/local/bin/opencode
rm /tmp/install-opencode
test "$(opencode --version)" = 1.17.11
# Persist the browser path for OpenCode's own Python/Playwright automation.
printf '\nexport PLAYWRIGHT_BROWSERS_PATH=/opt/playwright\n' >> /etc/profile
apt-get clean
rm -rf /var/lib/apt/lists/*
timeout 90 python -u - <<'PY'
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from playwright.sync_api import sync_playwright
options=Options();options.add_argument('-headless')
driver=webdriver.Firefox(service=Service('/usr/local/bin/geckodriver'),options=options)
driver.get('data:text/html,<button onclick="this.innerText=\'clicked\'">click</button>')
driver.find_element('tag name','button').click()
assert driver.find_element('tag name','button').text=='clicked'
driver.quit()
print('Firefox click test passed', flush=True)
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page();page.set_content('<p>browser ready</p>')
    assert page.locator('p').inner_text()=='browser ready'
    browser.close()
print('Firefox, Chromium and OpenCode ready')
PY
