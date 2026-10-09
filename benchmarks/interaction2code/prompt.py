# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Interaction2Code direct task, expressed for a file-writing coding agent."""

DIRECT_PROMPT = """Reconstruct the interactive webpage shown in /workspace/before.png and
/workspace/after.png. Read both images with your image-capable read tool before coding.
Infer the interaction that changes the first state into the second and implement it.
Aim for a near-exact visual copy of both states: match layout, placement, dimensions,
spacing, fonts, colors, text, and all visible elements. Do not reproduce annotation boxes.

Save the complete page as /workspace/index.html with HTML, CSS, and JavaScript embedded.
Give the element that triggers the interaction the id interact1. Use the relative URL
placeholder.jpg for every image, sized to match the references. Do not depend on other
files or remote assets, omit repeated content, or replace the interaction with a static
after image. End the document with </html>.

Firefox, Selenium, and Python Playwright with headless Chromium are installed. Serve
the page with a local HTTP server in the background and use Playwright or Selenium to
open it and perform the interaction. To keep the server alive between tool calls,
detach it with setsid and redirect stdin, stdout, and stderr, for example:
setsid python3 -m http.server 8080 --directory /workspace > /tmp/server.log 2>&1 < /dev/null &
Determine the browser viewport from the input image dimensions. Capture screenshots of
both the initial state and the state after the interaction. Read both screenshots with
your image-capable read tool and compare them visually with the references. Refine the
implementation and repeat until both states match as closely as possible visually.

Work only from the provided images; do not visit or search for the original website.
Your deliverable is the saved /workspace/index.html file.
"""


def build_prompt() -> str:
    """Return the task instructions; the agent infers the viewport from its images."""
    return DIRECT_PROMPT
