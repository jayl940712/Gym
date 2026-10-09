# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Interaction2Code direct task, expressed for a file-writing coding agent."""

DIRECT_PROMPT = """You are a web developer proficient in HTML, CSS and JavaScript.
Reconstruct an interactive webpage from /workspace/before.png and /workspace/after.png.
Read both images with your image-capable read tool before coding. The first shows the
original state; the second shows the state after interacting with an element. Infer
the interaction from their differences. Match the original layout, design, exact
text, fonts, colors, padding, margins, and repeated elements. Implement the changes
observed in the second image on the appropriate interactive element.

Create /workspace/index.html with all HTML, CSS and JavaScript embedded in that one
file. Set the interactive element's id to interact1, for example
<button id="interact1">Click Me!</button>. The evaluator will click that element.
Use /workspace/placeholder.jpg for every image in the reconstructed page, with
dimensions matching the screenshots. Do not depend on other files or remote assets.
Write the complete page; do not substitute comments or ellipses for repeated content.
End the HTML document with </html>. Do not reproduce annotation boxes.

You may use your coding tools to edit and test the page iteratively. Firefox and
Selenium are installed. For a local preview, run:
python /opt/interaction2code/render.py --html /workspace/index.html \\
  --output /workspace/preview --width {width} --height {height}
This writes the initial and clicked full-page screenshots and diagnostics. You can
read those images to check your implementation and revise it within your budget.
Work from the provided images only. Do not visit or search for the original website.
The deliverable is the saved /workspace/index.html file; your final message can be
a brief completion note. Private reference annotations and metrics are unavailable
during coding.
"""


def build_prompt(*, width: int, height: int) -> str:
    """Use the reference image dimensions for deterministic browser previews."""
    return DIRECT_PROMPT.format(width=width, height=height)
