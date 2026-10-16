# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Interaction2Code direct task, expressed for a file-writing coding agent."""

DIRECT_PROMPT = """Reconstruct the interactive webpage in /workspace/before.png and /workspace/after.png.
Produce /workspace/index.html with the same appearance and the interaction connecting
the two states. Match the layout, text, colors, typography, spacing, icons, borders,
floating controls, and overlays. Work only from the provided images; do not visit the
original website or use remote assets.

Build first, then refine through visual feedback:
1. Read both reference images and determine their actual pixel dimensions. Your
   next tool action after these reads MUST write a working first index.html.
   Do not crop images, sample pixels, probe fonts, or run exploratory analysis
   before this first page exists. Defer fine-detail investigation until you have
   completed two visual correction rounds below.
2. Embed all HTML, CSS, and JavaScript in index.html. Give the clickable trigger id
   interact1 and implement its transition to the after state. Use relative
   placeholder.jpg unchanged for every image, matching the reference sizes and
   positions. Treat all supplied files as read-only. Do not replace input assets or
   generate substitute images; keep your implementation self-contained in index.html.
   Recreate actual elements and behavior; do not substitute a screenshot or omit
   repeated content. End the document with </html>.
3. Write your own browser automation using the installed tools: Firefox/Selenium or
   Chromium/Playwright. Python Pillow is available too; no package installation is
   needed. Determine screenshot dimensions directly from the input images. Capture
   the initial page and the page after clicking interact1. Match each generated
   screenshot's pixel width and height to its corresponding input image; use each
   image's actual dimensions, not its resized display in the image tool. Save them under
   /workspace/feedback/round1/ with clear before/after filenames.
4. Open BOTH generated screenshots with the image-capable read tool. Compare each
   with its corresponding reference. Choose the most obvious visible mismatches and
   immediately edit index.html to correct them; do not pause for pixel measurements
   or more reference analysis. Render again under round2/ and open BOTH new
   screenshots with the image tool. Complete at least two visual inspection rounds
   with a real page revision between them before any fine-detail investigation.
   Repeat this visual correction loop while useful improvements remain and your
   budget permits. Reopening the revised screenshots is essential; numerical image
   measurements and successful browser commands do not replace visual inspection.

Prioritize overall layout and the changed interaction region, then smaller details.
After the first two visual rounds, use close-up crops or measurements when useful,
comparing the actual PNGs at the same scale. Avoid exhaustive pixel or font probing;
spend most of your effort making and visually checking meaningful page improvements.
Inspect unfamiliar symbols carefully rather than replacing them with guessed icons.
Verify both states after every correction, and preserve working click behavior.
Keep all preview images. Successful browser execution alone is not visual validation.

Deliver the file /workspace/index.html. Your final response should briefly describe
what you implemented and visually checked; do not repeat the HTML source.
"""


def build_prompt() -> str:
    """Return the task instructions; the agent infers the viewport from its images."""
    return DIRECT_PROMPT
