# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
FROM python:3.12-slim-bookworm
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends \
    firefox-esr ca-certificates curl git ripgrep fonts-dejavu fonts-liberation \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir selenium==4.35.0 Pillow==11.3.0
RUN case "$TARGETARCH" in arm64) gecko_arch=aarch64;; amd64) gecko_arch=64;; *) exit 1;; esac \
    && curl -fsSL "https://github.com/mozilla/geckodriver/releases/download/v0.36.0/geckodriver-v0.36.0-linux-${gecko_arch}.tar.gz" \
    | tar -xz -C /usr/local/bin
RUN curl -fsSL https://opencode.ai/install -o /tmp/install-opencode \
    && VERSION=1.17.11 bash /tmp/install-opencode \
    && mv /root/.opencode/bin/opencode /usr/local/bin/opencode \
    && rm /tmp/install-opencode \
    && test "$(opencode --version)" = 1.17.11
COPY responses_api_agents/interaction2code_agent/render.py /opt/interaction2code/render.py
ENV OPENCODE_DISABLE_AUTOUPDATE=1 OPENCODE_DISABLE_SHARE=1 OPENCODE_DISABLE_MODELS_FETCH=true
WORKDIR /workspace
