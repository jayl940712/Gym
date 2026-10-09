# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task fields for an independent screenshot-pair reconstruction."""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    """Identify one of the pinned dataset's numbered interactions."""

    model_config = ConfigDict(extra="allow")
    page_id: int = Field(ge=1, le=127)
    interaction_id: int = Field(ge=1)
