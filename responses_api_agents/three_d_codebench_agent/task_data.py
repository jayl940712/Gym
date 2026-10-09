# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    """Identify one prepared category and its conditioning mode."""

    model_config = ConfigDict(extra="allow")
    task: Literal["text_to_3d", "image_to_3d"]
    record_id: str = Field(pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
