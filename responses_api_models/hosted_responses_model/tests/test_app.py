# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from unittest.mock import AsyncMock

import pytest

from nemo_gym.openai_utils import NeMoGymChatCompletionCreateParamsNonStreaming, NeMoGymResponse
from responses_api_models.hosted_responses_model.app import HostedResponsesModel, response_to_chat


def response(output, **extra):
    return NeMoGymResponse.model_validate(
        dict(
            id="resp_test",
            object="response",
            created_at=1,
            model="hosted",
            output=output,
            status="completed",
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )
        | extra
    )


def test_reasoning_tool_calls_and_usage_preserved():
    r = response(
        [
            {"type": "reasoning", "id": "rs_test", "summary": [{"type": "summary_text", "text": "Read all views."}]},
            {
                "type": "function_call",
                "call_id": "call_read",
                "name": "read",
                "arguments": '{"filePath":"/workspace/Image_005.png"}',
            },
        ],
        usage={
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens_details": {"reasoning_tokens": 3},
        },
    )
    c = response_to_chat(r)
    assert c.choices[0].finish_reason == "tool_calls"
    assert c.choices[0].message.reasoning_content == "Read all views."
    call = c.choices[0].message.tool_calls[0]
    assert call.id == "call_read" and call.function.name == "read"
    assert json.loads(call.function.arguments) == {"filePath": "/workspace/Image_005.png"}
    assert c.usage.prompt_tokens == 10 and c.usage.completion_tokens == 5
    assert c.usage.completion_tokens_details.reasoning_tokens == 3


def test_truncated_response_preserves_python_and_length():
    c = response_to_chat(
        response(
            [
                {
                    "type": "message",
                    "role": "assistant",
                    "id": "msg_test",
                    "status": "incomplete",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": "import bpy\n", "annotations": []}],
                },
            ],
            status="incomplete",
        )
    )
    assert c.choices[0].message.content == "import bpy\n"
    assert c.choices[0].finish_reason == "length"


@pytest.mark.asyncio
async def test_chat_image_and_tool_history_go_to_responses():
    final = response(
        [
            {
                "type": "message",
                "role": "assistant",
                "id": "msg_test",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Done", "annotations": []}],
            }
        ]
    )
    upstream = AsyncMock(return_value=final)
    proxy = type("Proxy", (), {"responses": upstream})()
    params = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(
        {
            "model": "dummy_model",
            "max_tokens": 32768,
            "tools": [
                {"type": "function", "function": {"name": "read", "parameters": {"type": "object", "properties": {}}}}
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}],
                },
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "call_read", "type": "function", "function": {"name": "read", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "call_read", "content": "Image read successfully"},
            ],
        }
    )
    completion = await HostedResponsesModel.chat_completions(proxy, params)
    sent = upstream.await_args.args[0]
    items = sent.model_dump(exclude_none=True)["input"]
    assert sent.max_output_tokens == 32768
    assert sent.tools[0]["strict"] is False
    assert items[0]["content"][0]["type"] == "input_image"
    assert items[0]["content"][0]["image_url"] == "data:image/png;base64,AAAA"
    assert items[1]["type"] == "function_call" and items[1]["call_id"] == "call_read"
    assert items[2]["type"] == "function_call_output" and items[2]["call_id"] == "call_read"
    assert completion.choices[0].message.content == "Done"
