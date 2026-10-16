# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve chat clients through a hosted Responses endpoint using Gym's converters."""

from copy import deepcopy

from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.responses_converter import ResponsesConverter
from responses_api_models.openai_model.app import SimpleModelServer


def response_to_chat(response: NeMoGymResponse) -> NeMoGymChatCompletion:
    """Preserve text, reasoning, function calls, and usage for a chat-based harness."""
    converter = ResponsesConverter(return_token_id_information=False)
    converted = converter.responses_to_chat_completion_create_params(
        NeMoGymResponseCreateParamsNonStreaming(
            input=[
                item.model_copy(update={"phase": None}) if item.type == "message" else item
                for item in response.output
                if item.type != "reasoning"
            ],
            model=response.model,
        )
    )
    assistants = [message for message in converted.messages if message["role"] == "assistant"]
    if len(assistants) > 1:
        raise ValueError(f"Expected one assistant turn from hosted Responses, got {len(assistants)}")
    message = dict(assistants[0]) if assistants else {"role": "assistant", "content": ""}
    reasoning = "\n".join(part.text for item in response.output if item.type == "reasoning" for part in item.summary)
    if reasoning:
        message["reasoning_content"] = reasoning
    usage = None
    if response.usage:
        u = response.usage
        usage = {
            "prompt_tokens": u.input_tokens,
            "completion_tokens": u.output_tokens,
            "total_tokens": u.total_tokens,
            "completion_tokens_details": {
                "reasoning_tokens": u.output_tokens_details.reasoning_tokens,
            },
            "prompt_tokens_details": {"cached_tokens": u.input_tokens_details.cached_tokens},
        }
    reason = "tool_calls" if message.get("tool_calls") else "stop"
    if response.status == "incomplete":
        reason = "length"
    return NeMoGymChatCompletion.model_validate(
        {
            "id": response.id,
            "object": "chat.completion",
            "created": int(response.created_at),
            "model": response.model,
            "choices": [{"index": 0, "message": message, "finish_reason": reason}],
            "usage": usage,
        }
    )


class HostedResponsesModel(SimpleModelServer):
    """Reuse authenticated aiohttp Responses calls for both supported client dialects."""

    async def chat_completions(self, body: NeMoGymChatCompletionCreateParamsNonStreaming) -> NeMoGymChatCompletion:
        converter = ResponsesConverter(return_token_id_information=False)
        if body.max_completion_tokens is None and body.max_tokens is not None:
            body = body.model_copy(update={"max_completion_tokens": body.max_tokens})
        messages = deepcopy(body.messages)
        for message in messages:
            if message["role"] not in ("user", "system", "developer"):
                continue
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if part["type"] == "text":
                        part["type"] = "input_text"
                    elif part["type"] == "image_url":
                        image = part.pop("image_url")
                        part.update(type="input_image", image_url=image["url"], detail=image.get("detail", "auto"))
        body = body.model_copy(update={"messages": messages})
        params = converter.chat_completion_to_responses_create_params(body)
        # The converter creates optional fields with None; do not send explicit null
        # values to hosted providers that use defaults for omitted fields.
        payload = params.model_dump(exclude_none=True)
        for tool in payload.get("tools", []):
            if tool["type"] == "function":
                tool.setdefault("strict", False)
                tool.setdefault("parameters", {"type": "object", "properties": {}})
        params = NeMoGymResponseCreateParamsNonStreaming.model_validate(payload)
        response = await self.responses(params)
        return response_to_chat(response)


if __name__ == "__main__":
    HostedResponsesModel.run_webserver()
