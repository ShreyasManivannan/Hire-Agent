"""
LLM Inference — Wrapper for both local quantized model and API-based inference.
Provides structured prompt construction and JSON output parsing.
"""

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def generate_response(
    prompt: str,
    system_prompt: str = "You are a helpful AI assistant.",
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    json_output: bool = False,
) -> str:
    """
    Generate a response from the LLM (local or API).
    """
    from .model_loader import get_model, is_api_mode

    if is_api_mode():
        return _api_inference(prompt, system_prompt, max_new_tokens, temperature)

    model, tokenizer = get_model()
    if model is None or tokenizer is None:
        return _fallback_inference(prompt, system_prompt, json_output)

    return _local_inference(
        model, tokenizer, prompt, system_prompt,
        max_new_tokens, temperature, top_p
    )


def _local_inference(
    model, tokenizer, prompt, system_prompt,
    max_new_tokens, temperature, top_p
):
    """Run inference on local quantized model."""
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return response.strip()


def _api_inference(
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Fallback to API-based inference (uses a simple rule-based approach if no API key)."""
    import os

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")

    if api_key and os.getenv("OPENAI_API_KEY"):
        try:
            import httpx
            response = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=60,
            )
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"API inference failed: {e}")

    # Fallback to built-in inference engine
    return _fallback_inference(prompt, system_prompt)


def _fallback_inference(
    prompt: str,
    system_prompt: str,
    json_output: bool = False,
) -> str:
    """
    Rule-based fallback when no LLM is available.
    Provides structured responses for interview scenarios.
    """
    prompt_lower = prompt.lower()

    if "generate" in prompt_lower and "question" in prompt_lower:
        if json_output:
            return json.dumps({
                "question": "Can you explain your experience with this technology and provide a specific example of how you've used it in a project?",
                "difficulty": "medium",
                "domain": "general",
                "follow_up": "What challenges did you face and how did you overcome them?"
            })
        return "Can you explain your experience with this technology and provide a specific example of how you've used it in a project?"

    elif "evaluate" in prompt_lower or "score" in prompt_lower:
        return json.dumps({
            "score": 6,
            "feedback": "The answer demonstrates some understanding but could benefit from more specific examples.",
            "strengths": ["Shows basic understanding"],
            "weaknesses": ["Lacks specific implementation details"],
            "ai_probability": 0.3,
        })

    elif "report" in prompt_lower:
        return json.dumps({
            "summary": "Candidate shows moderate technical knowledge.",
            "recommendation": "Consider",
            "confidence": 0.6,
        })

    return "I understand your question. Could you provide more context?"


def parse_json_response(response: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try direct JSON parse
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try extracting JSON object
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return {"raw_response": response}
