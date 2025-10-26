

import asyncio
from collections import Counter
from itertools import islice
from typing import Dict, Iterable, List, Sequence
from vllm import LLM, SamplingParams
import ast
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
from google import genai
from openai import OpenAI, AsyncOpenAI
import os
import json

sys.path.insert(0, "../../..")
from config import REPO_ROOT

client_g = genai.Client()
client_o = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

################################################################################################################################################################################################
# UNIVERSAL LLM CLIENT FACTORY
################################################################################################################################################################################################
def _ensure_batch(msgs):
    """Return (batched_msgs, was_batched_bool)."""
    return (msgs, True) if msgs and isinstance(msgs[0], list) else ([msgs], False)

def chunks(iterable, size):
    """Yield successive `size`-length chunks."""
    it = iter(iterable)
    while (batch := list(islice(it, size))):
        yield batch

# 1. vLLM ───────────────────────────────────────────────────────────────────
class VllmClient:
    def __init__(self, model_name: str, dtype: str, tensor_parallel_size: int, max_concurrent, gpu_memory_utilization, wait, pipeline_parallel_size, vllm_params: dict = None, **_):
        self.model_name = model_name
        self.vllm_params = vllm_params or {}
        self.llm = LLM(model=model_name,
                       dtype=dtype,
                       tensor_parallel_size=tensor_parallel_size,
                       pipeline_parallel_size=pipeline_parallel_size,
                       trust_remote_code=True,
                       gpu_memory_utilization=gpu_memory_utilization,
                       )

    def chat(self, messages: List[List[Dict[str, str]]], temperature: float,
             max_tokens: int, extended_thinking: bool, seed: int = None, **_):
        # Create base parameters
        if seed:
            param_dict = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": seed
            }
        else:
            param_dict = {
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

        # Add custom vllm_params, allowing them to override defaults
        param_dict.update(self.vllm_params)
        
        params = SamplingParams(**param_dict)

        chat_template_kwargs = {}

        # add a special chat template for Qwen --> note that the extended_thinking param is a string
        qwen_reasoning = ["Qwen/Qwen3-8B"]
        if self.model_name in qwen_reasoning:
            if ast.literal_eval(extended_thinking) == True:
                chat_template_kwargs = {"enable_thinking": True}
            else:
                chat_template_kwargs = {"enable_thinking": False}

        outputs = self.llm.chat(messages, sampling_params=params, use_tqdm=True, chat_template_kwargs=chat_template_kwargs)            # EDIT here chat_template_kwargs={"enable_thinking": True}
        return [o.outputs[0].text for o in outputs] # this contains thinking traces

# 2. OpenAI ───────────────────────────────────────────────────────────────────
class OpenAIClient:
    """Simple asynchronous OpenAI wrapper."""

    REASONING_MODELS = [
        "o3-2025-04-16",
        "gpt-5-nano",
        "gpt-5-mini",
        "gpt-5"
    ]

    def __init__(self, model_name: str, max_concurrent: int = 8, wait: float = 0.0, **_):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model_name
        self.wait = wait
        self.batch_size = max_concurrent

    async def _single_call_async(self, conv, temperature, max_tokens, seed):
        for attempt in range(8):
            try:
                if self.model in self.REASONING_MODELS:  # reasoning models use different parameters
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=conv,
                        max_completion_tokens=max_tokens,
                        seed=seed,
                    )
                else:
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=conv,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        seed=seed,
                    )
                return resp.choices[0].message.content
            except Exception as e:
                print(f"OpenAI error: {e}")
                logging.exception("OpenAI chat attempt failed")
                if attempt == 7:
                    logging.warning("OpenAI chat failed: %s", e)
                    return None
                await asyncio.sleep(2 ** attempt)

    async def _chat_async(self, messages, temperature=0.0, max_tokens=256, seed=0):
        batched, already = _ensure_batch(messages)
        outs = []
        pbar = tqdm(total=len(batched), desc=f"OpenAI {self.model}", unit="chat", leave=False)
        for batch in chunks(batched, self.batch_size):
            results = await asyncio.gather(
                *(self._single_call_async(conv, temperature, max_tokens, seed) for conv in batch),
                return_exceptions=True,
            )
            outs.extend(results)
            pbar.update(len(batch))
            if self.wait > 0:
                await asyncio.sleep(self.wait)
        pbar.close()
        return outs if already else outs[0]

    def chat(self, messages: List[List[Dict[str, str]]], temperature:float, max_tokens: int, seed: int, **_):
        return asyncio.run(self._chat_async(messages, temperature, max_tokens, seed))

# 3. Anthropic ─────────────────────────────────────────────────────────────---
class AnthropicClient:
    """Simple asynchronous Anthropic wrapper."""

    REASONING_MODELS = [
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-3-7-sonnet-20250219",
    ]

    def __init__(self, model_name: str, max_concurrent: int = 8, wait: float = 0.0, extended_thinking: str = "disabled", **_):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model_name
        self.wait = wait
        self.batch_size = max_concurrent
        self.extended_thinking = extended_thinking

    async def _single_call_async(self, conv, temperature, max_tokens, seed, extended_thinking):
        for attempt in range(8):
            try:
                if self.model in self.REASONING_MODELS:  # reasoning models may support extended thinking
                    thinking_dict = {}
                    if extended_thinking=="enabled":
                        thinking_dict.update({"type":"enabled","budget_tokens": 10000})
                    else:
                        thinking_dict.update({"type":"disabled"})
                    resp = await self.client.messages.create(
                        model=self.model,
                        messages=conv,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        thinking=thinking_dict,
                    )
                    content = "".join(getattr(block, "text", "") for block in resp.content)
                    thinking = "".join(getattr(block, "thinking", "") for block in resp.content) # check this against API documentation
                else: # standard models e.g. Haiku
                    resp = await self.client.messages.create(
                        model=self.model,
                        messages=conv,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    content = "".join(getattr(block, "text", "") for block in resp.content)
                    thinking = ""
                return thinking + content # this might not be the best way to do this... would be nice to return them separately but this will do.
            except Exception as e:
                print(f"Anthropic error: {e}")
                logging.exception("Anthropic chat attempt failed")
                if attempt == 7:
                    logging.warning("Anthropic chat failed: %s", e)
                    return None
                await asyncio.sleep(2 ** attempt)

    async def _chat_async(self, messages, temperature=0.0, max_tokens=256, seed=0, extended_thinking="disabled"):
        batched, already = _ensure_batch(messages)
        outs = []
        pbar = tqdm(total=len(batched), desc=f"Anthropic {self.model}", unit="chat", leave=False)
        for batch in chunks(batched, self.batch_size):
            results = await asyncio.gather(
                *(self._single_call_async(conv, temperature, max_tokens, seed, extended_thinking) for conv in batch),
                return_exceptions=True,
            )
            outs.extend(results)
            pbar.update(len(batch))
            if self.wait > 0:
                await asyncio.sleep(self.wait)
        pbar.close()
        return outs if already else outs[0]

    def chat(self, messages: List[List[Dict[str, str]]], temperature:float, max_tokens: int, seed: int, extended_thinking: str, **_):
        return asyncio.run(self._chat_async(messages, temperature, max_tokens, seed, extended_thinking))

# 3. FACTORY ────────────────────────────────────────────────────────────────
_CLIENTS = {
    "vllm":      VllmClient,
    "openai":    OpenAIClient,
    "anthropic": AnthropicClient,
    #"google":    GeminiClient,
}

def make_client(provider: str, **kwargs):
    """Instantiate an LLM client implementing `.chat(...)`."""
    if provider not in _CLIENTS:
        raise ValueError(f"Unsupported provider '{provider}'")
    return _CLIENTS[provider](**kwargs)