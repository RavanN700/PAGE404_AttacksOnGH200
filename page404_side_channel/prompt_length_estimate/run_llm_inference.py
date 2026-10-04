#!/usr/bin/env python3
"""
LLM inference runner for fingerprinting.
Fixes:
  1. KV cache enabled/disabled via args.
  2. Suppress safetensors auto-conversion thread.
  3. Suppress bitsandbytes warnings.
  4. Manual greedy decode fallback for DynamicCache incompatible models.
  5. Monkey-patch for missing 'get_head_mask' (Falcon/Remote code fix).
"""

import argparse
import os
import time
import warnings
from pathlib import Path
import torch

# ── Suppress safetensors auto-conversion thread ──
os.environ["SAFETENSORS_FAST_GPU"] = "0"

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    MarianConfig,
)

# Monkey-patch: disable the safetensors auto-conversion thread
try:
    import transformers.safetensors_conversion as _sc
    _sc.auto_conversion = lambda *a, **kw: None
except Exception:
    pass

warnings.filterwarnings(
    "ignore",
    message="MatMul8bitLt: inputs will be cast from torch.float32 to float16",
    category=UserWarning,
)

def parse_args():
    p = argparse.ArgumentParser("LLM inference runner for fingerprinting")
    p.add_argument("--model", required=True, help="HF model name or path")
    p.add_argument("--task", required=True, choices=["causal", "seq2seq"])
    p.add_argument("--quant", default="none", choices=["none", "8bit", "4bit"])
    p.add_argument("--iters", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--use-cache", action="store_true", default=True)
    p.add_argument("--no-cache", dest="use_cache", action="store_false")
    p.add_argument("--prompt", type=str, default=None,
                    help="Prompt text. Mutually exclusive with --prompt-file.")
    p.add_argument("--prompt-file", type=str, default=None,
                    help="Path to a file containing the prompt. Use this for "
                         "long contexts that would exceed ARG_MAX on the CLI.")

    args = p.parse_args()
    # Exactly one of --prompt / --prompt-file must be given.
    if bool(args.prompt) == bool(args.prompt_file):
        p.error("exactly one of --prompt or --prompt-file is required")
    return args

def build_quant_config(quant: str):
    if quant == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    elif quant == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    return None

def patch_falcon_masking(model):
    """
    Injects get_head_mask if missing. Remote-code models like Falcon-RW 
    often expect this method to exist on the model or its backbone.
    """
    def get_head_mask(head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            return head_mask
        return [None] * num_hidden_layers

    # Patch the main wrapper
    if not hasattr(model, "get_head_mask"):
        model.get_head_mask = get_head_mask
    
    # Patch the underlying transformer backbone (where the error usually occurs)
    backbone = getattr(model, "transformer", None) or getattr(model, "model", None)
    if backbone and not hasattr(backbone, "get_head_mask"):
        backbone.get_head_mask = get_head_mask

def load_tokenizer(model_id: str, config):
    try:
        return AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    except Exception:
        if isinstance(config, MarianConfig):
            from transformers import MarianTokenizer
            return MarianTokenizer.from_pretrained(model_id)
        return AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)

def _is_dynamiccache_error(e: Exception) -> bool:
    msg = str(e)
    return "DynamicCache" in msg and ("not subscriptable" in msg or "subscript" in msg)

@torch.no_grad()
def greedy_generate_legacy_cache(model, tokenizer, inputs, max_new_tokens, use_cache):
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)
    eos_id = tokenizer.eos_token_id

    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache, return_dict=True)
    past = out.past_key_values if use_cache else None
    generated = input_ids

    for _ in range(max_new_tokens):
        next_token_logits = out.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        generated = torch.cat([generated, next_token], dim=1)

        if eos_id is not None and int(next_token.item()) == int(eos_id):
            break

        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)

        if use_cache:
            out = model(input_ids=next_token, attention_mask=attention_mask, past_key_values=past, use_cache=True, return_dict=True)
            past = out.past_key_values
        else:
            out = model(input_ids=generated, attention_mask=attention_mask, use_cache=False, return_dict=True)

    return generated

def main():
    args = parse_args()
    if args.prompt_file:
        args.prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    torch.manual_seed(0)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    requested_q = build_quant_config(args.quant)

    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
        "config": config,
    }
    if requested_q:
        model_kwargs["quantization_config"] = requested_q

    tokenizer = load_tokenizer(args.model, config)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[run_llm_inference] Loading: {args.model}")
    if args.task == "causal":
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model, **model_kwargs)

    # CRITICAL FIX: Patch missing masking methods for Falcon-RW and similar
    patch_falcon_masking(model)
    model.eval()

    inputs = {k: v.to(next(model.parameters()).device) for k, v in tokenizer(args.prompt, return_tensors="pt", padding=True).items()}
    
    model.generation_config.do_sample = False
    model.generation_config.use_cache = args.use_cache

    print("[run_llm_inference] Warmup run...")
    try:
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=5)
    except (TypeError, AttributeError) as e:
        # Fallback if standard generate fails
        _ = greedy_generate_legacy_cache(model, tokenizer, inputs, 5, args.use_cache)

    print(f"[run_llm_inference] Starting {args.iters} timed iteration(s)...")
    start = time.time()
    outputs = None

    for _ in range(args.iters):
        try:
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        except (TypeError, AttributeError):
            outputs = greedy_generate_legacy_cache(model, tokenizer, inputs, args.max_new_tokens, args.use_cache)

    elapsed = time.time() - start
    out_ids = outputs[0]
    print(f"\n[run_llm_inference] Result:\n{tokenizer.decode(out_ids, skip_special_tokens=True)}")
    print(f"total elapsed time for {args.iters} iteration(s): {elapsed:.4f}s")
    print(f"\nAvg time: {elapsed / args.iters:.4f}s/iter")

    input_tokens  = inputs["input_ids"].shape[1]
    output_tokens = out_ids.shape[0] - input_tokens
    print(f"Input tokens: {input_tokens}")
    print(f"Output tokens: {output_tokens}")

if __name__ == "__main__":
    main()