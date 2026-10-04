# PoC 2 — LLM Fingerprinting

Identify **which model** (from a known set) a victim is running, from the
migration-delay side channel. While a victim runs `run_llm_inference.py` on one
MIG slice, the shared collector probes memory on another slice; different models
produce distinguishable migration delay traces. A fresh random prompt is used on every
run so the trace reflects the *model*, not a fixed prompt.

> Part of the PAGE404 side-channel artifact — see the
> [parent README](../README.md) for the threat model and the shared collector.

---

## Pipeline

```
auto_collection.py                         # orchestrator: loops apps × runs
   ├─ promptgen.generate_prompt()          # fresh random prompt per run
   └─ collection_script.sh                 # pins both processes to MIG slices
        ├─ ATTACKER_MIG: ../build/migration_delay_side_channel  → amps + counter files
        └─ VICTIM_MIG:   python3 run_llm_inference.py --prompt "<prompt>"
```

Outputs:

```
data/amps/<app>/<app>_<run>.txt       migration-delay timestamps (collector)
data/counter/<app>/<app>_<run>.txt    access counters           (collector)
data/prompts/prompts.txt              log of every prompt used, per app
```

`<app>` comes from `VICTIM_LIST` in `auto_collection.py`, which is matched to the
lines of `victim_apps.txt` **by position** — keep the two in the same order.

---

## Requirements

### System
- NVIDIA **GH200** (Hopper, `sm_90`) with **MIG enabled** and **≥ 2 slices**.
- **CUDA 12.x** toolkit + NVIDIA driver.
- Linux, `bash`, and `g++-11` (to build the collector).

### Build the collector first
```bash
cd ..            # page404_side_channel/
./build.sh       # produces build/migration_delay_side_channel
cd llm_fingerprinting
```

### Python (victim runner: `run_llm_inference.py`)
- **Python 3.8+**.
- Core packages:

  ```bash
  pip install torch transformers accelerate bitsandbytes safetensors sentencepiece
  ```

  | Package        | Why it is needed                                                        |
  |----------------|-------------------------------------------------------------------------|
  | `torch`        | Model execution (CUDA build matching your toolkit).                     |
  | `transformers` | `AutoModelForCausalLM` / `AutoModelForSeq2SeqLM`, tokenizers, configs. Use a recent version for gemma-3 / Qwen3 / Falcon-E. |
  | `accelerate`   | `device_map="auto"` placement.                                          |
  | `bitsandbytes` | `--quant 8bit` / `4bit` models in `victim_apps.txt`.                    |
  | `safetensors`  | Loading `.safetensors` weights.                                         |
  | `sentencepiece`| Tokenizers for T5/FLAN-T5, Pegasus, and Marian (opus-mt).               |

- Some remote-code models may also need extras — install if a model errors:

  ```bash
  pip install einops sacremoses tiktoken
  ```

  (`einops` for Falcon-RW / phi-1_5; `sacremoses` for opus-mt; `tiktoken` for
  some tokenizers.) `promptgen.py` and `auto_collection.py` use only the Python
  standard library.

### Model access
`victim_apps.txt` lists 20 models. Most are openly downloadable; two are
**gated** and require accepting a license and authenticating:

- `meta-llama/Llama-3.2-1B`
- `google/gemma-3-1b-it`

```bash
huggingface-cli login      # paste a token with access to the gated models
```

>  Authenticate via `huggingface-cli
> login` or the `HF_TOKEN` environment variable. 

---

## Configure

1. **MIG slices** — edit `collection_script.sh` and set two *different* slice
   UUIDs (from `nvidia-smi -L`), or export them:

   ```bash
   export ATTACKER_MIG="MIG-...slice-a..."
   export VICTIM_MIG="MIG-...slice-b..."
   ```

2. **Collection parameters** — constants at the top of `auto_collection.py`:

   | Constant     | Default  | Meaning                                    |
   |--------------|----------|--------------------------------------------|
   | `NUM_RUNS`   | `100`    | Runs per app.                              |
   | `MemorySize` | `"90GB"` | Attacker buffer size (must fit the slice). |

3. **Model list** — `victim_apps.txt` (one victim command per line) and
   `VICTIM_LIST` in `auto_collection.py` must stay aligned by line order. Each
   command must contain a `{PROMPT}` placeholder, substituted per run.

---

## Prompts

Prompts are generated at run time by `promptgen.py` using OS entropy
(`secrets`), from a bank of subjects / verbs / templates — no external data
needed. Every prompt used is appended to `data/prompts/prompts.txt` for
reproducibility. `promptgen` also offers `generate_unique_prompt()` and
`generate_batch()` for other workflows.

---

## Run

From this directory:

```bash
python3 auto_collection.py
```

It aligns apps to victim commands, creates the `data/` trees, and loops
`NUM_RUNS × apps`, invoking `collection_script.sh` for each trial.

### Running a single trial directly

```bash
: > /tmp/amps.txt ; : > /tmp/counter.txt    # collector opens these for writing
./collection_script.sh \
    "python3 run_llm_inference.py --model distilgpt2 --task causal --quant none --prompt 'Explain hash tables.' --iters 5" \
    /tmp/amps.txt /tmp/counter.txt 90GB
```

### Running the victim alone

```bash
python3 run_llm_inference.py --model distilgpt2 --task causal \
    --prompt "Explain hash tables in simple terms." --iters 5 --max-new-tokens 128
```

---

## Output format

| File                                  | Contents                                                     |
|---------------------------------------|--------------------------------------------------------------|
| `data/amps/<app>/<app>_<run>.txt`     | no need to use this data |
| `data/counter/<app>/<app>_<run>.txt`  | Provenance header, then migration dela in terms of counter vaues.                     |
| `data/prompts/prompts.txt`            | `app | prompt` for every run.                                |

---

## Troubleshooting

- **`../build/migration_delay_side_channel not found`** — build it first
  (`cd .. && ./build.sh`).
- **`ATTACKER_MIG and VICTIM_MIG are the same slice`** — set two different MIG
  UUIDs.
- **Gated-model / 401 errors** — `huggingface-cli login` and accept the license
  for Llama-3.2-1B and gemma-3-1b-it, or remove those lines (and their
  `VICTIM_LIST` entries).
- **`bitsandbytes` import/CUDA errors** — install a build matching your CUDA, or
  change the affected lines to `--quant none`.
- **Remote-code model import errors** — `pip install einops sacremoses tiktoken`.
- **CUDA out of memory (victim)** — use quantization or drop the larger models;
  reduce `MemorySize` if the attacker buffer does not fit the slice.
