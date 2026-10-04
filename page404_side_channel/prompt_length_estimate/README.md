# PoC 1 — Prompt-Length Estimation

Estimate the **prompt length** of a victim LLM inference from the migration-delay
side channel. While a victim runs `run_llm_inference.py` on one MIG slice, the
shared collector probes memory on another slice; the resulting migration trace
scales with how much work (and memory) the victim's prompt induces.

> Part of the PAGE404 side-channel artifact — see the
> [parent README](../README.md) for the threat model and the shared collector.

---

## Pipeline

```
auto_collection.py                         # orchestrator: loops apps × runs
   └─ collection_script.sh                 # pins both processes to MIG slices
        ├─ ATTACKER_MIG: ../build/migration_delay_side_channel  → amps + counter files
        └─ VICTIM_MIG:   python3 run_llm_inference.py           → stdout parsed → metadata
```

For every victim command and every run, three files are written:

```
data/amps/<app>/<app>_<run>.txt       migration-delay timestamps (collector)
data/counter/<app>/<app>_<run>.txt    access counters           (collector)
data/metadata/<app>/<app>_<run>.txt   victim stats: input/output tokens, timing
```

`<app>` is derived automatically from each victim line's `--prompt-file` stem
(e.g. `prompts/01.txt` → `01`); stems must be unique across the file.

---

## Requirements

### System
- NVIDIA **GH200** (Hopper, `sm_90`) with **MIG enabled** and **≥ 2 slices**.
- **CUDA 12.x** toolkit + NVIDIA driver (for the collector and the victim GPU).
- Linux, `bash`, and `g++-11` (to build the collector).

### Build the collector first
```bash
cd ..            # page404_side_channel/
./build.sh       # produces build/migration_delay_side_channel
cd prompt_length_estimate
```

### Python (victim runner: `run_llm_inference.py`)
- **Python 3.8+**.
- Packages (derived from the imports in `run_llm_inference.py`):

  ```bash
  pip install torch transformers accelerate bitsandbytes safetensors sentencepiece
  ```

  | Package        | Why it is needed                                                        |
  |----------------|-------------------------------------------------------------------------|
  | `torch`        | Model execution (CUDA build matching your toolkit).                     |
  | `transformers` | `AutoModelForCausalLM` / `AutoModelForSeq2SeqLM`, tokenizers, configs.   |
  | `accelerate`   | `device_map="auto"` placement.                                          |
  | `bitsandbytes` | `--quant 4bit` / `8bit` (`BitsAndBytesConfig`). Omit if you use `none`.  |
  | `safetensors`  | Loading `.safetensors` weights.                                         |
  | `sentencepiece`| Tokenizers for some models (e.g. Marian, Gemma).                        |

- `auto_collection.py` itself uses only the Python standard library.

### Model access
The sample victim commands use **`google/gemma-3-1b-it`**, a **gated** model.
Request access on its Hugging Face page and authenticate once:

```bash
huggingface-cli login      # paste a token with access to the model
```

Or swap in any open causal model by editing `prompts/victim_apps_test.txt`
(e.g. `--model gpt2 --task causal --quant none`).

---

## Configure

1. **MIG slices** — edit `collection_script.sh` and set two *different* slice
   UUIDs (from `nvidia-smi -L`), or export them:

   ```bash
   export ATTACKER_MIG="MIG-...slice-a..."
   export VICTIM_MIG="MIG-...slice-b..."
   ```

2. **Collection parameters** — constants at the top of `auto_collection.py`:

   | Constant     | Default | Meaning                                   |
   |--------------|---------|-------------------------------------------|
   | `NUM_RUNS`   | `100`   | Repetitions per app.                      |
   | `MemorySize` | `"90GB"`| Attacker buffer size (must fit the slice).|
   | `N_ACCESSES` | `"256"` | Accesses probed per page.                 |

3. **Victim commands** — `prompts/victim_apps_test.txt`, one command per line
   (`#` comments allowed). Each must contain a `--prompt-file`; the file's stem
   becomes the app name and must be unique.

---

## Prompts

`prompts/01.txt` … `prompts/20.txt` in this repo are **short placeholders**.
The paper's runs used long-context prompts (≈15k–92k tokens). Replace each file
with your own prompt text to reproduce meaningful length estimates.

- `prompts/selection.json` documents how the paper's prompts were selected
  (source dataset ids, per-prompt token lengths, tokenizer `meta-llama/Llama-3.2-1B`,
  seed 42, 15k–100k-token filter). It is reference metadata, not read by the code.
- `prompts/VICTIM_LIST.py` is a reference list of descriptive app names.

---

## Run

From this directory:

```bash
python3 auto_collection.py
```

It derives the app list, pre-creates the `data/` trees, and loops
`NUM_RUNS × apps`, invoking `collection_script.sh` for each trial.

### Running a single trial directly

```bash
# output files must already exist (the collector opens them for writing)
: > /tmp/amps.txt ; : > /tmp/counter.txt
./collection_script.sh \
    "python3 run_llm_inference.py --model gpt2 --task causal --prompt-file prompts/01.txt --iters 1" \
    /tmp/amps.txt /tmp/counter.txt 90GB 256
```

### Running the victim alone

```bash
python3 run_llm_inference.py --model gpt2 --task causal \
    --prompt-file prompts/01.txt --iters 1 --max-new-tokens 128
# exactly one of --prompt / --prompt-file is required
```

---

## Output format

| File                                   | Contents                                                      |
|----------------------------------------|---------------------------------------------------------------|
| `data/amps/<app>/<app>_<run>.txt`      | Provenance header, then migration-delay timestamps (cycles).  |
| `data/counter/<app>/<app>_<run>.txt`   | Provenance header, then access counters.                      |
| `data/metadata/<app>/<app>_<run>.txt`  | `run`, `app`, `avg_time`, `input_tokens`, `output_tokens`, `result` parsed from the victim's stdout. |

---

## Analysis

Once the `data/` trees are populated by collection, two scripts turn the traces
into token estimates and paper metrics:

- **`linear_regression.py`** — the detection + regression pipeline. It reads the
  **`data/counter/`** streams (the raw access-counter signal — *not* `data/amps/`)
  and the **`data/metadata/`** labels, locates the end-of-activity "wave" in each
  stream (`find_wave`), and fits wave-start → total-tokens. Parsed streams are
  cached as `.npy` under `data/_npy_cache/`; plots are written to `results/`.

  ```bash
  python3 linear_regression.py
  ```

- **`evaluate.py`** — imports `linear_regression` (as `D`) and reports k-fold CV
  metrics, detection-localization quality, and attack-realism curves, writing
  plots to `results/evaluation/`.

  ```bash
  python3 evaluate.py
  ```

Both expect runs `1..N_RUNS` per class, where **`N_RUNS = 71`** is set at the top
of `linear_regression.py`. Collect at least that many runs per app (the
collector's `NUM_RUNS` default of `100` covers it), or lower `N_RUNS` to match
what you collected.

**Analysis Python packages** (separate from the victim runner):

```bash
pip install numpy pandas scikit-learn scipy matplotlib
```

---

## Troubleshooting

- **`../build/migration_delay_side_channel not found`** — build it first
  (`cd .. && ./build.sh`).
- **Gated-model / 401 errors** — `huggingface-cli login`, or switch to an open
  model in `victim_apps_test.txt`.
- **CUDA out of memory (victim)** — use a smaller model or `--quant 4bit`;
  reduce `MemorySize` if the attacker buffer does not fit the slice.
- **`bitsandbytes` import/CUDA errors** — install a build matching your CUDA, or
  use `--quant none` (drops the `bitsandbytes` dependency).
- **Analysis can't find files / `FileNotFoundError` under `data/counter` or
  `data/metadata`** — run collection first, and make sure you collected at least
  `N_RUNS` (71) runs per class, or lower `N_RUNS` in `linear_regression.py`.
- **Stale analysis results after re-collecting** — delete `data/_npy_cache/`
  (and `results/evaluation/feature_table.csv`) so the cached streams/table are
  rebuilt.
