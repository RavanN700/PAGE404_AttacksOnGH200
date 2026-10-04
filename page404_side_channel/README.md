# PAGE404 — Migration-Delay Side Channels (GH200)

Proof-of-concept data-collection code for the **migration-delay side channel**
described in the PAGE404. An
attacker process on one MIG slice observes **page migrations** caused by a
victim process on a *neighbouring* MIG slice of the same physical GPU, purely by
timing its own page migrations.

---

## The signal

Both slices draw from the same physical memory. When the victim's activity
causes a page to migrate (become locally resident), the attacker's **next access
to that page briefly gets faster**. The shared collector
([`migration_delay_side_channel.cu`](migration_delay_side_channel.cu)) walks a
large buffer one 128 KB page at a time, repeatedly timing accesses to each page,
and records two per-page series:

- **migration delay** — the `clock64` timestamp at which the fast (migrated)
  access was observed;
- **access counter** — how many probe accesses were issued before that point.

Together these form the raw side-channel trace. What the trace reveals depends
on the victim; see the PoCs below.

---

## The two PoCs

| Directory | What it demonstrates |
|-----------|----------------------|
| [`prompt_length_estimate/`](prompt_length_estimate/) | Estimating the **prompt length** of a victim LLM inference from the migration trace. See its [README](prompt_length_estimate/README.md). |
| [`llm_fingerprinting/`](llm_fingerprinting/) | **Fingerprinting the victim model** (which of a known set is running) from its migration trace. See its [README](llm_fingerprinting/README.md). |

Both PoCs use the **same collector binary**, built once from the shared CUDA
source in this directory.

---

## Repository layout

| Path                              | Purpose                                                        |
|-----------------------------------|----------------------------------------------------------------|
| `migration_delay_side_channel.cu` | Shared collector: probes pages and records the migration trace.|
| `build.sh`                        | Builds the collector into `./build/`.                          |
| `prompt_length_estimate/`         | PoC 1 — orchestration, victim runner, prompts, analysis.       |
| `llm_fingerprinting/`             | PoC 2 — orchestration, victim runner, prompt generator.        |

---

## Requirements

- **GPU:** NVIDIA GH200 (Hopper, `sm_90`) with **MIG enabled** and at least
  **two** MIG slices (one attacker, one victim).
- **CUDA toolkit:** tested with **CUDA 12.2** (`nvcc`); NVIDIA driver with MIG.
- **Host compiler:** `g++-11` (override with `CCBIN=`).
- **OS:** Linux.

Unlike the covert-channel tool, the collector does **not** read
`/proc/self/pagemap`, so it does not require root. Each PoC has its own
additional requirements (e.g. the Python / PyTorch stack for the victim LLM) —
see the PoC's README.

---

## Build

```bash
./build.sh          # builds build/migration_delay_side_channel
./build.sh clean    # removes ./build/
```

Environment overrides: `ARCH` (default `sm_90`), `NVCC` (default `nvcc`),
`CCBIN` (default `g++-11`). Example: `CCBIN=g++-12 ./build.sh`.

Standalone usage of the collector (normally a PoC script runs it for you):

```bash
build/migration_delay_side_channel <time_out> <counter_out> [mem_size] [N]
#   time_out    file for migration-delay timestamps (one per line)
#   counter_out file for access counters (one per line)
#   mem_size    buffer size, e.g. 90GB or 128MB (default 128MB)
#   N           accesses probed per page (default 256)
```

---

## Quick start (PoC 1)

```bash
./build.sh                         # build the shared collector
cd prompt_length_estimate
# edit MIG UUIDs in collection_script.sh (see: nvidia-smi -L)
python3 auto_collection.py         # runs victim + collector, writes data/
```

See [`prompt_length_estimate/README.md`](prompt_length_estimate/README.md) for
the full setup, Python requirements, and output format.
