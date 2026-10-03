# Reverse-engineering the GH200 hardware access counters set-index hash functions

Tools for recovering the **physical-address → set-index hash function** for hardware access counters on an
NVIDIA **GH200 (Hopper)** system. The CUDA programs build *eviction sets* —
groups of physical pages that map to the same counter set — and the Python
script reverse-engineers the XOR hash that produces those sets.

---

## Overview

On GH200, unified memory lets a page live on the CPU or migrate to the GPU.
A page migrates once a hardware access counter for that specific page crosses a
threshold (default is 256).  Each
tool uses a GPU pointer-chase to drive accesses and `/proc/self/pagemap` to
watch whether the target's physical address changed (migrated). Collect enough
same-set page groups and the set-index hash can be solved for directly.

---

## Layout

| File | What it does |
|------|--------------|
| `eviction_set_builder.cu`        | Build a single eviction set for one target page. |
| `reaction_to_counter_values.cu`  | Check the reaction of eviction policy based on counter values. |
| `produce_unique_sets.cu`         | Discover the unique eviction sets across all sets. |
| `collect_set_of_addresses.cu`    | Collect `TOTAL_SETS` unique sets and dump their physical addresses. |
| `xor_brute_force.py`             | Reverse-engineer the XOR set-hash from collected address groups (GF(2) linear algebra). |
| `precollected_addresses_per_sets.txt` | Example input for `xor_brute_force.py` (real captured data). |
| `build.sh`, `CMakeLists.txt`     | Build the CUDA tools. |

---

## Prerequisites

- NVIDIA **GH200** (Hopper, compute capability **sm_90**).
- **CUDA toolkit** (tested with 12.2) on `PATH`.
- A working host C++ compiler. On this machine the default `gcc` is v12 but only
  `g++-11` has a usable `cc1plus`, so the build points `nvcc` at `g++-11`.
- **root** — the tools read `/proc/self/pagemap` for virtual→physical
  translation, which requires elevated privileges.
- Python **3** for the reverse-engineering step (standard library only).

---

## Build

```bash
./build.sh                 # build every *.cu  -> ./build/<name>
./build.sh eviction_set_builder.cu   # build just one
./build.sh clean           # remove ./build/
```

Overrides: `ARCH=sm_90`, `CCBIN=g++-11`, `NVCC=nvcc` (environment variables).

Or with CMake:

```bash
cmake -B build && cmake --build build
```

All binaries land in `build/`.

---

## Running

Every CUDA tool takes the same options (all optional, sensible defaults shown):

| Flag | Long form | Default | Meaning |
|------|-----------|---------|---------|
| `-M` | `--num-pages`          | 8   | Number of distractor pages. |
| `-N` | `--accesses-thr`       | 256 | Access threshold for migration. |
| `-d` | `--delta-n`            | 6   | Access-count margin. |
| `-p` | `--page-size-mb`       | 2   | MB page size. No need to change. |
| `-E` | `--eviction-set-size`  | 16  | Target eviction-set size. |
| `-T` | `--test`               | 5   | Repeat count for confirmation. |

Each tool must be run from a directory where it can create a `./texts/`
subfolder (it is created automatically) — that is where results are written.
Run as root:

```bash
cd page404_re
sudo ./build/collect_set_of_addresses
```

### Typical pipeline

```text
collect_set_of_addresses   ──►  ./texts/all_sets_pa      (groups of same-set PAs)
                                        │
                                        ▼
xor_brute_force.py  ──►  recovered hash_bit[k] = PA[i] ^ PA[j] ^ ...
```

```bash
# 1. Collect same-set address groups on the GPU (writes ./texts/all_sets_pa)
sudo ./build/collect_set_of_addresses

# 2. Reverse-engineer the XOR hash from those groups
python3 xor_brute_force.py ./texts/all_sets_pa

#    …or try it immediately on the bundled example capture:
python3 xor_brute_force.py precollected_addresses_per_sets.txt
```

---

## Output files (`./texts/`)

| File | Written by | Contents |
|------|-----------|----------|
| `all_sets_pa`               | `collect_set_of_addresses` | Physical addresses grouped by set, separated by `====` lines. |
| `same_set_pas_2.txt`        | `eviction_set_builder`     | Physical addresses of the recovered eviction set. |
| `eviction_set_sizes_found_<k>` | `reaction_to_counter_values` | Set sizes found while sweeping parameters. |
| `hw_counters_number_guess`  | all tools | Raw per-access latency timings. |
| `migration_results`         | all tools | Per-trial migration flags (1 = distracted counter, 0 = migrated). |
| `migration_points`          | all tools | Access index at which each migration was detected. |

The whole `texts/` folder (and `build/`) is git-ignored; only
`precollected_addresses_per_sets.txt` is kept as a checked-in example.

---

## Input format for `xor_brute_force.py`

Plain text: one physical address per line as `0x...`, with groups of same-set
addresses separated by any line containing `====`:

```text
0x000000026da60000
0x0000006ee7b40000
...
============================================
0x00000070a0100000
...
```

The script finds, over GF(2):

1. the **kernel** from within-group XOR differences (same-set pages differ only
   in bits the hash ignores), then
2. the **hash matrix** as the null-space complement, and prints each output bit
   as an XOR of physical-address bits, e.g. `hash_bit[0] = PA[16] ^ PA[17]`.

It also verifies every address re-hashes to its group and reports the match rate.

---

## Notes & gotchas

- **Run from `page404_re/`** so relative `./texts/` paths resolve.
- Results are hardware-specific; expect run-to-run variation
  and tune `-N` / `-d` for your system if migrations are not detected.
- `va_to_pa()` returns 0 and warns if a page is not resident — root is required
  for real PFNs from `pagemap`.
