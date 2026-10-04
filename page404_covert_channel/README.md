# PAGE404 — Eviction-Based Covert Channel (GH200)

Reproduction code for the **cross-MIG covert channel** described in the PAGE404
study of the NVIDIA GH200 (Hopper, `sm_90`). A **sender** and a **receiver**
run on two *different* MIG slices of the same physical GPU and communicate by
modulating contention on shared L2 cache sets — no explicit IPC, shared memory,
or network between them.

> **Research artifact.** This is reproduction code for a published, responsibly
> disclosed academic result (accepted to IEEE S&P 2027; disclosed to NVIDIA,
> which does not classify it as a security issue). It targets a specific lab
> setup and is **not** a deployable or general-purpose attack tool.

---

## How it works

The two processes share the GPU's L2 cache even though MIG isolates their
compute and memory allocations. Each L2 **set** is selected by a set of parity
functions over the *physical* address (`MASK0..2` in the source). Both sides
recover physical addresses via `/proc/self/pagemap` and pick buffers that land
in the same target set.

Communication happens in fixed-length **time slots**, one bit per slot:

| Bit | Sender does…                              | Receiver observes…                |
|-----|-------------------------------------------|-----------------------------------|
| `1` | Hammers a rotating series of eviction sets, evicting the receiver's probe lines | A **slow** probe access (line was evicted) |
| `0` | Stays idle for the slot                   | A **fast** probe access (line still resident) |

A short run of synchronization `1` bits at the start lets the receiver lock onto
the sender's slot boundaries before the payload begins. Multiple independent
channels ("sets") can run in parallel, each targeting a different cache set.

---

## Repository layout

| File                       | Purpose                                                        |
|----------------------------|----------------------------------------------------------------|
| `sender.cu`                | Sender: transmits a random bit stream per channel.             |
| `receiver.cu`              | Receiver: times probe accesses and decodes the bit stream.     |
| `build.sh`                 | Builds both binaries with `nvcc` into `./build/`.              |
| `launch_overt_channel.sh`  | Launches sender + receiver, each pinned to its own MIG slice.  |
| `error_rate.py`            | Compares sent vs. received streams (edit-distance error rate). |

Output traces are written to `./texts/parallel_channel/` at run time.

---

## Requirements

- **GPU:** NVIDIA GH200 (Hopper, `sm_90`) with **MIG enabled** and at least
  **two** MIG slices available.
- **CUDA toolkit:** tested with **CUDA 12.2** (`nvcc`).
- **Host compiler:** `g++-11` (override with `CCBIN=` — see below).
- **OS:** Linux (developed on an arm64 host with 64 KB base pages).
- **Privileges:** **root** (or `CAP_SYS_ADMIN`). The physical-address lookup
  reads PFNs from `/proc/self/pagemap`, which the kernel zeroes for unprivileged
  processes. Without this, the cache-set targeting cannot work.
- **Huge pages:** Transparent Huge Pages should be available
  (`madvise(MADV_HUGEPAGE)` is used over a 512 MB-aligned region).

---

## Build

```bash
./build.sh          # builds both tools into ./build/
./build.sh clean    # removes ./build/
```

The script invokes `nvcc` directly (no CMake needed) and produces:

```
build/sender_eviction_parallel_sets
build/receiver_eviction_parallel_sets
```

Environment overrides:

| Variable | Default  | Meaning                               |
|----------|----------|---------------------------------------|
| `ARCH`   | `sm_90`  | Target GPU architecture.              |
| `NVCC`   | `nvcc`   | Compiler to invoke.                   |
| `CCBIN`  | `g++-11` | Host C++ compiler passed to `nvcc`.   |

Example: `ARCH=sm_90 CCBIN=g++-12 ./build.sh`

---

## Configure the MIG slices

1. List your MIG slices:

   ```bash
   nvidia-smi -L
   ```

   Look for lines such as:

   ```
   MIG 1g.12gb  Device  0:  (UUID: MIG-48b2f4b8-...)
   MIG 1g.12gb  Device  1:  (UUID: MIG-97f521d2-...)
   ```

2. Edit `launch_overt_channel.sh` and set two **different** slice UUIDs:

   ```bash
   RECEIVER_MIG="MIG-...your receiver slice..."
   SENDER_MIG="MIG-...your sender slice..."
   ```

   The script refuses to run if the two are identical.

---

## Run

```bash
sudo ./launch_overt_channel.sh [num_channels]
```

- `num_channels` (optional, default `1`) is passed as `--num-sets` to both
  binaries and sets how many parallel channels run.
- The script creates `./texts/parallel_channel/`, starts the **receiver first**
  (it waits for the sender's sync bits), then the sender, and waits for both.

The binaries print progress (`Receiver started`, `Sender started`, `…finished`)
and the receiver prints an estimated **bandwidth** in bits/s at the end.

---

## Output files

Written to `./texts/parallel_channel/`, one set of files per channel `<i>`:

| File                          | Written by | Contents                                            |
|-------------------------------|------------|-----------------------------------------------------|
| `sender_message_<i>`          | sender     | The payload bits actually transmitted.              |
| `sender_measurement_<i>`      | sender     | Per-slot timing (cycles) on the sender side.        |
| `message_bits_<i>`            | receiver   | Recovered payload bits (after sync lock).           |
| `receiver_all_messages_<i>`   | receiver   | Full decoded stream, including pre-sync bits.       |
| `receiver_measurements_<i>`   | receiver   | Per-slot timing (cycles) on the receiver side.      |

---

## Measure the error rate

Compare what was sent against what was received:

```bash
python3 error_rate.py <num_bits> <num_channels> [start]
```

- `num_bits` — length of the receiver window to compare.
- `num_channels` — number of channels to evaluate (matches your run).
- `start` — 1-based offset into the receiver stream (default `1`), useful to
  skip leading noise before alignment.

It reports a per-channel and average error rate using **Levenshtein (edit)
distance** normalized by the received length. Edit distance is used instead of a
bit-by-bit comparison so that an occasional inserted or dropped bit does not
desynchronize the entire comparison.

---

## Tuning: number of channels

There are **two** related knobs:

1. **Compile-time timing** — `NUM_CHANNELS` selects the slot length and spacing
   constants (`IDLE_TIME`, `BIT1_DELAY`, `STARTING_DELAY`, etc.). It defaults to
   **1**. The sender and receiver **must be compiled with the same value**:

   ```bash
   nvcc -O3 -std=c++17 -arch=sm_90 -ccbin g++-11 -DNUM_CHANNELS=4 sender.cu   -o build/sender_eviction_parallel_sets
   nvcc -O3 -std=c++17 -arch=sm_90 -ccbin g++-11 -DNUM_CHANNELS=4 receiver.cu -o build/receiver_eviction_parallel_sets
   ```

   (Supported: `1`–`4`. Add `-DNUM_CHANNELS=…` to the `nvcc` line in `build.sh`
   to make it the default.)

2. **Run-time channel count** — `--num-sets` / the `launch` argument controls how
   many channels actually run. For meaningful results, keep it consistent with
   the `NUM_CHANNELS` the binaries were compiled for.

Other flags accepted by the binaries (see `main()` in each source):

- **sender:** `--threshold --num-eviction-sets --gpu --delta-n --num-bits --num-sets`
- **receiver:** `--threshold --num-pages --num-bits --gpu --delta-n --num-sets`

The values used for the paper's runs are the ones hard-coded in
`launch_overt_channel.sh`.

---

## Troubleshooting

- **`Page not present in RAM!` / all-zero PAs** — you are not running as root, or
  `/proc/self/pagemap` is unreadable. Run with `sudo`.
- **`Not correctly collected`** — not enough pages mapped to the target set were
  found; ensure huge pages are available and the buffer is large enough
  (`--num-pages` on the receiver).
- **High error rate** — sender and receiver timing is sensitive. Make sure both
  were built with the **same** `NUM_CHANNELS`, that the two MIG slices are
  distinct, and that the GPU is otherwise idle.
- **Segfault writing output** — the output directory must exist. The launch
  script creates it; if you run a binary directly, first
  `mkdir -p texts/parallel_channel`.
