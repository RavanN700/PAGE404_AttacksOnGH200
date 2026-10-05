# Modified PAGE404 Driver

A modified build of NVIDIA's open GPU kernel modules (`open-gpu-kernel-modules`,
version **590.48.01**, `aarch64` / GH200) used for the PAGE404 attack research.

## What's modified

The change lives in the Unified Virtual Memory module (`nvidia-uvm`). A small
**CPU page cache** was added to it.

Normally, when UVM moves managed memory back from the GPU to the CPU, the kernel
is free to give the data a *different* physical page each time. This modified
driver instead remembers the page used for each virtual address and **reuses the
same physical page on every migration** — so a page's physical address stays
fixed across repeated GPU-CPU migrations instead of moving around.

This stable physical address is what makes the PAGE404 reverse engineering
experiments possible.

The behavior is **off by default** and must be turned on explicitly (see below).
When enabled, the driver also logs page activity to the kernel log for inspection.

## Building and installing

Requires the matching NVIDIA `.run` installer
(`NVIDIA-Linux-aarch64-590.48.01.run`) placed in `open-gpu-kernel-modules/`. You can get it from https://drive.google.com/file/d/1PD-81Lar_4nmeO-DfwAJbWpmli9DUL-e/view?usp=sharing
```bash
./driver_install.sh
```

This builds the modified kernel modules, installs them, and runs the NVIDIA
installer for the userspace components (without overwriting the kernel modules).

## Enabling the modification

The CPU page cache is controlled by the `uvm_cpu_page_cache_enable` module
parameter (0 = off, 1 = on). To turn it on:

```bash
./page_cache_enable.sh
```

This reloads `nvidia-uvm` with the cache enabled and prints the parameter value
to confirm it is active (`1`).

To watch the cache in action:

```bash
sudo dmesg | grep "\[UVM_CACHE\]"
```

## Scripts

| Script                  | Purpose                                                        |
| ----------------------- | -------------------------------------------------------------- |
| `driver_install.sh`     | Build and install the modified driver.                         |
| `page_cache_enable.sh`  | Reload `nvidia-uvm` with the CPU page cache enabled.           |
| `purge_nvidia.sh`       | Fully remove existing NVIDIA drivers, then reboot (run first on a clean setup). |

> **Note:** these scripts use `sudo` and modify kernel modules on the system.
