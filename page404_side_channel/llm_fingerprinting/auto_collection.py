#!/usr/bin/env python3
"""Drive side-channel collection for LLM fingerprinting.

For each victim model and each run, this launches collection_script.sh, which
pins the attacker collector and the victim LLM to two different MIG slices (edit
that script to set your slice UUIDs). A fresh random prompt (from promptgen) is
used per run so the trace reflects the model, not a memorized prompt.

Outputs:
    data/amps/<app>/<app>_<run>.txt      migration-delay timestamps
    data/counter/<app>/<app>_<run>.txt   access counters
    data/prompts/prompts.txt             log of every prompt used, per app

Each victim command in victim_apps.txt is matched to an app name in VICTIM_LIST
*by line order*, so the two lists must stay aligned. Victim commands contain a
`{PROMPT}` placeholder that is substituted with the generated prompt.
"""
import os
import subprocess
import sys
from pathlib import Path

from promptgen import generate_prompt

# App names, matched to victim_apps.txt lines by position (keep them aligned).
VICTIM_LIST = [
    "flan_t5_large",
    "flan_t5_base",
    "llama3_1b",
    "gptneo_1p3b",
    "gptneo_125m",
    "pythia_1b",
    "pythia_410m",
    "distilgpt2",
    "opus_mt_en_de",
    "bart_base",
    "bart_large",
    "pegasus_xsum",
    "switch_base_8",
    "phi_1_5",
    "qwen3_0p6b",
    "falcon_e_3b",
    "falcon_rw_1b",
    "gemma_1b_it",
    "opt_1p3b",
    "luth_0p6b",
]

# Files/paths (relative to this script's directory).
SCRIPT_DIR           = Path(__file__).resolve().parent
VICTIM_FILE          = SCRIPT_DIR / "victim_apps.txt"
COLLECTION_SCRIPT    = SCRIPT_DIR / "collection_script.sh"
SAMPLES_ROOT_AMPS    = SCRIPT_DIR / "data/amps/"
SAMPLES_ROOT_COUNTER = SCRIPT_DIR / "data/counter/"
PROMPTS_LOG_DIR      = SCRIPT_DIR / "data/prompts/"

# Collection parameters.
NUM_RUNS   = 100       # runs per app
MemorySize = "90GB"    # attacker buffer size (passed to the collector)


def read_victim_lines(victim_file: Path):
    """Return non-empty, non-comment lines from the victim command file."""
    lines = []
    with victim_file.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if s and not s.startswith("#"):
                lines.append(s)
    return lines


def ensure_executable(path: Path) -> bool:
    """Ensure `path` exists and is executable, chmod-ing it if needed."""
    if not path.exists():
        print(f"Error: {path} does not exist", file=sys.stderr)
        return False
    if not os.access(path, os.X_OK):
        try:
            path.chmod(path.stat().st_mode | 0o111)
            print(f"Made {path} executable")
        except OSError as e:
            print(f"Warning: could not make {path} executable: {e}", file=sys.stderr)
            return False
    return True


def main():
    if not COLLECTION_SCRIPT.exists():
        print(f"Error: cannot find {COLLECTION_SCRIPT}", file=sys.stderr)
        sys.exit(1)
    if not ensure_executable(COLLECTION_SCRIPT):
        sys.exit(1)
    if not VICTIM_FILE.exists():
        print(f"Error: cannot find {VICTIM_FILE}", file=sys.stderr)
        sys.exit(1)

    victim_lines = read_victim_lines(VICTIM_FILE)
    if len(victim_lines) < len(VICTIM_LIST):
        print(f"Error: {VICTIM_FILE} has only {len(victim_lines)} lines, "
              f"but VICTIM_LIST needs {len(VICTIM_LIST)}.", file=sys.stderr)
        sys.exit(1)

    # Map each app to its victim command template by line order.
    app_to_victim_template = {app: victim_lines[i] for i, app in enumerate(VICTIM_LIST)}

    # One shared prompt log for all apps/runs.
    PROMPTS_LOG_DIR.mkdir(parents=True, exist_ok=True)
    prompt_log_file = PROMPTS_LOG_DIR / "prompts.txt"
    prompt_log_file.write_text("# app | prompt\n", encoding="utf-8")

    for app in VICTIM_LIST:
        victim_template = app_to_victim_template[app]
        print(f"\n=== App: {app} ===")

        app_dir_amps    = SAMPLES_ROOT_AMPS    / app
        app_dir_counter = SAMPLES_ROOT_COUNTER / app
        app_dir_amps.mkdir(parents=True, exist_ok=True)
        app_dir_counter.mkdir(parents=True, exist_ok=True)

        for i in range(1, NUM_RUNS + 1):
            prompt = generate_prompt()
            victim_cmd = victim_template.replace("{PROMPT}", prompt)
            print(f"Victim command: {victim_cmd}")

            with prompt_log_file.open("a", encoding="utf-8") as f:
                f.write(f"{app} | {prompt}\n")

            # One file per run for both outputs (seeded so the collector, which
            # opens them for writing, finds them existing).
            pointer_file_amps    = app_dir_amps    / f"{app}_{i}.txt"
            pointer_file_counter = app_dir_counter / f"{app}_{i}.txt"
            header = f"run={i}\napp={app}\nvictim_cmd={victim_cmd}\n"
            pointer_file_amps.write_text(header, encoding="utf-8")
            pointer_file_counter.write_text(header, encoding="utf-8")

            print(f"[{app}] Run {i}/{NUM_RUNS} -> amps:    {pointer_file_amps.resolve()}")
            print(f"[{app}] Run {i}/{NUM_RUNS} -> counter: {pointer_file_counter.resolve()}")

            cmd = [
                str(COLLECTION_SCRIPT),
                victim_cmd,
                str(pointer_file_amps.resolve()),
                str(pointer_file_counter.resolve()),
                MemorySize,
            ]
            print(f"Executing: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"ERROR: {COLLECTION_SCRIPT.name} failed for {app} run {i} "
                      f"(exit {e.returncode})", file=sys.stderr)
                # continue to the next run
            except KeyboardInterrupt:
                print("\nInterrupted by user.")
                sys.exit(130)

    print("\nAll runs completed.")


if __name__ == "__main__":
    main()
