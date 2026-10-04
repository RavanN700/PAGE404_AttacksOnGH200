#!/usr/bin/env python3
"""Drive side-channel collection for every victim command in victim_apps_test.txt.

For each app and each run it calls collection_script.sh, which pins the attacker
collector and the victim LLM to two different MIG slices (edit that script to set
your slice UUIDs). Three output trees are produced:

    data/amps/<app>/<app>_<run>.txt      migration-delay timestamps
    data/counter/<app>/<app>_<run>.txt   access counters
    data/metadata/<app>/<app>_<run>.txt  parsed victim stats (tokens, timing)

App names are derived automatically from each line's --prompt-file path, e.g.
    python3 run_llm_inference.py ... --prompt-file prompts/01.txt ...
  => app name "01"
"""
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

# Files/paths (relative to this script's directory).
SCRIPT_DIR            = Path(__file__).resolve().parent
VICTIM_FILE           = SCRIPT_DIR / "prompts/victim_apps_test.txt"   # one victim cmd per line
COLLECTION_SCRIPT     = SCRIPT_DIR / "collection_script.sh"
SAMPLES_ROOT_AMPS     = SCRIPT_DIR / "data/amps/"
SAMPLES_ROOT_COUNTER  = SCRIPT_DIR / "data/counter/"
SAMPLES_ROOT_METADATA = SCRIPT_DIR / "data/metadata/"

# Collection parameters.
NUM_RUNS   = 100       # repetitions per app
MemorySize = "90GB"    # attacker buffer size (passed to the collector)
N_ACCESSES = "256"     # accesses probed per page


def derive_app_name(victim_cmd: str) -> str:
    """Return the stem of the --prompt-file path in a shell command line.

    Uses shlex so quoted paths with spaces still parse; falls back to a regex
    for malformed lines. Raises ValueError if no --prompt-file is present.
    """
    try:
        toks = shlex.split(victim_cmd)
    except ValueError:
        toks = victim_cmd.split()

    for i, tok in enumerate(toks):
        if tok == "--prompt-file" and i + 1 < len(toks):
            return Path(toks[i + 1]).stem
        if tok.startswith("--prompt-file="):
            return Path(tok.split("=", 1)[1]).stem

    m = re.search(r"--prompt-file[=\s]+(\S+)", victim_cmd)
    if m:
        return Path(m.group(1)).stem

    raise ValueError(f"No --prompt-file argument found in: {victim_cmd!r}")


def read_victim_lines(victim_file: Path) -> list:
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


def main() -> None:
    if not COLLECTION_SCRIPT.exists():
        print(f"Error: cannot find {COLLECTION_SCRIPT}", file=sys.stderr)
        sys.exit(1)
    if not ensure_executable(COLLECTION_SCRIPT):
        sys.exit(1)
    if not VICTIM_FILE.exists():
        print(f"Error: cannot find {VICTIM_FILE}", file=sys.stderr)
        sys.exit(1)

    victim_lines = read_victim_lines(VICTIM_FILE)
    if not victim_lines:
        print(f"Error: {VICTIM_FILE} is empty", file=sys.stderr)
        sys.exit(1)

    # Derive a unique app name from each line's --prompt-file.
    pairs = []            # list of (app_name, victim_cmd)
    seen_names = set()
    for n, line in enumerate(victim_lines, 1):
        try:
            app = derive_app_name(line)
        except ValueError as e:
            print(f"Error on line {n}: {e}", file=sys.stderr)
            sys.exit(1)
        if app in seen_names:
            print(f"Error: duplicate app name {app!r} (line {n}). "
                  f"prompt-file stems must be unique.", file=sys.stderr)
            sys.exit(1)
        seen_names.add(app)
        pairs.append((app, line))

    print(f"Derived {len(pairs)} app names from {VICTIM_FILE.name}:")
    for app, _ in pairs:
        print(f"  - {app}")
    print()

    # Pre-create per-app output directories.
    for app, _ in pairs:
        (SAMPLES_ROOT_AMPS     / app).mkdir(parents=True, exist_ok=True)
        (SAMPLES_ROOT_COUNTER  / app).mkdir(parents=True, exist_ok=True)
        (SAMPLES_ROOT_METADATA / app).mkdir(parents=True, exist_ok=True)

    # Iterate run-by-run across all apps.
    for i in range(1, NUM_RUNS + 1):
        print(f"\n========== Run {i}/{NUM_RUNS} ==========")
        for app, victim_cmd in pairs:
            print(f"\n=== Run {i} | App: {app} ===")
            print(f"Victim command: {victim_cmd}")

            pointer_file_amps    = SAMPLES_ROOT_AMPS    / app / f"{app}_{i}.txt"
            pointer_file_counter = SAMPLES_ROOT_COUNTER / app / f"{app}_{i}.txt"

            # The collector requires its output files to already exist; seed
            # them with a small provenance header.
            header = f"run={i}\napp={app}\nvictim_cmd={victim_cmd}\n"
            pointer_file_amps.write_text(header, encoding="utf-8")
            pointer_file_counter.write_text(header, encoding="utf-8")

            cmd = [
                str(COLLECTION_SCRIPT),
                victim_cmd,
                str(pointer_file_amps.resolve()),
                str(pointer_file_counter.resolve()),
                MemorySize,
                N_ACCESSES,
            ]
            print(f"Executing: {' '.join(cmd)}")
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                output = result.stdout + result.stderr
                print(result.stdout, end="")

                # Parse victim-side stats from run_llm_inference.py output.
                m_time   = re.search(r"Avg time:\s*([\d.]+)s/iter", output)
                m_in     = re.search(r"Input tokens:\s*(\d+)", output)
                m_out    = re.search(r"Output tokens:\s*(\d+)", output)
                m_result = re.search(
                    r"\[run_llm_inference\] Result:\n(.*?)(?=\ntotal elapsed|\nAvg time:)",
                    output, re.DOTALL)

                metadata_payload = (
                    f"run={i}\napp={app}\n"
                    f"avg_time={m_time.group(1) if m_time else 'N/A'}\n"
                    f"input_tokens={m_in.group(1) if m_in else 'N/A'}\n"
                    f"output_tokens={m_out.group(1) if m_out else 'N/A'}\n"
                    f"result={m_result.group(1).strip() if m_result else 'N/A'}\n"
                )
                (SAMPLES_ROOT_METADATA / app / f"{app}_{i}.txt").write_text(
                    metadata_payload, encoding="utf-8")
            except subprocess.CalledProcessError as e:
                print(f"ERROR: {COLLECTION_SCRIPT.name} failed for {app} run {i} "
                      f"(exit {e.returncode})", file=sys.stderr)
                # continue to the next app/run
            except KeyboardInterrupt:
                print("\nInterrupted by user.")
                sys.exit(130)

    print("\nAll runs completed.")


if __name__ == "__main__":
    main()
