#!/usr/bin/env python3
"""
Reverse-engineer XOR-based set-hash functions from address groups.

Key insight: we don't need to know *which* set index each group maps to.
Within-group XOR differences must lie in the kernel of H.  Find the
kernel from those differences, then H = null-space complement.

Usage:  python3 xor_brute_force.py <pa_file>
"""

import sys


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_groups(filename):
    groups, cur = [], []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if '====' in line:
                if cur:
                    groups.append(cur)
                    cur = []
            elif line.lower().startswith('0x'):
                cur.append(int(line, 16))
    if cur:
        groups.append(cur)
    return groups


# ── GF(2) primitives ──────────────────────────────────────────────────────────

def gf2_rref(M_in):
    """
    In-place RREF over GF(2).
    Returns (rref_matrix, pivot_col_to_pivot_row dict).
    """
    M = [row[:] for row in M_in]
    rows = len(M)
    cols = len(M[0]) if rows else 0
    pivot_row = 0
    pc2pr = {}  # pivot_col -> pivot_row

    for col in range(cols):
        found = next((r for r in range(pivot_row, rows) if M[r][col]), None)
        if found is None:
            continue
        M[pivot_row], M[found] = M[found], M[pivot_row]
        pc2pr[col] = pivot_row
        for row in range(rows):
            if row != pivot_row and M[row][col]:
                for j in range(cols):
                    M[row][j] ^= M[pivot_row][j]
        pivot_row += 1

    return M, pc2pr


def gf2_null_space(M_in):
    """Null space of M over GF(2).  Returns list of basis vectors."""
    if not M_in:
        return []
    cols = len(M_in[0])
    M, pc2pr = gf2_rref(M_in)
    free_cols = [c for c in range(cols) if c not in pc2pr]
    null_vecs = []
    for fc in free_cols:
        v = [0] * cols
        v[fc] = 1
        for pc, pr in pc2pr.items():
            v[pc] = M[pr][fc]
        null_vecs.append(v)
    return null_vecs


def gf2_row_basis(vecs):
    """Return a linearly independent basis for the span of vecs (GF(2))."""
    if not vecs:
        return []
    M, pc2pr = gf2_rref(vecs)
    return [M[pr] for pr in sorted(pc2pr.values())]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python3 xor_brute_force.py <pa_file>")

    groups = parse_groups(sys.argv[1])
    n_groups = len(groups)
    n_out = (n_groups - 1).bit_length()

    print(f"Groups : {n_groups}  ->  {n_out} hash bits needed")
    for i, g in enumerate(groups):
        print(f"  group {i}: {len(g)} addresses")

    # Active (varying) bits
    all_or, all_and = 0, (1 << 64) - 1
    for g in groups:
        for a in g:
            all_or |= a
            all_and &= a
    vary = all_or & ~all_and
    active_bits = [i for i in range(64) if (vary >> i) & 1]
    n_bits = len(active_bits)

    print(f"\nVarying PA bits ({n_bits}): {active_bits}")

    def addr_vec(addr):
        return [(addr >> bp) & 1 for bp in active_bits]

    # ── Step 1: collect within-group differences ──────────────────────────────
    # Any two addresses in the same group satisfy  H*(a XOR b) = 0,
    # so their XOR lies in ker(H).  Collect enough to span ker(H).

    print("\nBuilding kernel from within-group XOR differences...")
    diff_vecs = []
    for g in groups:
        if len(g) < 2:
            continue
        rep_vec = addr_vec(g[0])
        for a in g[1:]:
            av = addr_vec(a)
            dv = [x ^ y for x, y in zip(rep_vec, av)]
            if any(dv):
                diff_vecs.append(dv)

    kernel_basis = gf2_row_basis(diff_vecs)
    ker_dim = len(kernel_basis)
    expected_ker_dim = n_bits - n_out

    print(f"  Collected {len(diff_vecs)} raw difference vectors")
    print(f"  Kernel dimension : {ker_dim}  (expected {expected_ker_dim})")

    if ker_dim > expected_ker_dim:
        print(f"\n  WARNING: Kernel is larger than expected ({ker_dim} > {expected_ker_dim}).")
        print("  Some groups may be merged sets, or extra PA bits are needed.")
    elif ker_dim < expected_ker_dim:
        print("\n  WARNING: Not enough independent differences collected.")
        print("  Need more addresses per group.")

    # ── Step 2: H = null space of kernel ─────────────────────────────────────
    print("\nComputing hash matrix (null space of kernel)...")
    hash_rows = gf2_null_space(kernel_basis)
    print(f"  Hash matrix rows : {len(hash_rows)}  (expected {n_out})")

    if len(hash_rows) != n_out:
        print(f"\nERROR: Could not recover {n_out}-bit hash function.")
        print(f"  Got {len(hash_rows)} hash rows — check that all PA bits are present.")
        return

    # ── Step 3: hash every address, determine group->set mapping ─────────────
    def hash_addr(addr):
        bv = addr_vec(addr)
        h = 0
        for b, row in enumerate(hash_rows):
            v = sum(row[i] & bv[i] for i in range(n_bits)) & 1
            h |= v << b
        return h

    print("\nMapping groups to set indices...")
    group_to_set = {}
    set_to_group = {}
    all_ok = True

    for g_idx, group in enumerate(groups):
        hvals = {hash_addr(a) for a in group}
        if len(hvals) != 1:
            print(f"  ERROR group {g_idx} hashes to {len(hvals)} different values: {hvals}")
            all_ok = False
            continue
        hval = next(iter(hvals))
        if hval in set_to_group:
            print(f"  ERROR hash value {hval} appears in both group {set_to_group[hval]} and group {g_idx}")
            all_ok = False
            continue
        group_to_set[g_idx] = hval
        set_to_group[hval] = g_idx

    if all_ok:
        print(f"  All {n_groups} groups map to distinct set indices ✓")

    # ── Verification ──────────────────────────────────────────────────────────
    correct = total = 0
    for g_idx, group in enumerate(groups):
        expected = group_to_set.get(g_idx, -1)
        for a in group:
            total += 1
            if hash_addr(a) == expected:
                correct += 1

    print(f"\nVerification: {correct}/{total} ({100*correct/total:.1f}%) correct")
    if correct < total:
        print("WARNING: mismatches remain — hash may have a non-linear component.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Hash function (bit 0 = LSB of physical address):")
    for b, row in enumerate(hash_rows):
        used = [active_bits[i] for i, v in enumerate(row) if v]
        terms = " ^ ".join(f"PA[{bp}]" for bp in used)
        print(f"  hash_bit[{b}] = {terms}")

    print("\nGroup -> set index mapping:")
    for g_idx in range(n_groups):
        s = group_to_set.get(g_idx, '?')
        print(f"  group {g_idx:2d}  ->  set {s}")


if __name__ == "__main__":
    main()