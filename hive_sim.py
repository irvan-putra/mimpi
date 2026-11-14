#!/usr/bin/env python3
"""
hive_sim.py - toy simulation of the mimpi hive-mind aggregation loop.

Simulates the federated part of the proposal in proposal-gemini.md at toy
scale: N intermittently heterogeneous clients each train a small local
"adapter delta" on private synthetic data, compress it (Top-K sparsification
+ 8-bit quantization, as the proposal suggests), and a coordinator aggregates
deltas two ways:

  * naive mean
  * trimmed mean with a cosine-similarity filter (the proposal's
    Byzantine-robust sketch)

Some clients are adversarial and send inverted updates.

THIS IS A TOY SIMULATION ON SYNTHETIC DATA. It demonstrates aggregation
mechanics, not the real system. The small linear model stands in for a LoRA
adapter; nothing here is a measured performance claim about the proposed
companion, and the feasibility review's open research questions (merging
LoRA factors across heterogeneous clients) are NOT solved by this toy.

Python 3.9+ with numpy. --viz needs matplotlib.

Usage:
    python3 hive_sim.py                       # 30 rounds, 12 clients, 2 Byzantine
    python3 hive_sim.py --rounds 50 --byzantine 4
    python3 hive_sim.py --viz hive_sim.png
"""

import argparse
import sys

import numpy as np


def make_data(n_clients, n_per_client, seed, rotation_range):
    """Non-IID synthetic 2D classification data: each client's class boundary
    is rotated by a different angle, so clients disagree about the model.
    Larger `rotation_range` = more heterogeneous clients."""
    rng = np.random.default_rng(seed)
    clients = []
    for i in range(n_clients):
        angle = rng.uniform(-rotation_range, rotation_range)
        rot = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
        X = rng.normal(0, 1.5, size=(n_per_client, 2))
        y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(float)
        X = X @ rot.T
        clients.append((X, y))
    # shared held-out test set with the canonical boundary
    Xt = rng.normal(0, 1.5, size=(400, 2))
    yt = (Xt[:, 0] + 0.5 * Xt[:, 1] > 0).astype(float)
    return clients, (Xt, yt)



def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def local_train(w_global, X, y, steps, lr, rng):
    """A few SGD steps of logistic regression from the global weights.
    The returned delta stands in for a LoRA pseudo-gradient."""
    w = w_global.copy()
    n = len(y)
    Xb = np.hstack([X, np.ones((n, 1))])
    for _ in range(steps):
        idx = rng.integers(0, n, size=16)
        grad = Xb[idx].T @ (sigmoid(Xb[idx] @ w) - y[idx]) / len(idx)
        w -= lr * grad
    return w - w_global


def compress(delta, keep_frac=0.2):
    """Top-K sparsification + 8-bit quantization (proposal's upload sketch).

    Note: the toy delta has 3 parameters, so k = max(1, floor(0.2*3)) = 1 -
    the toy keeps 1 of 3 coordinates (33%), not 20%. The 20% figure only
    applies at realistic adapter sizes."""
    flat = delta.copy()
    k = max(1, int(len(flat) * keep_frac))
    thresh = np.partition(np.abs(flat), -k)[-k]
    flat[np.abs(flat) < thresh] = 0.0
    scale = np.abs(flat).max() or 1.0
    q = np.round(flat / scale * 127)
    return q / 127 * scale


def cosine_filter(deltas):
    """Drop deltas whose cosine to the coordinate-wise median is negative."""
    med = np.median(deltas, axis=0)
    keep = []
    for i, d in enumerate(deltas):
        denom = (np.linalg.norm(d) * np.linalg.norm(med)) or 1e-12
        if float(d @ med) / denom >= 0:
            keep.append(i)
    return deltas[keep] if keep else deltas


def trimmed_mean(deltas, trim=0.2):
    """Genuine coordinate-wise trimmed mean: sort each coordinate across
    clients, discard the top and bottom `trim` fraction, average the rest."""
    n = len(deltas)
    k = int(n * trim)
    if n - 2 * k < 1:
        return deltas.mean(axis=0)
    return np.sort(deltas, axis=0)[k:n - k].mean(axis=0)


def accuracy(w, X, y):
    Xb = np.hstack([X, np.ones((len(y), 1))])
    return float(np.mean((Xb @ w > 0) == y))


def run_one(args, seed):
    clients, (Xt, yt) = make_data(args.clients, 60, seed, args.rotation_range)
    byz = set(np.random.default_rng(seed + 1)
              .choice(args.clients, size=args.byzantine, replace=False).tolist())

    w_naive = np.zeros(3)
    w_robust = np.zeros(3)
    hist = {"naive": [], "robust": []}

    for rnd in range(args.rounds):
        # Two independent federated trajectories. Each arm trains from its OWN
        # global state; client data and minibatch draws are matched across arms
        # (same child seed per round+client), and the same attack and
        # compression apply to both, so the comparison is controlled.
        arm_deltas = {"naive": [], "robust": []}
        for i, (X, y) in enumerate(clients):
            child = seed * 1_000_003 + rnd * 1_009 + i
            for arm, w in (("naive", w_naive), ("robust", w_robust)):
                d = local_train(w, X, y, args.inner_steps, 0.5,
                                np.random.default_rng(child))
                if i in byz:
                    d = -5.0 * d  # adversarial: amplified inverted update
                arm_deltas[arm].append(compress(d))
        dn = np.array(arm_deltas["naive"])
        dr = np.array(arm_deltas["robust"])

        w_naive += dn.mean(axis=0)
        w_robust += trimmed_mean(cosine_filter(dr))

        hist["naive"].append(accuracy(w_naive, Xt, yt))
        hist["robust"].append(accuracy(w_robust, Xt, yt))

    toy_bytes = dr[0].size  # toy delta: 3 float64 values (not serialized)
    # Honest projection for a real rank-16 LoRA adapter (formula, not a
    # measurement): 32 layers x 2 target matrices x (d*r + r*d), d=2048.
    lora_params = 32 * 2 * (2048 * 16 + 16 * 2048)
    kept = int(lora_params * 0.2)  # Top-K 20%
    proj_mb_values = kept * 1 / 1e6          # 8-bit values
    proj_mb_with_idx = kept * (1 + 3) / 1e6  # + 3-byte indices
    return hist, byz, (toy_bytes, proj_mb_values, proj_mb_with_idx)


def make_viz(hists, byz, args, seeds, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = np.arange(1, args.rounds + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, color, label in (("naive", "#c0392b", "naive mean (no defense)"),
                               ("robust", "#2980b9", "trimmed mean + cosine filter")):
        curves = np.array([h[name] for h in hists])  # seeds x rounds
        mean = curves.mean(axis=0)
        ax.plot(rounds, mean, color=color, lw=2, label=label)
        if len(seeds) > 1:
            ax.fill_between(rounds, curves.min(axis=0), curves.max(axis=0),
                            color=color, alpha=0.2)
    ax.set_xlabel("aggregation round")
    ax.set_ylabel("held-out accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    seed_note = f"seeds {seeds}" if len(seeds) > 1 else f"seed {seeds[0]}"
    band_note = "; band = min-max across seeds" if len(seeds) > 1 else ""
    ax.set_title(
        f"TOY SIMULATION: {args.clients} clients, {args.byzantine} adversarial, "
        f"{seed_note}{band_note}\n"
        f"Top-K(20%) + 8-bit deltas, H={args.inner_steps} local steps - "
        f"synthetic data, not measured performance",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clients", type=int, default=12)
    ap.add_argument("--byzantine", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--inner-steps", type=int, default=50)
    ap.add_argument("--seeds", default="7",
                    help="comma-separated fixed seeds (default '7'); use several "
                         "to see variance, e.g. --seeds 7,11,23")
    ap.add_argument("--rotation-range", type=float, default=0.6,
                    help="client data heterogeneity: max rotation of each client's "
                         "class boundary in radians (default 0.6)")
    ap.add_argument("--viz", metavar="PNG", help="write the accuracy chart to this path")
    args = ap.parse_args()

    try:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    except ValueError:
        ap.error("--seeds must be comma-separated integers")
    if not seeds:
        ap.error("--seeds must name at least one seed")
    if any(s < 0 for s in seeds):
        ap.error("--seeds must be non-negative integers")
    if args.clients < 2:
        ap.error("--clients must be >= 2")
    if not 0 <= args.byzantine < args.clients:
        ap.error("--byzantine must be >= 0 and < --clients")
    if args.byzantine >= args.clients / 2:
        print("warning: adversarial clients are not a minority; trimmed-mean "
              "filtering is not expected to help in that regime.",
              file=sys.stderr)
    if args.rounds < 1 or args.inner_steps < 1:
        ap.error("--rounds and --inner-steps must be >= 1")
    import math

    if not math.isfinite(args.rotation_range) or args.rotation_range < 0:
        ap.error("--rotation-range must be finite and >= 0")

    hists = []
    byz_seen = set()
    for seed in seeds:
        hist, byz, payload_info = run_one(args, seed)
        hists.append(hist)
        byz_seen |= byz
    toy_bytes, proj_mb_values, proj_mb_with_idx = payload_info

    print("TOY SIMULATION - synthetic 2D data, a tiny linear model whose delta")
    print("stands in for an adapter update. This does NOT implement LoRA or its")
    print("merging; it is not a measurement of the proposed system.\n")
    print(f"clients={args.clients}, adversarial per seed: {sorted(byz_seen)}, "
          f"rounds={args.rounds}, H={args.inner_steps}, "
          f"rotation_range={args.rotation_range}, seeds={seeds}")
    print(f"toy delta: {toy_bytes} float64 values (24 bytes in memory, not")
    print(f"serialized; 3 bytes if 8-bit quantized, before scale/index overhead).")
    print(f"The toy keeps k=1 of 3 coordinates (33%; Top-K 20% only applies at")
    print(f"realistic sizes). Formula-only projection for a rank-16 LoRA adapter")
    print(f"(32 layers x 2 targets, d=2048) with Top-K 20% + 8-bit:")
    print(f"~{proj_mb_values:.1f} MB of values, ~{proj_mb_with_idx:.1f} MB with "
          f"indices and per-tensor scale - the proposal's ~2-5 MB upload claim")
    print(f"is in reach on size, but its convergence impact is untested")
    print(f"(see feasibility-review.md).\n")
    finals = {}
    for name in ("naive", "robust"):
        label = "naive mean" if name == "naive" else "trimmed mean + cosine filter"
        ends = [h[name][-1] for h in hists]
        finals[name] = ends
        if len(seeds) > 1:
            print(f"  {label:32s} final accuracy: mean {np.mean(ends):.3f} "
                  f"[min {min(ends):.3f}, max {max(ends):.3f}] over {len(seeds)} seeds")
        else:
            print(f"  {label:32s} final accuracy: {ends[0]:.3f} "
                  f"(single seed {seeds[0]})")
    gap = np.mean(finals["robust"]) - np.mean(finals["naive"])
    if args.byzantine:
        config = "one attack type: inverted amplified updates"
    else:
        config = "no adversarial clients (baseline)"
    print(f"\nIn this configuration ({config}; "
          f"{len(seeds)} seed(s)), trimmed mean + cosine filter ends "
          f"{gap:+.3f} vs naive averaging.")
    attack_word = "one attack type on" if args.byzantine else "a no-attack baseline on"
    print(f"This is {attack_word} synthetic data - not evidence of general")
    print("robustness. The review's open questions about merging real LoRA factors")
    print("remain open research, and extreme data heterogeneity degrades even")
    print("filtered aggregation (try --rotation-range 3.0).")
    if args.viz:
        print(f"\nChart written to {make_viz(hists, byz_seen, args, seeds, args.viz)}")


if __name__ == "__main__":
    main()
