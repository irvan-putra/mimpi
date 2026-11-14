# Testing and dependencies

Python 3.9+. Simulation: numpy. Chart: matplotlib.

Commands run and verified (2026-11-15, Python 3.10):

    python3 hive_sim.py --seeds 7,11,23                  # default attack config
    python3 hive_sim.py --byzantine 0 --seeds 7,11,23    # zero-attack baseline
    python3 hive_sim.py --rotation-range 3.0 --byzantine 0 --seeds 7,11,23
    python3 hive_sim.py --rounds 3 --clients 5 --byzantine 1
    python3 hive_sim.py --seeds 7,11,23 --viz hive_sim.png
    python3 hive_sim.py --clients 1         # rejected: --clients must be >= 2
    python3 hive_sim.py --byzantine 12      # rejected: must be < --clients
    python3 hive_sim.py --rounds 0          # rejected: must be >= 1
    python3 hive_sim.py --seeds a,b         # rejected: must be integers

Observed (3 fixed seeds 7/11/23, 12 clients, 2 adversarial; two independent
trajectories with matched client data and minibatch draws per round/client,
same attack and compression on both arms): naive mean ends at 0.21 mean
held-out accuracy, trimmed mean + cosine filter at 0.97. Zero-attack baseline:
0.98 vs 0.98 - the defense neither helps nor hurts without an attack. Extreme
heterogeneity (rotation 3.0, no attack): both strategies degrade together to
~0.73 - the filter no longer helps; that is the honest non-IID failure case.
Aggregation is a genuine coordinate-wise trimmed mean (tails discarded, not
clipped) after a cosine filter. The toy delta is 3 float64 values (24 bytes,
not serialized; Top-K keeps 1 of 3 = 33%). All results are synthetic toy runs,
one attack type; not evidence of general robustness. LoRA is not implemented;
a tiny linear model's delta stands in for an adapter update.
Negative seeds and non-finite/negative --rotation-range are rejected cleanly.

The Pages site (docs/) was rendered headless and inspected at 1280px and
390px widths.
