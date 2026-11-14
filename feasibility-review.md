# Feasibility Review: Daytime Companion + Nighttime Hive-Mind Adapter

**Status: independent review, November 15, 2026.** This review assesses the Gemini
proposal in `proposal-gemini.md`. It is not a build plan, benchmark, or guarantee of
performance. Nothing here has been implemented or measured on real hardware.

## What is established

- **DiLoCo exists, and the proposal borrows its core structure.** Google DeepMind's
  DiLoCo (November 2023) is a federated-averaging variant with many local AdamW
  "inner" steps and a Nesterov-momentum "outer" optimizer, communicating roughly
  500x less than synchronous training, with published robustness to island drop-out
  and heterogeneous data across workers. The proposal reuses that inner/outer
  shape. However, what it describes on top - LoRA-only updates, Top-K
  sparsification, 8-bit quantization, asynchronous Byzantine-robust aggregation -
  is not the published DiLoCo algorithm, and DiLoCo's results come from full-model
  training on large islands. The convergence of this combined variant is untested.
  Sources: https://deepmind.google/research/publications/57039/ ,
  https://arxiv.org/abs/2311.08105
- **The named components are real projects.** llama.cpp (GGUF Q4_K_M quantization),
  Unsloth, Hugging Face PEFT, bitsandbytes (4-bit QLoRA), Tauri v2, FastAPI, and
  safetensors all exist and are publicly documented. Of these, only Unsloth's
  current hardware support was checked directly against its documentation, which
  confirms 4-bit fine-tuning on consumer NVIDIA GPUs; the current state of the
  others was not individually verified for this review.
  Source: https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements
- **The day/night GPU-state-machine concept is sound.** Unloading inference before
  loading a training graph is the right way to avoid VRAM oversubscription on a
  single consumer GPU; this is a well-known constraint, not a novel risk.

## What is plausible but unverified

- **VRAM budgets.** "~2.5-3.5 GB for a 3B-4B Q4_K_M inference model" and
  "~5-6 GB peak for QLoRA r=16 with gradient checkpointing and batch size 1" are in
  the commonly reported range for 3B-4B models, but the real numbers depend on
  context length, KV-cache settings, LoRA target modules, and the specific GPU and
  driver. Treat them as planning estimates to measure on the target machine, not
  budgets to design around.
- **Upload payload of ~2-5 MB.** Top-K sparsification plus 8-bit quantization of a
  rank-16 adapter delta can reach that size, but compression at that level interacts
  with outer-optimizer convergence; DiLoCo itself does not use sparsified deltas.
  This combination needs its own experiment.
- **Tauri shell at ~40 MB RAM.** In the right ballpark for a minimal Tauri app, but
  a Three.js or Live2D animated overlay can grow well beyond it.

## What is optimistic or needs testing first

- **Sub-500 ms checkpoint write, and the unmeasured rest.** The proposal's literal
  target is narrow: "saves the current step checkpoint to disk in < 500ms". That is
  more tractable than it first sounds, because a LoRA checkpoint holds the adapter
  weights plus optimizer state for the trainable parameters only, and its size
  depends on the trainable parameter count, dtypes, and optimizer choice - far less
  than full-model state. Still, even the write alone is unmeasured: disk flush
  behavior and contention decide whether 500 ms is realistic, so treat it as a
  design target. Separately, the full preemption-and-resume sequence - the trainer
  reaching a safe interrupt point, VRAM release, llama.cpp relaunch, and model
  reload - has its own latency that the proposal does not quantify at all. That
  end-to-end resume time, not the checkpoint write, is what the user will actually
  feel when they touch the mouse, and it is entirely unmeasured.
- **"Instant" LoRA hot-reload in llama.cpp.** llama.cpp supports applying LoRA
  adapters, but swapping an adapter on a live server without a reload penalty is not
  a documented guarantee; verify against the current llama.cpp server behavior.
- **Privacy claims.** Two separate issues. First, "anonymized" preference pairs
  stored locally is a design choice, not a property of the stack; what gets logged
  needs its own review. Second, adding noise to uploaded updates is not by itself a
  differential-privacy guarantee. A quantified DP guarantee requires clipping of
  each client's update, noise calibrated to that clipping, and epsilon/delta
  accounting that composes over training rounds; the proposal specifies none of
  these, so as written it offers no quantified privacy guarantee. Meanwhile it is
  established that weight updates can leak information about training data, so
  "individual chat phrases cannot be reverse-engineered from the weights" is
  stronger than the cited techniques support.
- **Byzantine-robust aggregation.** Cosine-similarity filtering plus trimmed mean
  is a reasonable starting point, but robust aggregation against adaptive poisoning
  of LoRA updates is an open research area; treat it as a mitigation, not a solved
  problem.
- **Idle detection coverage.** The proposal names Windows and macOS idle-time APIs
  only. Linux (X11/Wayland) needs its own path, and the "<15% GPU for 10 minutes"
  rule can misfire during video playback, streaming, or background renders.

## The core research risk: merging LoRA factors across clients

The proposal's hardest problem is one it does not mention. LoRA's A/B factorization
is not unique: the same adapter can be written as different (A, B) pairs related by
scaling and rotation, so coordinate-wise averaging or Top-K sparsification of
deltas from different clients is not guaranteed to correspond to a meaningful merge
of what they learned. On top of that, clients must agree on the base model,
tokenizer, target modules, rank, and adapter format version before their updates
are comparable at all, and asynchronous updates add staleness that trimmed means do
not address. This is an active research area, not an engineering detail: FLoRA
(NeurIPS 2024) exists precisely because naive federated aggregation of LoRA factors
degrades, and a run of 2025-2026 papers on basis-consistent and geometry-aware
federated LoRA aggregation shows the problem is still unsettled. Treat the
aggregation design as unresolved research, and expect the first version to need
experiments rather than a library call.
Sources: https://arxiv.org/abs/2409.05976 (FLoRA, NeurIPS 2024);
https://arxiv.org/html/2611.07598 and https://arxiv.org/html/2605.06733 (recent
examples of ongoing work).

## Open questions the proposal does not answer

- **Coordinator operations.** Who runs and pays for the FastAPI/Redis/S3
  coordinator, how companions authenticate, and what happens when the coordinator is
  down. This is an ongoing service, not a one-time build.
- **Catastrophic forgetting and eval drift.** The 50/50 local/shared-curriculum mix
  and the held-out benchmark suite are asserted, not specified; the benchmark suite
  is itself a significant piece of work.
- **Model and adapter licensing.** Qwen2.5-3B and Gemma-3-4B have their own
  licenses; redistributing merged or hive-trained adapters needs a license check.
- **Minimum hive size.** How a rank-16 adapter aggregated from a handful of
  heterogeneous consumer PCs behaves is unknown, and feeds directly into the
  aggregation risk above.

## Bottom line

The architecture is coherent and every named component exists, but the proposal's
numbers are design targets, not measurements, and its central mechanism - merging
sparsified LoRA deltas from many heterogeneous, intermittently online clients - is
an open research problem rather than a stack choice. A sensible first milestone is
a single-machine prototype that proves the two most consequential local claims:
preemption latency (checkpoint and inference-restore) and peak QLoRA VRAM on the
actual GPU. The federated loop should only be built after those hold, and its
aggregation scheme will need experiments of its own.

## Sources

- DiLoCo paper: https://deepmind.google/research/publications/57039/ and
  https://arxiv.org/abs/2311.08105
- FLoRA (federated fine-tuning with heterogeneous LoRA, NeurIPS 2024):
  https://arxiv.org/abs/2409.05976
- Unsloth requirements and supported hardware: https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements
- LoRA/QLoRA VRAM sizing background (third-party calculator, indicative only):
  https://vramcalculator.io/guides/lora-qlora-vram-minimums/
- llama.cpp: https://github.com/ggml-org/llama.cpp
- Hugging Face PEFT: https://github.com/huggingface/peft
- bitsandbytes: https://github.com/bitsandbytes-foundation/bitsandbytes
- Tauri v2: https://v2.tauri.app/
