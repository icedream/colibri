# colibrì on AMD GPUs — how the engine runs, and how the HIP/ROCm port works

This fork adds AMD GPU support to colibrì (`make HIP=1`) and a new VRAM tier mode
(`CUDA_EXTEND=1`). Upstreamed as [JustVugg/colibri#112](https://github.com/JustVugg/colibri/pull/112);
benchmark record in [issue #93](https://github.com/JustVugg/colibri/issues/93).
This document explains the whole pipeline first, because the GPU work only makes
sense against the engine's actual bottlenecks.

## 1. The core idea: most of a giant model is asleep at any moment

GLM-5.2 is a **Mixture-of-Experts (MoE)** model with 744B parameters. An MoE
model never uses all of itself at once:

- **The dense part** — attention, embeddings, shared expert, first 3 dense
  layers — is ~17B parameters, needed for *every* token.
- **The routed experts** — 21,504 of them (75 MoE layers × 256 experts, plus
  the MTP head) — are small independent FFNs, ~19 MB each at int4. Per token,
  each layer's *router* picks just **8 of its 256**.

So per token only ~40B parameters activate, and only ~11 GB of weights *change*
token to token. colibrì's architecture follows directly:

> Keep the dense ~9.9 GB **resident in RAM** (int4). Leave the ~370 GB of
> experts **on disk**, streaming in only the 8-per-layer each token routes to.

That makes the engine disk-bound: a cold token costs ~600 expert reads ≈ 11 GB
of disk traffic. Every cache in colibrì exists to avoid re-reading experts.

## 2. The moving parts

The runtime is one C file (`c/glm.c`) plus small headers — no BLAS, no Python
at runtime, no GPU required:

- `st.h` — safetensors reader (`pread`, no 370 GB mmap)
- `tok.h` / `tok_unicode.h` — byte-level BPE tokenizer in C (320k merges)
- `json.h` — minimal JSON parser · `tier.h` — tier heat logic
- `compat.h` — all Windows shims · `backend_gpu_compat.h` — all GPU-vendor shims (this port)
- `backend_cuda.cu/.h` — the opt-in GPU expert backend (10-function opaque C ABI)
- `coli` — Python CLI wrapper (spawns `glm`, sentinel-framed stdio, env-var config)
- `openai_server.py` — stdlib-only OpenAI-compatible HTTP gateway

## 3. The memory hierarchy

When an expert is needed, tiers are searched fastest-first:

1. **The pin (hot-store, the learning cache)** — experts permanently in RAM.
   Every turn the engine appends routing counts to `<model>/.coli_usage`; at
   startup it ranks experts by that history and pins the hottest into spare
   RAM (budget scales with history confidence, up to half the expert budget),
   then `mlock`s them.
2. **The per-layer LRU cache** — recently streamed experts, auto-sized from
   the `--ram` budget via an honest peak-RSS projection (cap auto-raise).
3. **The OS page cache** — free L2 for recently read disk blocks.
4. **Disk** — one coalesced ~19 MB `pread` per expert, with `WILLNEED`
   readahead of the next expert block while the current one multiplies.

With this port, tier **0** can be **VRAM** (see §7). A crucial detail: an
expert slot (`ESlot`) is zero-copy — its three matrices (gate/up/down) are
*views into one slab* filled by a single read. That constraint shaped the
VRAM-only slot design.

## 4. The life of a token

1. **Tokenize** (C BPE) and wrap in GLM's chat template.
2. **Prefill** the prompt as one batch. **Batch-union MoE**: each unique
   expert needed by any position is read once and applied to every position
   routing to it — long prompts are cheap; output tokens are what's metered.
3. **Decode**, per layer: **MLA attention** with a compressed KV cache (576
   floats/token instead of 32,768 — 57× smaller; the absorption trick avoids
   reconstruction entirely at decode) → **DSA sparse attention** past 2,048
   context tokens (top-2048 key selection) → **sigmoid router** picks top-8
   experts (`--topp` trims adaptively, 30–40% less disk) → tier lookup above.
4. **MTP speculative decoding**: GLM-5.2's own draft head proposes 3 tokens;
   the main model verifies them in one batched forward. The head must be
   **int8** (int4 collapses acceptance to 0–4%; int8 measures ~40% = 2.2–2.8
   tokens/forward). Lossless, also under sampling (rejection sampling).
5. **Sample**, emit, append compressed KV to `.coli_kv` (conversations reopen
   warm, zero re-prefill), update `.coli_usage`, print the stats line.

## 5. The AMD port: one source file, two vendors

The CUDA backend is portable-by-design: an opaque C ABI, kernels using only
`__global__`/`__shared__`/`__syncthreads__` — no warp intrinsics, no cuBLAS.
Instead of forking a `.hip` copy that would drift, this port applies the
repo's own Windows pattern (`compat.h`) to GPU vendors:

- **`backend_gpu_compat.h`** (~29 lines): compiled by nvcc it passes through
  to `cuda_runtime.h`; compiled by hipcc it maps the exact 14-symbol CUDA
  runtime surface the backend uses onto HIP 1:1.
- **Makefile**: a `HIP=1` block mirroring `CUDA=1` (hipcc, `--offload-arch`,
  `-lamdhip64`), a shared `GPUCC` so build rules stay single-path, and
  `make hip-test` running the existing kernel-correctness test on ROCm.

`backend_cuda.cu` compiles **unchanged** under both vendors; upstream CUDA
fixes flow to AMD automatically.

```sh
cd c
make hip-test HIP_ARCH=gfx1201     # kernel correctness on the local AMD GPU
make glm HIP=1 HIP_ARCH=gfx1201    # full engine with the ROCm backend
HIP_VISIBLE_DEVICES=0 COLI_CUDA=1 COLI_GPU=0 CUDA_EXPERT_GB=12 CUDA_EXTEND=1 \
  ./coli chat --ram 40 --topp 0.7
```

(`HIP_VISIBLE_DEVICES=0` masks unsupported iGPUs from ROCm's enumeration.)

## 6. Why the stock VRAM tier didn't help a disk-bound machine

Upstream's VRAM tier is a **mirror**: it promotes the hottest experts *already
in the RAM pin* to the GPU, keeping the RAM copy — a compute accelerator,
ideal for matmul-bound machines (fast-disk boxes measure ~57% matmul). On a
disk-bound machine it adds zero cache coverage: measured perf-neutral here
(0.22 vs 0.23 tok/s), confirming the README's own design note.

## 7. `CUDA_EXTEND=1`: VRAM as additional pin capacity

With `CUDA_EXTEND=1`, after the RAM pin fills its budget, `pin_load` keeps
walking down the frequency ranking: each next expert is loaded from disk once,
uploaded to VRAM, and its **host slab freed**. These `vram_only` slots cost
VRAM, not RAM — the LRU keeps its full budget. Legacy mirror stays the default.

Safety engineering (a GPU-only slot has no CPU fallback):

- Cached device tensors stay callable after host pointers die (backend upload
  check-order fix, pinned by a test assertion).
- If the GPU refuses a VRAM-only expert mid-run: the matmul flags the tensor
  and emits zeros, the expert loop repairs in place (one disk reload + CPU
  recompute — not even that token is wrong), and the pin lookup skips the
  dead slot afterwards.
- REPIN will not swap into slab-less slots.

## 8. Measured results (honest numbers)

Hardware: RX 9070 XT 16 GB (gfx1201, ROCm 7.2.4) on a Ryzen 7 5700G — an APU
that is **PCIe Gen3 only**, so the GPU links at Gen3 x16 (13.4 GB/s measured)
and the Gen5 NVMe at Gen3 x4 (3.5 GB/s). GLM-5.2 int4 + int8 MTP head,
`--ram 40 --topp 0.7 --temp 0 --ngen 32`, greedy fixed prompt, 10 runs/config:

| config | median tok/s | expert hit | MTP acceptance |
|---|---|---|---|
| CPU only | 0.22 | 31–38% | 40% steady |
| HIP mirror (`CUDA_EXPERT_GB=12`) | 0.22 | 33% | 33% |
| **HIP `CUDA_EXTEND=1`** | **0.24** (best 0.30) | **44–49%** | 23–46%, mean 31% |

The extension structurally lifts hit-rate ~10 points (pinned coverage 620 →
1,254 experts, RSS flat). The remaining friction: GPU float matmuls round
differently than the CPU int8-dot (IDOT) kernels, nudging the greedy
trajectory and dropping MTP acceptance to ~31% mean — extra verification
forwards absorb most of the caching win. When acceptance held (run 9, 46%),
throughput hit **0.30 tok/s (+36% vs CPU median)**.

## 9. Known follow-ups

- **Numerics-matched GPU kernel** — int8 activation quantization + integer
  dot on the GPU, mirroring the CPU IDOT path, to stop the MTP acceptance
  drop and make ~0.30 the steady state on this class of machine.
- Parallelize the extension's startup load/upload loop (currently serial,
  ~15 s for 634 experts).
- `coli plan` / `resource_plan.py` do not yet model the extension tier.
- A Vulkan backend remains the vendor-neutral endgame for machines without
  ROCm; the HIP port keeps all kernels single-source until then.
