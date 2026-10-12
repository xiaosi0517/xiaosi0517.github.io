---
layout: project
title: Efficient LLM Inference & Scheduling
description: Optimizing LLM inference through prefill/decode scheduling, W8A16 quantization, and GPU memory efficiency.
hero_subtitle: Exploring scheduling and memory optimization in a lightweight LLM inference engine.
permalink: /projects/efficient-llm-inference-scheduling/
category: research
project_category: AI Infrastructure
importance: 5
featured: false
img: assets/img/projects/llm-inference-scheduling-thumbnail.png
img_alt: Technical illustration of LLM inference scheduling, W8A16 model quantization, and GPU memory management.
tech:
  - PyTorch
  - GPU Computing
  - Inference Scheduling
  - W8A16 Quantization
  - Performance Optimization
---

<p class="project-disclaimer">This page describes targeted experiments in a lightweight LLM inference engine. The scheduling and quantization changes are presented as engineering modifications and benchmarked prototypes, not as a claim that the full nano-vLLM engine was built from scratch or that the optimizations are production-ready across all workloads.</p>

<div class="project-hero-visual">
{% include figure.liquid loading="eager" path="assets/img/projects/llm-inference-scheduling-thumbnail.png" class="img-fluid project-diagram" alt="Technical illustration showing multiple requests, blue prefill blocks, purple decode blocks, W8A16 quantization, and GPU memory allocation." %}
</div>

The engineering question behind this project was simple: when a model is already small enough to run on one GPU, where does inference efficiency actually come from? The answer is not only model math. Request scheduling, KV-cache capacity, weight storage, CUDA graph behavior, and workload shape all decide how the system behaves.

I explored two complementary optimization directions:

- **Prefill/decode interleaving.** Modify the scheduler so long prompt prefill work and active decode requests can both make progress under mixed traffic.
- **W8A16 weight-only quantization.** Store decoder linear weights as INT8 with per-output-channel scales while keeping activations in FP16/BF16.

The two experiments target different bottlenecks. Scheduling affects request latency and execution order. Quantization affects persistent model storage and how much GPU memory remains for KV cache and runtime state.

<div class="project-links project-links-top">
  <a href="https://github.com/xiaosi0517/nano-vllm" target="_blank" rel="noopener">Primary repository</a>
  <a href="https://github.com/xiaosi0517/llm-inference-feasibility" target="_blank" rel="noopener">Experiment repository</a>
</div>

## The Challenge: Inference Is a Systems Problem

LLM inference alternates between two phases with different resource profiles:

<div class="llm-phase-grid">
  <section class="llm-phase-card prefill">
    <span class="phase-chip">Prefill</span>
    <h3>Prompt processing</h3>
    <p>Consumes input prompt tokens and builds KV-cache state. It is often compute-heavy and benefits from parallel work across many prompt tokens.</p>
  </section>
  <section class="llm-phase-card decode">
    <span class="phase-chip">Decode</span>
    <h3>Autoregressive generation</h3>
    <p>Generates one token per active request step. It repeatedly reads model weights and KV-cache state, so memory bandwidth and batching matter.</p>
  </section>
</div>

A scheduler has to decide which work enters the next model forward. A long prompt can consume many chunked prefill steps. Active generations need repeated decode steps to keep producing tokens. GPU memory is split among model weights, KV cache, activations, temporary tensors, and runtime overhead. Optimizing one part can expose a different bottleneck somewhere else.

## Experiment 01: Prefill/Decode Interleaving

### Problem

Chunked prefill makes long prompts possible under a limited `max_num_batched_tokens` budget, but it creates a scheduling question. If the scheduler prefers waiting prompt work whenever it exists, a long prompt may run several prefill chunks before active decoders continue. If the scheduler always prioritizes decode, new requests may wait for prompt processing.

The trade-off is between time to first token, decode progress, overall completion time, and fairness across request types.

### Implementation

The scheduler change adds a configuration switch, `prefill_decode_interleave`, and tracks whether the previous engine step was prefill. When interleaving is enabled, the scheduler chooses decode first only when all of these are true:

- the previous scheduled step was prefill
- there is waiting prompt work
- there are active running decode requests
- interleaving is enabled

Otherwise, the scheduler follows the normal prefill-first path, then falls back to decode when no prompt work can be scheduled. This is scheduler-level interleaving, not simultaneous prefill and decode execution in one model forward.

Chunked prefill reuses existing cached-prefix mechanics. The scheduler sets `seq.num_scheduled_tokens` for the next prompt chunk. The model runner starts from `seq.num_cached_tokens`, writes new KV slots, and lets later chunks attend to cached prefix blocks through the block table.

<div class="project-figure">
<div class="project-figure-head"><span class="project-figure-title">Figure 1 · Scheduler-level execution order</span><span class="figure-badge">Conceptual trace</span></div>
<div class="schedule-compare" role="img" aria-label="Conceptual comparison of baseline scheduling and interleaved scheduling. Baseline runs several prefill chunks before decode resumes. Interleaving alternates prefill and decode while both queues have work.">
  <div class="schedule-row">
    <h3>Baseline</h3>
    <div class="schedule-track">
      <span class="block prefill wide">P</span>
      <span class="block decode">D</span>
      <span class="block prefill">P</span>
      <span class="block prefill">P</span>
      <span class="block prefill">P</span>
      <span class="block prefill">P</span>
      <span class="block decode">D</span>
      <span class="block decode">D</span>
    </div>
  </div>
  <div class="schedule-row">
    <h3>Interleaved</h3>
    <div class="schedule-track">
      <span class="block prefill wide">P</span>
      <span class="block decode">D</span>
      <span class="block prefill">P</span>
      <span class="block decode">D</span>
      <span class="block prefill">P</span>
      <span class="block decode">D</span>
      <span class="block prefill">P</span>
      <span class="block decode">D</span>
    </div>
  </div>
</div>
<p class="project-figure-caption">Each block represents one engine step. Requests may be batched within a step, but prefill and decode are not mixed in the same forward pass in this implementation.</p>
</div>

### Measured Results

The recorded scheduling experiment used Qwen3-0.6B on an RTX 4090 with eager execution, `max_num_batched_tokens=128`, `max_num_seqs=8`, four initial short requests, one long prompt of about 1024 tokens, four late short requests, and stochastic sampling with `temperature=0.6`.

<div class="metric-comparison">
  <section>
    <span>Total runtime</span>
    <strong>44.95s → 42.57s</strong>
    <p>5.3% lower benchmark wall-clock completion time.</p>
  </section>
  <section>
    <span>Average TTFT</span>
    <strong>11.70s → 10.76s</strong>
    <p>About 8.0% lower average time to first token.</p>
  </section>
  <section>
    <span>Tail TTFT</span>
    <strong>25.67s → 23.60s</strong>
    <p>Improved p90 and p99 TTFT in this nine-request run.</p>
  </section>
</div>

Other observations were important too: preemptions stayed at zero, peak KV-cache block usage stayed at 13 blocks, and decode throughput stayed similar. The clearest qualitative result was the execution trace: with interleaving disabled, a long prompt produced consecutive prefill chunks; with interleaving enabled, active decoders resumed every other step while both queues had work.

### Engineering Insight

The result supports the scheduler hypothesis for this workload, but it does not prove a universal win. The absolute TTFT numbers include cold-start overhead, one late request waits because the test has nine requests and `max_num_seqs=8`, and the benchmark was a single controlled workload on one RTX 4090. The useful engineering result is narrower and stronger: a small scheduler policy change produced the intended execution pattern and improved measured latency metrics under the documented mixed workload.

## Experiment 02: W8A16 Weight-Only Quantization

### Problem

GPU memory is divided between persistent model state and runtime state. Model weights occupy memory before any request arrives. KV cache grows with context length, batch size, and active sequence count. Reducing model weight storage can make more memory available for serving, but it does not automatically make matrix multiplication faster.

### Implementation

The W8A16 path quantizes linear weights at load time. For each output channel, the implementation computes a scale from the maximum absolute weight value, stores the quantized weights as INT8, and stores the scale values as FP32 buffers. During forward execution, weights are dequantized back to the activation dtype and used in the existing linear path.

<div class="project-figure">
<div class="project-figure-head"><span class="project-figure-title">Figure 2 · Weight storage versus compute path</span><span class="figure-badge">Implementation summary</span></div>
<div class="quant-flow" role="img" aria-label="W8A16 quantization stores INT8 weights and FP32 scales, then dequantizes during forward execution before higher precision matrix multiplication.">
  <div class="quant-box source">
    <strong>FP16/BF16 weights</strong>
    <span>Original floating-point storage</span>
  </div>
  <div class="quant-arrow">→</div>
  <div class="quant-box compressed">
    <strong>INT8 weights + FP32 scales</strong>
    <span>Per-output-channel scaling</span>
  </div>
  <div class="quant-arrow">→</div>
  <div class="quant-box compute">
    <strong>Runtime dequantization</strong>
    <span>Higher-precision linear operation</span>
  </div>
</div>
<p class="project-figure-caption">This is memory-efficient weight storage, not an optimized INT8 GEMM implementation. Temporary dequantized tensors can offset peak-memory and latency gains.</p>
</div>

### Measured Results

The quantization benchmark measured persistent model storage and request performance across eager and CUDA graph modes. The most stable result was the storage reduction:

<div class="metric-comparison quant">
  <section>
    <span>Original model storage</span>
    <strong>1.42 GiB</strong>
    <p>Recorded persistent model parameter and buffer storage.</p>
  </section>
  <section>
    <span>W8A16 model storage</span>
    <strong>1.01 GiB</strong>
    <p>About 29% lower persistent model storage.</p>
  </section>
  <section>
    <span>KV-cache capacity</span>
    <strong>Higher in W8A16 runs</strong>
    <p>For example, CUDA graph interleaved mode increased total KV blocks from 671 to 700.</p>
  </section>
</div>

The speed story was more mixed. Some W8A16 runs were faster than their baseline counterpart, but the best interleaved CUDA graph result remained the non-quantized baseline. That is consistent with the implementation: the weights are stored compactly, but each forward pass still pays for dequantization and then runs a higher-precision linear operation.

## GPU Memory Architecture

<div class="project-figure">
<div class="project-figure-head"><span class="project-figure-title">Figure 3 · What competes for GPU memory</span><span class="figure-badge">Conceptual allocation</span></div>
<div class="gpu-memory-diagram" role="img" aria-label="Conceptual GPU memory allocation split among model weights, KV cache, activations, temporary buffers, and runtime overhead.">
  <div class="memory-stack">
    <span class="weights">Model weights</span>
    <span class="kv">KV cache</span>
    <span class="activations">Activations</span>
    <span class="temp">Temporary buffers</span>
    <span class="runtime">Runtime overhead</span>
  </div>
  <div class="memory-grid" aria-hidden="true">
    {% for i in (1..64) %}<span></span>{% endfor %}
  </div>
</div>
<p class="project-figure-caption">Quantization can reduce persistent weight storage and leave room for more KV-cache blocks. The scheduling improvement above should not be attributed to quantization; the two experiments probe different bottlenecks.</p>
</div>

The memory budget is shaped by model size, precision, context length, active sequence count, batch size, and temporary allocations created by the execution backend. W8A16 reduced persistent model storage and increased available KV-cache blocks in the recorded runs, but peak CUDA allocation did not clearly drop because runtime dequantization creates temporary tensors.

## Experimental Methodology

The scheduling benchmark records per-step traces, per-request latencies, aggregate summaries, preemption counts, KV-cache block usage, and CUDA memory statistics. The workload shape was chosen to force a long prompt to be chunked while short requests were already decoding.

The quantization benchmark measures model storage by summing parameter and buffer bytes, separates INT8 weights and scale buffers, records KV-cache capacity, and compares request latency across eager and CUDA graph execution. Both experiments used synthetic prompts and controlled request mixes rather than production traffic.

<div class="project-tradeoffs">
  <section>
    <h3>Scheduling fairness vs. throughput</h3>
    <p>Interleaving gives decode requests regular opportunities to progress when a long prompt is chunked. That can improve latency distribution, but a different workload may prefer larger uninterrupted prefill batches.</p>
  </section>
  <section>
    <h3>Chunk size vs. overhead</h3>
    <p>Smaller chunks improve scheduling flexibility. They can also increase scheduler and kernel overhead, so the chunk budget is a serving-policy decision rather than a universal constant.</p>
  </section>
  <section>
    <h3>Memory savings vs. runtime speed</h3>
    <p>W8A16 reduced persistent weight storage. It did not establish production-speed quantized inference because the implementation dequantizes weights during forward execution.</p>
  </section>
  <section>
    <h3>Concurrency vs. KV-cache capacity</h3>
    <p>More active requests can improve utilization, but each sequence consumes KV-cache space. Freeing memory for more KV blocks helps capacity, while scheduling still decides which requests progress.</p>
  </section>
</div>

## Explore the Implementation

The implementation work is organized around a few focused code paths:

- Scheduler policy and chunked prefill behavior in <a href="https://github.com/xiaosi0517/nano-vllm/blob/main/nanovllm/engine/scheduler.py" target="_blank" rel="noopener"><code>nanovllm/engine/scheduler.py</code></a>
- Cached-prefix preparation in <a href="https://github.com/xiaosi0517/nano-vllm/blob/main/nanovllm/engine/model_runner.py" target="_blank" rel="noopener"><code>nanovllm/engine/model_runner.py</code></a>
- Configuration switches in <a href="https://github.com/xiaosi0517/nano-vllm/blob/main/nanovllm/config.py" target="_blank" rel="noopener"><code>nanovllm/config.py</code></a>
- W8A16 linear layers in <a href="https://github.com/xiaosi0517/nano-vllm/blob/main/nanovllm/layers/linear.py" target="_blank" rel="noopener"><code>nanovllm/layers/linear.py</code></a>
- Benchmark scripts for latency and quantization profiling in the experiment repository

Future work would move the quantization path from compact storage toward compute-efficient inference: optimized INT8 GEMM kernels, fused dequantization, smaller temporary allocations, kernel-level profiling, and repeated measurements across larger models and heavier workloads.

<div class="project-links">
  <a href="{{ '/portfolio/' | relative_url }}">Back to portfolio</a>
  <a href="{{ '/projects/scalable-synthetic-workload-generation/' | relative_url }}">Previous project · Scalable synthetic workload generation</a>
</div>
