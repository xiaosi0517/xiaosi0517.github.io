---
layout: page
title: Scalability in ML Systems - Intuitive Guide
description: Understanding scalability through concrete examples and practical techniques
permalink: /scalability-guide/
---

# Scalability: From Theory to Practice

## Part 1: The Basics - What Happens When Input Doubles?

Imagine you have a search system. Let's see what happens when you double the input size:

| Algorithm | 1M items | 2M items | Growth |
|-----------|----------|----------|---------|
| **O(N) - Linear Search** | 1 second | 2 seconds | 2x cost |
| **O(N²) - Nested Loop** | 1 million sec (11 days!) | 4 million sec (46 days!) | 4x cost |
| **O(log N) - Binary Search** | 20 checks | 21 checks | Almost no change |

**Key insight:** With O(N²), doubling input makes your system 4x slower. With O(N), it's just 2x slower. With O(log N), barely slower.

**But here's the critical question:** Can you fix a slow O(N²) system by adding more machines?
- **Short answer: Not really.**
- If processing 1M items takes 11 days with one machine, adding 10 machines brings it down to ~27 hours. Still terrible.
- With O(N), adding 10 machines: 1 second → 0.1 second. Much better.

**Why?** Because parallelization (adding machines) only helps with work you can split up. If your algorithm has fundamental inefficiency, machines can't save you.

---

## Part 2: Real Example - Finding Similar Images Across 1 Million Photos

### The Brute-Force Disaster

Your product: "Find all images similar to this photo of a cat."

**Brute-force approach:**
```
For each of 1M photos:
  - Load image
  - Compute 2048-dim embedding (CNN)
  - Compare with query embedding (compute distance)
  - Check if distance < threshold
```

**The numbers:**
- Embedding computation: ~50ms per image
- 1M images × 50ms = **50,000 seconds = 14 hours**
- User clicks "find similar" and waits 14 hours? ❌

### Why Adding Machines Doesn't Help Much

Even if you have 100 machines (unrealistic), each processes ~10K images:
- 10K × 50ms = 500 seconds = **8.3 minutes**
- Still way too slow. And you've bought 100 machines.

**The real problem:** You're doing expensive work (CNN embedding) unnecessarily.

---

## Part 3: Scaling Techniques (With Concrete Numbers)

### Technique 1: **Candidate Filtering** (Reduce Search Space)

**Problem it solves:** You don't need to check all 1M images; just the promising ones.

**How it works:**
1. **Quick indexing stage** (cheap): Hash images by color histogram, brightness, edges
2. **Filter** (eliminates 95% of images instantly)
   - Query: Cat photo (orange/brown colors, 800×600, cat-like edges)
   - Look up images with similar color/size/edges → ~10K candidates
3. **Expensive stage** (on small set): Compute CNN embeddings only for 10K candidates

**The numbers:**
- Original: 1M × 50ms = 14 hours
- With filtering: 10K × 50ms = 500 seconds = **~8 minutes** ✓
- **35x speedup without buying new machines**

**What happens without it:**
- Every request takes 14 hours. Your service is dead.

---

### Technique 2: **Grid / Spatial Hashing** (Smarter Indexing)

**Problem it solves:** When your data lives in a geometric space (images, locations, embeddings).

**How it works:**
- Divide your 1M images into regions (grid buckets)
- When searching, only check images in nearby buckets

**Concrete example: Semantic image search**
- You have embeddings in a 2048-dimensional space
- Divide into 1000 grid buckets (hashing)
- Query lands in one bucket with ~1K neighbors
- Check only those neighbors for similarity

**The numbers:**
- Full brute-force: 1M comparisons = seconds
- With grid hashing: 1K comparisons = milliseconds
- **1000x speedup**

**What happens without it:**
- Every similarity query is milliseconds × 1M = slow.

---

### Technique 3: **Parallelization** (Splitting Work Across Machines)

**Problem it solves:** You have embarrassingly parallel work (each image is independent).

**How it works:**
```
Task: Process 1M images, compute embeddings

Machine 1 → Images 0-100K → 5000 seconds
Machine 2 → Images 100K-200K → 5000 seconds
...
Machine 10 → Images 900K-1M → 5000 seconds

Total time: 5000 sec = 83 minutes (instead of 14 hours with 1 machine)
```

**The numbers:**
- 1 machine: 50,000 seconds
- 10 machines: 5,000 seconds = **10x speedup**
- 100 machines: 500 seconds = **100x speedup**

**But wait—what does this NOT solve?**
- If one image takes 50ms, 10 machines won't make it 5ms
- Parallelization helps with volume, not fundamental efficiency

**Example where parallelization fails:**
- Query: "Find closest image" among 1M
  - Even split 100 machines: each finds closest in their 10K
  - Still need to merge and find global closest
  - Can't parallelize the merge completely

---

### Technique 4: **Pipeline Design** (Staged Processing)

**Problem it solves:** Different stages have different costs; you want to fail fast.

**How it works:**
```
CHEAP STAGE          → MEDIUM STAGE       → EXPENSIVE STAGE
Fast pre-filter         Rough ranking        Precise ranking
(1ms)                  (10ms)               (100ms)
↓                       ↓                    ↓
1M images           → 100K images        → 1K images
```

**Real example:**
1. **Stage 1 (1ms):** Filter by metadata
   - Query: "Blue cat photo"
   - Keep only images labeled "cat" → 50K remain (20x reduction)

2. **Stage 2 (10ms):** Rough visual features
   - Check color histogram, texture → 5K remain (4x reduction)

3. **Stage 3 (100ms):** Expensive CNN embedding
   - Only on 5K images: 5K × 50ms = 250 seconds ✓ (instead of 14 hours)

**The numbers:**
- Without pipeline: 1M × 100ms = 100,000 seconds
- With pipeline: (1M × 1ms) + (100K × 10ms) + (5K × 100ms) = 1,000 + 1,000 + 500 = **2,500 seconds**
- **40x speedup**

**What happens without it:**
- You run the expensive stage on all 1M images. Wasteful.

---

## Part 4: Visual Guide - Putting It All Together

### Growth Curves: Why Algorithm Choice Matters
```
Time vs Input Size

Time (log scale)
  │     ╱╱╱╱╱╱  O(N²) - polynomial
  │    ╱╱╱╱    - gets bad fast
  │   ╱╱╱
  │  ╱╱ O(N) - linear
  │ ╱╱ - adds machines helps
  │╱
  │───────  O(log N) - logarithmic
  │         scales beautifully
  └────────────────── Input size
```

### The Scaled Search Pipeline (Visual)

```
INPUT: Find similar images

STAGE 1 (Cost: 1ms per image)
1M images → [Filter by metadata, color]
           → 50K candidates (50x reduction)

STAGE 2 (Cost: 10ms per candidate)
50K images → [Rough features, texture]
           → 5K candidates (10x reduction)

STAGE 3 (Cost: 100ms per candidate)
5K images → [CNN embedding, cosine similarity]
          → Results ✓

TOTAL TIME:
- Stage 1: 1M × 1ms = 1 second
- Stage 2: 50K × 10ms = 500ms
- Stage 3: 5K × 100ms = 500ms
- Total: ~2 seconds (vs 14 hours brute-force!)
```

### Parallelization: Where It Helps & Where It Doesn't

**HELPS:**
```
Task: Process 1M independent images

┌─────────────────┐
│ Machine 1: 100K │
├─────────────────┤
│ Machine 2: 100K │  → 10x speedup with 10 machines
├─────────────────┤
│ Machine 10: 100K│
└─────────────────┘
```

**DOESN'T HELP:**
```
Task: Find single closest point in 1M points

Machine 1 finds closest in its 100K: takes 100ms
Machine 2 finds closest in its 100K: takes 100ms
...
All machines done in 100ms (parallelized)
But now you need to find: closest of 10 closest = 1ms
Total: ~100ms (not 1000ms / 10 = 100ms)
Parallelization gives almost NO benefit
```

---

## Part 5: Summary

> **Three techniques work together: First, reduce the search space aggressively (metadata filtering, spatial hashing) so you're not doing expensive work on everything. Second, build a pipeline where cheap stages filter before expensive stages—this multiplies your reductions. Third, parallelize only the independent work (batch processing), not dependencies (finding global best). For example, finding similar images: instead of checking 1M images with a CNN (14 hours), filter to 50K with metadata (cheap), then 5K with rough features, then run the expensive CNN—total ~2 seconds.**

---

## Key Takeaways to Remember

| Challenge | Solution | Example Speedup |
|-----------|----------|-----------------|
| Problem too big | Candidate filtering | 50x–1000x |
| Expensive stage | Pipeline (cheap first) | 10x–100x |
| Purely parallelizable work | Add machines | Linear (N machines = N× speedup) |
| Sequential dependencies | Better algorithm | Can't parallelize |


