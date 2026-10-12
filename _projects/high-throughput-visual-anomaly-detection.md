---
layout: project
title: High-Throughput Visual Anomaly Detection
description: Detecting and explaining meaningful anomalies across large-scale image datasets.
permalink: /projects/high-throughput-visual-anomaly-detection/
category: research
importance: 2
featured: false
img: assets/img/projects/anomaly-field.svg
img_alt: Synthetic field of noisy geometric images, with a few tiles marked as bright, dark, gray, and background departures
tech:
  - Computer Vision
  - Robust Statistics
  - Parallel Computing
  - Visual Diagnostics
pipeline:
  - title: Image Inputs
    detail: Each observation is one noisy view of the same kind of geometry.
  - title: Registration & Alignment
    detail: Place the observation on the reference. A weak agreement is kept as a result instead of being forced into a match.
  - title: Region-Based Measurements
    detail: Read intensity where the geometry expects structure, and treat the empty field as a separate question.
  - title: Parallel Statistical Analysis
    detail: Summarize the collection with a median and a spread that extremes do not define.
  - title: Anomaly Identification
    detail: Mark departures from that typical behavior, and keep the kind of departure attached to the mark.
  - title: Visual Diagnostics
    detail: Show the expected pattern, the observation, and the region so a person can check the flag.
highlights:
  - Image-to-reference alignment
  - Region-based intensity measurements
  - Median and MAD
  - Parallel per-image measurement
  - Signature-level departures
  - Expected-versus-observed review
---

<p class="project-disclaimer">All figures on this page are conceptual illustrations built from synthetic geometry. None of them shows measured data, real images, or benchmark results. The review layout is an illustration, not a copy of internal software.</p>

## The Challenge

A useful anomaly is not every pixel that differs from its neighbor. In a large collection, images are noisy, the expected geometry repeats, and small changes sit inside normal variation. The task is to notice a mismatch that matters, say what kind of mismatch it is, and still finish the collection.

Three pressures shape the design:

- **Volume.** Reading every image by eye does not survive the size of the collection.
- **Noise.** Grain is normal. A rule that flags all of it hides the departures that matter.
- **Mixed causes.** A bright patch, a dark gap, a washed-out tone, an extra background mark, and a failed alignment do not look alike, and they are not one kind of error.

The design measures images in a consistent way, compares those measurements with what is typical for the collection, and then shows the evidence.

## Understanding Visual Anomalies

<div class="project-figure">
<div class="project-figure-head"><span class="project-figure-title">Figure 1 · Where departures hide</span><span class="figure-badge">Conceptual · synthetic</span></div>
{% include figure.liquid loading="eager" path="assets/img/projects/anomaly-field.svg" class="img-fluid project-diagram" alt="Synthetic grid of noisy line-pattern images. Most tiles repeat. Four are outlined as a bright patch, a dark gap, a gray wash, and an unexpected background mark." %}
<p class="project-figure-caption">Most of the field is ordinary variation. The outlined tiles are the ones that ask a different question than their neighbors.</p>
</div>

<div class="anomaly-gallery">
<div class="anomaly-card">
<p class="geometry-step-title"><span>1</span> Normal pattern</p>
{% include figure.liquid loading="lazy" path="assets/img/projects/anomaly-normal.svg" class="img-fluid project-diagram" alt="Expected lines, a noisy observation of the same lines, and a deviation panel with nothing marked." %}
<p>Grain is present, and nothing is marked. Normal variation belongs to the reference, not to the anomaly list.</p>
</div>
<div class="anomaly-card">
<p class="geometry-step-title"><span>2</span> Bright anomaly</p>
{% include figure.liquid loading="lazy" path="assets/img/projects/anomaly-bright.svg" class="img-fluid project-diagram" alt="Expected lines beside an observation with a bright block, and a deviation panel that outlines only that block." %}
<p>A region is far lighter than the lines around it. The departure is local and high.</p>
</div>
<div class="anomaly-card">
<p class="geometry-step-title"><span>3</span> Gray anomaly</p>
{% include figure.liquid loading="lazy" path="assets/img/projects/anomaly-gray.svg" class="img-fluid project-diagram" alt="Expected lines beside an observation washed in an intermediate tone, and a deviation panel outlining that washed region." %}
<p>The structure is still there, but an intermediate tone has flattened it. This is a loss of contrast, not a missing shape.</p>
</div>
<div class="anomaly-card">
<p class="geometry-step-title"><span>4</span> Dark anomaly</p>
{% include figure.liquid loading="lazy" path="assets/img/projects/anomaly-dark.svg" class="img-fluid project-diagram" alt="Expected lines beside an observation with a dark gap, and a deviation panel outlining that gap." %}
<p>A gap replaces lines the geometry expects. The departure is local and low.</p>
</div>
<div class="anomaly-card">
<p class="geometry-step-title"><span>5</span> Background anomaly</p>
{% include figure.liquid loading="lazy" path="assets/img/projects/anomaly-background.svg" class="img-fluid project-diagram" alt="Expected lines beside an observation with an extra mark outside the pattern, and a deviation panel outlining only that mark." %}
<p>The lines agree, and an extra mark sits where the pattern should be empty. Measuring only the expected lines would miss it.</p>
</div>
<div class="anomaly-card">
<p class="geometry-step-title"><span>6</span> Low-correlation alignment</p>
{% include figure.liquid loading="lazy" path="assets/img/projects/anomaly-alignment.svg" class="img-fluid project-diagram" alt="Expected lines beside a shifted observation inside a dashed reference frame, and a deviation panel showing the offset between two frames." %}
<p>The pattern is shifted relative to the reference frame. Treating that shift as a local blot describes the wrong problem.</p>
</div>
</div>

These are ways of describing a mismatch, not a set of exclusive classes. One image can show more than one. A weak alignment is different in kind from a local intensity change: the pattern may be intact and simply not where the measurement assumed.

## System Architecture

Alignment, measurement, summary, and review fail in different ways, so they stay separate. A weak alignment means the measurement is not about the intended region. A measurement can be faithful and still be typical. A real departure still needs a person to see which signature it is.

{% include project_pipeline.liquid %}

Per-image work comes first because alignment and region readings do not need the rest of the collection. The typical image is a property of the set, so the summary waits. Review waits longer: it explains a flag, and it does not have to ride along with every measurement.

## Robust Statistical Anomaly Detection

<div class="project-figure">
<div class="project-figure-head"><span class="project-figure-title">Figure 2 · What “typical” should ignore</span><span class="figure-badge">Conceptual · not measured</span></div>
{% include figure.liquid loading="lazy" path="assets/img/projects/anomaly-stats.svg" class="img-fluid project-diagram" alt="Conceptual histogram of region intensity. A shaded band sits around the median. A dashed mean is pulled toward two separated orange bars." %}
<p class="project-figure-caption">The cluster is the typical image. The orange bars are departures. Bar heights illustrate the shape of the problem, not a dataset.</p>
</div>

A mean, and a spread based on squared deviations, moves when a few images are extreme. Those images then help define what counts as normal, which is the opposite of what a detector needs. The median stays with the dense part of the collection. The median absolute deviation measures spread from that median, so the same extremes do not set the width of the band.

A candidate departure is a measurement that sits far from the median relative to that width. How far is a tuning choice. This page does not state a threshold, a calibration, or a detection rule.

## High-Throughput Processing

<div class="project-figure">
<div class="project-figure-head"><span class="project-figure-title">Figure 3 · Measure in parallel, summarize after</span><span class="figure-badge">Conceptual · not measured</span></div>
{% include figure.liquid loading="lazy" path="assets/img/projects/anomaly-parallel.svg" class="img-fluid project-diagram" alt="A stack of synthetic images fans out to four workers that measure independently, then gather into one summary. A separate box says review comes later." %}
<p class="project-figure-caption">Workers do not need each other’s images. The summary does. The figure shows that split, not a runtime or a machine count.</p>
</div>

Alignment and region measurements for one image do not depend on the others, so that work can run side by side as the collection grows. Median, spread, and the list of departures cannot start until those measurements exist. Opening every picture during that pass would spend time on images that never depart. Review stays after the summary, on the images a person needs to see.

## From Detection to Diagnosis

<div class="project-figure">
<div class="project-figure-head"><span class="project-figure-title">Figure 4 · From a flag to a look</span><span class="figure-badge">Illustrative · synthetic</span></div>
{% include figure.liquid loading="lazy" path="assets/img/projects/anomaly-review.svg" class="img-fluid project-diagram" alt="Illustrative review screen with signature filters, an expected pattern beside an observation, an outlined bright region, and a short list of flagged regions." %}
<p class="project-figure-caption">The pair of images is the explanation. Filters group signatures. Nothing in this layout is a measured count or an internal tool.</p>
</div>

A flag answers whether a measurement is unusual. It does not show what an engineer should look at. The review puts the expected pattern next to the observation, outlines the region that departed, and keeps the alignment in view so a shift is not mistaken for a local blot. Grouping by signature lets a person walk through one kind of mismatch at a time.

## Engineering Trade-offs

<div class="project-tradeoffs">
  <section>
    <h3>Sensitivity and false alarms</h3>
    <p>Widening what counts as a departure catches subtler mismatches and also promotes grain into the review queue. Tightening it keeps the queue short and can pass over a real but small signature.</p>
  </section>
  <section>
    <h3>Throughput and what gets measured</h3>
    <p>Reading only the expected regions is cheaper, and it can miss a surprise in the background. Reading the whole image costs more, and that cost is what makes an unexpected mark visible. The two questions stay separate so the extra work is paid only where the question requires it.</p>
  </section>
  <section>
    <h3>Automation and a check a person can make</h3>
    <p>A score scales past a manual scan of the collection. It is not an explanation. The review exists so a person can confirm the signature and the alignment instead of trusting the score alone.</p>
  </section>
</div>

## Engineering Outcomes

These are capabilities of the design, not measured results.

<div class="project-tradeoffs">
  <section>
    <h3>The same measurement, at collection scale</h3>
    <p>Each image is aligned and read in the same way. The collection is summarized only after those readings exist, so the analysis can grow with the set.</p>
  </section>
  <section>
    <h3>Signatures stay distinguishable</h3>
    <p>Bright, gray, dark, and background mismatches are not collapsed into one score. A failed alignment stays visible as an alignment problem.</p>
  </section>
  <section>
    <h3>Investigation starts from the flag</h3>
    <p>Review opens on the expected pattern, the observation, and the region that departed, rather than on a pass through every image.</p>
  </section>
  <section>
    <h3>Review stays off the measurement pass</h3>
    <p>Pictures are opened to account for a departure. Images that stay inside the typical band do not have to be rendered for that account.</p>
  </section>
</div>
