---
layout: project
title: Robust Visual Search & Registration
description: Finding reliable visual correspondences in large-scale, repetitive, and ambiguous geometric patterns.
permalink: /projects/robust-visual-search-registration/
category: research
importance: 1
featured: true
img: assets/img/projects/robust-visual-search-registration.jpg
img_alt: Synthetic chip layout with callouts for two distinctive registration locations and a repetitive memory region
tech:
  - Computer Vision
  - Geometric Computing
  - Visual Search
  - Search Optimization
pipeline:
  - title: Geometric Layout
    detail: Keep repetition as placed instances, so the field does not have to be copied before a window is known.
  - title: Windowed Geometry Processing
    detail: Expand, clean, and rasterize only the geometry one search window needs.
  - title: Candidate Generation
    detail: Propose locations inside spatial constraints instead of every coordinate.
  - title: Template Matching
    detail: Score those candidates by how their structural edges agree with the template.
  - title: Quality Evaluation & Ranking
    detail: Prefer a unique, well-supported peak over the highest score.
  - title: Selected Registration Locations
    detail: Keep stable correspondences, and stop once enough of them are accepted.
highlights:
  - Windowed geometry extraction and rasterization
  - Spatially constrained candidate search
  - Normalized cross-correlation
  - Sobel edge features
  - Correlation peak uniqueness
  - Multi-metric candidate ranking
  - Adaptive early stopping
---

<p class="project-disclaimer">All figures on this page are conceptual illustrations built from synthetic geometry. None of them shows measured data, real layouts, or benchmark results.</p>

## The Challenge

A reliable registration location is not a window that merely resembles a template. In a large geometric layout, the same cells, lines, and spacings recur. A local match can look convincing and still be one of many interchangeable positions.

Three effects make a score-only, exhaustive search fail:

- **Repetition** produces several plausible correspondences, often close together. The strongest local score can land on the wrong copy.
- **Rasterization** can soften the small geometric break that made one window distinctive, so the cue that would separate the copies is easy to lose.
- **Scale** makes the same expensive test at every coordinate wasteful, while pruning too aggressively can skip the rare locations that are actually useful.

The design has to search selectively, then accept a correspondence only when it is unique enough to trust.

## Search Space Visualization

<div class="project-figure">
<div class="project-figure-head"><span class="project-figure-title">Figure 1 · Search complexity</span><span class="figure-badge">Conceptual · synthetic</span></div>
{% include figure.liquid loading="eager" path="assets/img/projects/search-space.svg" class="img-fluid project-diagram" alt="Synthetic layout tile with repeating cells, three dashed search regions labeled A, B, and C, orange candidate markers on repeating cells, and one blue candidate on the only distinctive cell" %}
<ul class="diagram-legend">
  <li><span class="legend-mark legend-geometry"></span> Repeating cell</li>
  <li><span class="legend-mark legend-distinct"></span> Distinctive cell</li>
  <li><span class="legend-mark legend-region"></span> Search region</li>
  <li><span class="legend-mark legend-dimmed"></span> Not searched</li>
  <li><span class="legend-mark legend-ambiguous"></span> Ambiguous candidate</li>
  <li><span class="legend-mark legend-selected"></span> Distinctive candidate</li>
</ul>
<p class="project-figure-caption">Most candidates are interchangeable. In a repetitive layout, almost every proposed location looks like its neighbors, and only one sits on geometry that does not repeat.</p>
</div>

Search is limited to regions A, B, and C because testing every coordinate does not scale. That choice has a cost: a distinctive cell outside every region is never proposed, so where the regions sit is part of the algorithm. Constraint alone is also not enough. Regions A and C still repeat, and only B contains the geometric break.

## Robust Matching

<div class="project-figure">
<div class="project-figure-head"><span class="project-figure-title">Figure 2 · Matching ambiguity</span><span class="figure-badge">Conceptual · synthetic</span></div>
<div class="row project-compare">
<div class="col-md-6 mb-2">
{% include figure.liquid loading="lazy" path="assets/img/projects/robust-matching-ambiguous.svg" class="img-fluid project-diagram" alt="Synthetic correlation map for a repetitive window: a lattice of nine nearly equal peaks; peak 1 and peak 2 have almost the same height" %}
</div>
<div class="col-md-6 mb-2">
{% include figure.liquid loading="lazy" path="assets/img/projects/robust-matching-distinctive.svg" class="img-fluid project-diagram" alt="Synthetic correlation map for a distinctive window: one dominant peak with weak secondary responses; a large gap separates peak 1 from peak 2" %}
</div>
</div>
<p class="project-figure-caption">The higher maximum is the less reliable match. Both maps share one color scale. The repetitive window on the left reaches the higher peak, but its runner-up is almost as strong.</p>
</div>

Normalized cross-correlation scores how well the template agrees with the window at each offset. It does not say whether that agreement is unique, which is the question repetition makes urgent.

Peak uniqueness compares the best peak with the strongest competitor outside its neighborhood. On the left, the winning peak is higher, but the gap is small enough that noise or a slight edge change can move the offset onto a neighboring copy. On the right, the peak is lower and the gap is decisive. Ranking uses uniqueness together with correlation strength and Sobel edge support, so a repeated high score cannot outrank a stable one. Sobel support is there because a match should be carried by structural boundaries, not by a smooth region that many cells share.

## System Architecture

<div class="project-figure">
<div class="project-figure-head"><span class="project-figure-title">Figure 3 · Geometry processing</span><span class="figure-badge">Conceptual · synthetic</span></div>
<div class="geometry-steps">
<div class="geometry-step">
<p class="geometry-step-title"><span>1</span> Hierarchical layout</p>
{% include figure.liquid loading="lazy" path="assets/img/projects/geometry-hierarchy.svg" class="img-fluid project-diagram" alt="Layout tree with a top cell placing four blocks, each placing nine cells, next to the corresponding repeated geometry" %}
<p>Each cell is defined once and placed by reference many times. Flattening the whole layout up front would multiply every repeated cell.</p>
</div>
<div class="geometry-step">
<p class="geometry-step-title"><span>2</span> Spatial window extraction</p>
{% include figure.liquid loading="lazy" path="assets/img/projects/geometry-window.svg" class="img-fluid project-diagram" alt="A dashed window over flattened geometry, with only the shapes inside the window kept and clipped at its boundary" %}
<p>Only instances that overlap the window are expanded, and their shapes are clipped at the window boundary.</p>
</div>
<div class="geometry-step">
<p class="geometry-step-title"><span>3</span> Polygon operations</p>
{% include figure.liquid loading="lazy" path="assets/img/projects/geometry-boolean.svg" class="img-fluid project-diagram" alt="Four overlapping rectangles merged by a polygon union into one frame-shaped polygon with an inner hole" %}
<p>A polygon union merges overlapping shapes into clean outlines and keeps holes explicit before any pixel exists.</p>
</div>
<div class="geometry-step">
<p class="geometry-step-title"><span>4</span> Rasterization</p>
{% include figure.liquid loading="lazy" path="assets/img/projects/geometry-raster.svg" class="img-fluid project-diagram" alt="The merged frame polygon sampled onto a coarse pixel grid; filled pixels approximate the dashed outline and the hole remains empty" %}
<p>The polygons are sampled onto a pixel grid. Edges snap to pixel boundaries, so resolution sets how much geometric detail survives.</p>
</div>
</div>
<p class="project-figure-caption">Geometry is resolved in vector space and only then converted to pixels. Work is limited to one window at a time, and topology is fixed before rasterization.</p>
</div>

That bound is what makes the later search selective. Geometry is paid for only where a region will actually be tested, so a larger field does not require a larger matcher.

{% include project_pipeline.liquid %}

## Engineering Trade-offs

<div class="project-figure">
<div class="project-figure-head"><span class="project-figure-title">Figure 4 · Search efficiency</span><span class="figure-badge">Conceptual · not measured</span></div>
<div class="search-funnel">
<div class="funnel-row">
<div class="funnel-track"><div class="funnel-bar funnel-all"></div></div>
<div class="funnel-text"><strong>Every window offset</strong><span>Exhaustive matching would correlate the template at every position in the layout.</span></div>
</div>
<div class="funnel-row">
<div class="funnel-track"><div class="funnel-bar funnel-constrained"></div></div>
<div class="funnel-text"><strong>Spatially constrained candidates</strong><span>Only locations inside search regions that satisfy geometric rules are proposed.</span></div>
</div>
<div class="funnel-row">
<div class="funnel-track"><div class="funnel-bar funnel-filtered"></div></div>
<div class="funnel-text"><strong>Pass quality filters</strong><span>Candidates with weak edge content or repeated surroundings are dropped before full scoring.</span></div>
</div>
<div class="funnel-row">
<div class="funnel-track"><div class="funnel-bar funnel-evaluated"></div></div>
<div class="funnel-text"><strong>Evaluated before stopping</strong><span>Ranked candidates are scored in order until enough reliable locations are accepted.</span></div>
</div>
</div>
<p class="stopping-title">Adaptive stopping on a ranked candidate list</p>
<ol class="stopping-strip" aria-label="Ranked candidates: accepted, rejected, accepted, rejected, accepted, then stop; the remaining candidates are not evaluated">
<li class="stop-accept">1</li>
<li class="stop-reject">2</li>
<li class="stop-accept">3</li>
<li class="stop-reject">4</li>
<li class="stop-accept">5</li>
<li class="stop-line" aria-hidden="true">stop</li>
<li class="stop-skip">6</li>
<li class="stop-skip">7</li>
<li class="stop-skip">8</li>
<li class="stop-skip">9</li>
<li class="stop-skip">10</li>
<li class="stop-skip">11</li>
<li class="stop-skip">12</li>
</ol>
<ul class="diagram-legend">
  <li><span class="legend-mark legend-accept"></span> Accepted: unique, well-supported peak</li>
  <li><span class="legend-mark legend-reject"></span> Rejected: ambiguous peak</li>
  <li><span class="legend-mark legend-skip"></span> Never evaluated</li>
</ul>
<p class="project-figure-caption">Most of the search work is never performed. Bar widths and candidate counts show the shape of the reduction, not measured savings.</p>
</div>

Each cut in the funnel saves work only if the locations it drops were not the ones worth trusting.

<div class="project-tradeoffs">
  <section>
    <h3>Search quality</h3>
    <p>A high score on a repeated pattern can move when the window changes slightly. Rejecting peaks that are not clearly separated avoids correspondences that would not stay put.</p>
  </section>
  <section>
    <h3>Computational cost</h3>
    <p>Comparing competing peaks and checking edge support costs more than taking the first maximum. That extra work is what makes an ambiguous high score visible.</p>
  </section>
  <section>
    <h3>Candidate coverage</h3>
    <p>Constraints and early filters shrink the search. Drawn too tightly, they remove the uncommon neighborhood that breaks the repeat before it is scored.</p>
  </section>
  <section>
    <h3>Execution efficiency</h3>
    <p>Stopping early is safe only because unique peaks are ranked first and the stop rule counts accepted locations, not high scores. Otherwise the search can halt on a repeated maximum.</p>
  </section>
</div>

## Engineering Outcomes

These are capabilities of the design, not measured benchmark results.

<div class="project-tradeoffs">
  <section>
    <h3>Reliable correspondence</h3>
    <p>A selected location can hold in a repeated pattern because acceptance depends on uniqueness, not on which copy scored highest.</p>
  </section>
  <section>
    <h3>Selective large-scale search</h3>
    <p>Most of the layout is never fully compared. The expensive test is reserved for candidates that already survived a spatial and structural filter.</p>
  </section>
  <section>
    <h3>Robustness from structure</h3>
    <p>Matches stay anchored to geometry that repetition does not erase. A change in pixel resolution does not silently rewrite which shapes exist.</p>
  </section>
  <section>
    <h3>An adjustable workflow</h3>
    <p>Coverage, ranking, and stopping can be tightened or relaxed on their own. The balance between reliability and cost does not require replacing the matcher.</p>
  </section>
</div>
