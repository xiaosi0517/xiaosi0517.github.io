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
    detail: Repeated cells, lines, and spacings define the field in which a registration target must be distinguished from its neighbors.
  - title: Windowed Geometry Processing
    detail: Local windows are extracted from the layout and rasterized so geometry can be compared with image-space matching.
  - title: Candidate Generation
    detail: Spatial constraints propose locations inside those windows instead of treating every coordinate as equally worth testing.
  - title: Template Matching
    detail: Normalized cross-correlation measures agreement with the template, while Sobel edge features keep the comparison focused on structural boundaries.
  - title: Quality Evaluation & Ranking
    detail: Candidates are ranked from several measurements, including correlation peak uniqueness. Adaptive early stopping skips the remaining work in a window once the evidence is already decisive.
  - title: Selected Registration Locations
    detail: The locations that remain are the ones with a stable correspondence, not merely the strongest local score.
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

A reliable registration location is not simply a window that resembles a template. In a large geometric layout, the same cells, lines, and spacings recur across the field. A local match can look convincing and still be one of many interchangeable positions.

Three effects make the search difficult:

- **Repetition** creates several plausible correspondences, often close together.
- **Rasterization** can soften the small geometric break that made one window distinctive.
- **Scale** makes an identical expensive test at every coordinate wasteful, while pruning too aggressively can skip the rare useful locations.

The engineering problem is to search this space selectively and then decide whether a correspondence is unique enough to trust.

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

The figure shows one tile of a much larger layout; the faded cells continue past every edge. Search is constrained to regions A, B, and C instead of scanning every coordinate. Regions A and C still yield only ambiguous candidates, because every cell inside them repeats. Region B is useful only because it happens to contain the one geometric break. A distinctive cell outside every region would never be proposed, which is why region placement is itself a design decision.

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

A correlation map shows how well the template agrees with the window at every offset. Normalized cross-correlation answers "how similar is this position?", not "is this position the only one that is similar?".

- **Left:** the window repeats, so the map is a lattice of nearly equal peaks. Peak 1 wins by a margin smaller than what noise or a slight edge change could produce, so the selected offset can jump to a neighboring copy.
- **Right:** a geometric break leaves one dominant peak. Its absolute height is lower than the best peak on the left, yet its position is unambiguous.

Peak uniqueness measures this directly. It compares the best peak with the strongest competing peak outside the best peak's immediate neighborhood. Ranking uses uniqueness together with correlation strength and Sobel edge support, so a strong but repeated response does not outrank a slightly weaker, unique one.

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

The processing order matters. Hierarchical layouts compress repetition, so expanding only what a window needs keeps memory bounded. Resolving overlaps and holes with polygon operations is exact, while doing the same in pixel space is not. Rasterization then becomes a mechanical step whose only free parameter is resolution, which is the representation template matching needs. The full pipeline follows.

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

Each reduction step saves work and carries a risk:

<div class="project-tradeoffs">
  <section>
    <h3>Search quality</h3>
    <p>A trustworthy location needs a separated correlation peak and edge support, not only a high score. Rejecting ambiguous peaks avoids correspondences that would move under small changes in the window.</p>
  </section>
  <section>
    <h3>Computational cost</h3>
    <p>Extracting windows, rasterizing them, computing Sobel features, and comparing competing peaks costs more than accepting the first maximum. That cost is what makes an ambiguous high score detectable.</p>
  </section>
  <section>
    <h3>Candidate coverage</h3>
    <p>Spatial constraints and early filters narrow the funnel. If they are too strict, the uncommon neighborhood that breaks the repeat is removed before it is ever scored.</p>
  </section>
  <section>
    <h3>Execution efficiency</h3>
    <p>Adaptive stopping skips the tail of the ranked list once enough reliable locations are accepted. It is only safe because ranking puts unique peaks first and the stop rule counts only accepted candidates, not high scores.</p>
  </section>
</div>
