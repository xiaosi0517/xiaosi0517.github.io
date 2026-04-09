---
layout: post
title: Running a Modern Python CV Stack on a 15-Year-Old Linux Server
date: 2026-03-24
description: Lessons from deploying NumPy, SciPy, and OpenCV in an offline legacy engineering environment
tags: [blog-index, python, linux, devops, conda]
categories: [engineering]
giscus_comments: false
toc:
  beginning: true
---

## When Modern Software Meets Legacy Infrastructure

Installing a scientific Python stack today usually takes only a few minutes.
Most developers simply run something like:

```bash
pip install numpy scipy opencv-python
```

and move on.

But real-world engineering environments are often far less convenient.

Recently I needed to deploy a Python computer vision pipeline on a legacy Linux server used for engineering workloads. The pipeline relied on a fairly typical scientific stack:

- NumPy
- SciPy
- OpenCV
- scikit-image
- matplotlib

Nothing unusual. Except the target system looked like this:

| Property         | Value                      |
|------------------|----------------------------|
| Operating system | CentOS 6                   |
| glibc version    | 2.12                       |
| Network access   | None (air-gapped)          |
| Package sources  | Restricted to conda-forge  |

In other words, this was a fully offline system running infrastructure from more than a decade ago.

What initially seemed like a routine environment setup quickly turned into a surprisingly interesting engineering challenge.

---

## Why Legacy Systems Still Exist

In fast-moving software ecosystems like AI and web development, infrastructure evolves rapidly.

Industrial engineering environments are very different.

Many production systems prioritize **stability over frequent upgrades**. Upgrading an operating system may break critical components such as:

- Vendor toolchains
- Simulation pipelines
- Hardware drivers
- Certified production workflows

Because of this, it is still quite common to encounter systems like CentOS 6, CentOS 7, RHEL-based HPC clusters, and air-gapped compute nodes.

These systems can run reliably for years. However, deploying modern software stacks on them can be surprisingly difficult.

**The gap between modern software ecosystems and legacy infrastructure is often larger than expected.**

---

## Scientific Python Is Not Just Python

A common misconception is that scientific Python libraries are simply Python packages.

In reality, many of them rely heavily on **compiled native code**.

Libraries such as NumPy, SciPy, OpenCV, and h5py depend on underlying C and C++ libraries including BLAS, libstdc++, image codecs, and compression libraries.

If these binary dependencies are incompatible with the system environment, installation fails.

This was the first major obstacle. Most modern Python wheels assume **glibc >= 2.17**, but the target system was running **glibc 2.12**.

That alone eliminated many installation paths.

---

## Why pip Wheels Didn't Work

One of the first approaches we tried was the most obvious one: installing packages via pip wheels.

The idea was simple. Download the wheels on a machine with internet access and install them offline:

```bash
pip download --platform manylinux2014_x86_64
pip install --no-index --find-links wheels/
```

Unfortunately this failed immediately with an error like:

```
ImportError: /lib64/libc.so.6: version 'GLIBC_2.17' not found
```

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blog/legacy-linux/pip-failure-flowchart.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Why pip install fails on legacy Linux: the manylinux2014 wheel requires glibc >= 2.17, but the system only has glibc 2.12." %}
    </div>
</div>

The reason is that most modern scientific Python wheels follow the **manylinux2014** standard, which assumes **glibc >= 2.17**. Since the server was running glibc 2.12, the dynamic loader could not resolve the required symbols.

In practice this means that many pip wheels for packages such as numpy, scipy, scikit-learn, and opencv-python simply **cannot run on very old Linux systems**.

This quickly ruled out pip as the primary installation strategy.

---

## The Offline Environment Problem

The next challenge was that the server environment was completely offline.

Typical installation commands like `pip install` and `conda install` were not possible. All packages needed to be downloaded elsewhere and transferred manually.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blog/legacy-linux/compatibility-gap-solution.png" class="img-fluid rounded z-depth-1" zoomable=true caption="The compatibility gap between modern and legacy environments, and the offline deployment pipeline that bridges it." %}
    </div>
</div>

This meant the workflow had to follow a multi-step pipeline:

1. **Online machine** -- resolve dependencies
2. **Export** -- generate explicit environment specification
3. **Download** -- fetch all package tarballs
4. **Transfer** -- move to the offline server
5. **Install** -- recreate the environment without a solver

However, resolving the dependency graph turned out to be another challenge.

---

## A Small Trick: Using Micromamba for Dependency Solving

Modern scientific Python environments can easily contain **hundreds of packages** once all dependencies are included.

Libraries like OpenCV in particular pull in a large dependency tree including image libraries, compression tools, and sometimes GUI components.

When attempting to resolve the environment using the classic conda solver, the process became extremely slow and memory intensive. The solver spent a long time exploring dependency combinations and consumed several gigabytes of RAM.

A much better approach was to use **micromamba**, which relies on the **libsolv** dependency solver. In practice this makes a significant difference for large environments:

```bash
micromamba create -n env --dry-run -f environment.yml
```

Using `--dry-run` allows dependency resolution without extracting packages, which is especially helpful when resolving Linux environments from another platform.

This made dependency resolution much faster and allowed the full environment specification to be generated reliably.

---

## The Key Trick: Explicit Conda Environments

The most reliable approach ultimately turned out to be using **explicit conda environments**.

Instead of installing packages incrementally, the entire dependency graph is solved once and then exported. The key command is:

```bash
conda list --explicit
```

This produces a file containing the **exact package URLs** required to recreate the environment.

The workflow becomes:

1. Resolve dependencies on the online machine
2. Export the explicit environment
3. Download all packages
4. Transfer to the offline server
5. Create the environment offline (no solver needed)

Because the dependency graph is already solved, the offline server does not need to run the solver again. This **dramatically improves reliability** in constrained environments.

---

## Lessons Learned

Several practical lessons came out of this process.

**Resolve dependencies only once.** Incremental installations often produce incompatible dependency trees.

**Explicit environments improve reproducibility.** Exporting exact package versions avoids dependency drift.

**Conda works better than pip for binary stacks.** Conda packages bundle many system libraries, making them easier to deploy on constrained systems.

**Avoid unnecessary dependencies.** For compute pipelines, GUI libraries are often unnecessary and can complicate installations.

---

## AI Tools and Real Engineering Work

During this process I frequently used AI tools to understand error messages, dependency conflicts, and packaging behavior. They were extremely helpful.

But the final solution still required:

- Experimentation
- System understanding
- Iterative debugging

Real engineering problems rarely exist in perfectly documented environments. They often involve constraints like legacy infrastructure, restricted networks, and cross-platform dependencies.

**AI can accelerate the debugging process, but solving these problems still requires human reasoning about systems.**

---

## Final Thoughts

Modern software ecosystems evolve rapidly. Infrastructure often evolves much more slowly.

Bridging modern tools with legacy systems remains an important engineering skill.

Sometimes the most interesting engineering problems are not about building new algorithms. **They're about making modern tools work in environments that were never designed for them.**
