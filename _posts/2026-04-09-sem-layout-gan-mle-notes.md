---
layout: post
title: "SEM ↔ Layout GAN — MLE Technical Notes"
date: 2026-04-09 12:00:01
description: Structured narrative, mock Q&A, and deep-dive follow-ups for SEM/layout GAN systems — personal notes (unlisted).
tags: []
categories: []
private: true
feed: false
sitemap: false
giscus_comments: false
toc:
  beginning: true
---

Structured explanation, mock Q&A, and deep-dive follow-ups for technical review. Aligned with `DESIGN.md` and the codebase.

---

## Table of contents

1. [Part 1 — Structured explanation (5–10 min narrative)](#part-1--structured-explanation-510-min-narrative)
2. [Part 2 — Mock practice questions](#part-2--mock-practice-questions)
3. [Part 3 — Deep dive questions (MLE level)](#part-3--deep-dive-questions-mle-level)

---

# Part 1 — Structured explanation (5–10 min narrative)

## Opening (30 seconds)

> I built a bidirectional image-to-image translation system for the semiconductor domain. It converts between Scanning Electron Microscope images and design layout templates using a GAN-based architecture. SEM-to-Layout recovers the intended design from physical measurement; Layout-to-SEM predicts what a design will look like under the microscope. The system includes a production serving layer with monitoring for drift, degradation, and a human-in-the-loop feedback pipeline.

## 1. Problem & motivation

**What it solves.** In semiconductor manufacturing, engineers compare what was *designed* (layout) with what was *fabricated* (SEM). Manual comparison does not scale. This system automates bidirectional translation so engineers can verify fabrication fidelity or predict measurement outcomes from a design.

**Why it is hard.**

- **Noise:** SEM images have beam noise, contrast variation, and material-dependent intensity; they are not clean binary images like layouts.
- **Alignment:** Raw SEM (304×304) is larger than templates (250×250). Naive center crop introduces up to roughly ±25 pixels of misalignment, which breaks paired pixel-level supervision.
- **Domain gap:** SEM is continuous-valued grayscale with noise; layout is near-binary with sharp edges. The generator must learn a highly nonlinear mapping.
- **Limited data:** On the order of ~2k paired samples total, with imbalance (e.g., line/space vs via/contact).

## 2. Data understanding

**Alignment is feature engineering, not trivial preprocessing.** Normalized Cross-Correlation (NCC) template matching (`cv2.matchTemplate` with `TM_CCOEFF_NORMED`) finds the optimal translational offset; SEM may be Gaussian-blurred before matching to reduce noise, then cropped at the peak. Low NCC scores are flagged for QC.

**Why alignment is critical.** L1 compares pixels at corresponding locations. Misalignment means the model is penalized for correct structure at wrong coordinates and rewarded for wrong structure — the loss landscape is noisy and the generator struggles to learn sharp mappings.

**Normalization.** `uint8 [0, 255]` → `float32 [-1, 1]` via `pixel / 127.5 - 1.0`, matching Tanh output range and improving gradient behavior.

**Augmentation.** Only geometric transforms (horizontal/vertical flip, 90° rotation), applied identically to SEM and template per pair. No photometric augmentation: SEM intensity carries physical meaning (contrast, beam settings) that should be learned, not randomized away.

## 3. Model choice

**GAN vs pure regression CNN.** Pure L1/L2 regression tends toward the conditional mean and blurry outputs. Adversarial training pushes outputs toward realism and sharpness beyond mean regression.

**Why not diffusion (for this project).** Diffusion typically needs more data, has slower inference (many steps), and this task often behaves like a constrained mapping where geometric accuracy matters more than sampling diversity.

**Bidirectional (cycle).** With limited pairs, one direction can overfit. Cycle consistency regularizes by requiring approximate inverses between domains — a structural prior that tightens the hypothesis space.

## 4. Architecture deep dive

### Generator (U-Net)

- **Encoder–decoder** with skip connections; channel schedule e.g. `min(base_filters × 2^k, 256)`.
- **Skip connections** concatenate encoder features at matching resolutions so the decoder uses both semantics (bottleneck) and fine spatial detail (encoder).
- **Bottleneck** holds the most compressed representation; deeper networks (e.g., `n_down=5`) use an 8×8 bottleneck for more global context vs 16×16.
- **Decoder** uses transposed convolutions, BN, ReLU, dropout on early decoder blocks in this project.

### Discriminator (PatchGAN, conditional)

- Input is **concatenation** of `(input, output)` so D judges plausibility *conditioned on* the source image.
- Outputs a **spatial map** of logits; with typical settings the effective receptive field is on the order of **~70×70** pixels — local patches, not one global score.
- **Why local:** SEM/layout quality is dominated by edges, linewidths, and local pattern structure; localized feedback tells G *where* to fix artifacts.

### Attention (improved model)

- Self-attention **augments** U-Net; two blocks at **low resolution** (e.g., 8×8 / 16×16) keep attention cost tractable.
- **Why low resolution only:** Full-resolution self-attention on 256×256 is expensive; small data also argues against huge attention-heavy models.
- **Gamma initialized to 0:** attention starts as identity and ramps in, stabilizing early training.
- **Role:** long-range spatial structure (e.g., periodic patterns) while U-Net + skips handle local detail.

## 5. Activation functions

- **LeakyReLU** in encoder/discriminator: avoids dead ReLU regions and preserves gradient flow in adversarial training.
- **ReLU** in decoder (Pix2Pix-style convention).
- **Tanh** at generator output: bounded [-1, 1] matching input normalization.
- **Kaiming init** (PyTorch defaults): appropriate for ReLU/LeakyReLU stacks; attention `gamma=0` is a separate stability choice.

## 6. Loss functions (critical)

| Component | Role | If removed / weakened |
|-----------|------|------------------------|
| **LSGAN (MSE on D outputs)** | Smoother gradients than saturated BCE when D is confident | Easier “D wins early,” training collapse or blur |
| **Paired L1 (λ large, e.g. 100)** | Pixel supervision from aligned pairs | Realistic but wrong geometry, or unstable GAN-only behavior |
| **Cycle L1 (λ e.g. 10)** | Round-trip reconstruction | Degenerate mappings, worse generalization with limited data |
| **Feature matching (attention, λ e.g. 10)** | Match D intermediate features real vs fake | Less texture/structure signal; can be less stable |
| **Identity (optional, often 0)** | Penalize changing inputs already in target domain | Less color/identity preservation when domains differ |

**Intuition for total G loss:** average adversarial pressure across both directions (small effective weight vs L1), plus strong L1 anchoring, plus cycle regularization, plus FM when enabled.

**D side:** real vs fake with **label smoothing** on real labels (e.g. 0.9) to stop D from becoming overconfident.

## 7. Training stability

**Why unstable:** non-stationary minimax game; capacity imbalance; saturated losses.

**“D wins too early” (baseline):** BCE can saturate; D becomes too accurate too fast → tiny gradients to G.

**Mitigations used in improved configs:**

- **LSGAN:** gradients scale with distance from target, not only at the decision boundary.
- **Spectral normalization (D):** bounds Lipschitz behavior of D.
- **Label smoothing:** prevents perfect D confidence on reals.
- **Image buffer:** D trains on a mix of recent fakes, reducing oscillation.
- **Adam β₁ = 0.5:** less momentum than 0.9 — more responsive in non-stationary optimization.

## 8. Evaluation metrics

| Metric | Measures | Notes |
|--------|----------|--------|
| **L1** | Mean absolute error in [0,1] after remap | Simple; can miss structural errors |
| **SSIM** | Structural similarity | Often primary for promotion; closer to perceptual structure |
| **PSNR** | Log-scale MSE-based quality | Common benchmark; can disagree with perception |
| **IoU** | Overlap after binarization (e.g. 0.5) | Layout-like fidelity; threshold-sensitive |

**Why SSIM is preferred here:** emphasizes structural agreement beyond raw pixel averaging.

## 9. System design (production)

**Serving:** FastAPI → `InferenceEngine` → `G_A` or `G_B` depending on direction; checkpoint and config driven.

**Inference pipeline:** decode upload → grayscale → resize to `image_size` → normalize → forward → postprocess to uint8.

**Three monitors (SQLite-backed in design):**

1. **InferenceMonitor:** latency, errors, lightweight input/output stats — *operational* health.
2. **DriftDetector:** scalar features + histogram vs training reference — *input* shift early warning.
3. **PerformanceTracker:** register at inference; later ground-truth submission — *delayed* quality truth.

**Silent failure:** model returns 200 OK with plausible-looking wrong geometry; no exception. Dangerous because ops metrics look fine. Mitigation: drift alerts + human GT pipeline + SSIM/L1 trends on reviewed samples.

## 10. Feedback loop & continuous learning

- Collect reviewed pairs via API; accumulate until triggers.
- **Triggers (example):** enough new pairs, high drift rate, SSIM degradation vs baseline — with cooldown.
- **Fine-tune:** low LR, few epochs, mix original + feedback; evaluate before promotion.
- **Promotion:** gate on SSIM (and logging) vs current production.

## 11. Trade-offs & design decisions

- **PatchGAN** vs global D: local structure and sharpness vs single global score.
- **Cycle** vs one-way: regularization and invertibility vs simpler training.
- **L1-heavy** vs perceptual (ImageNet VGG): domain mismatch risk for grayscale SEM; L1 matches metrology goals.
- **Not full Transformer:** data size + need for U-Net skips + cost of full-resolution attention.

## 12. Failure cases

- Misalignment residual, poor NCC, OOD pattern types.
- Blur from over-strong averaging or weak adversary.
- **Detection:** drift stats, visual QC, cycle reconstruction checks, reviewed metrics.

## 13. Generalization & robustness

- Train vs deployment shift (tooling, materials, patterns).
- Imbalance biases dominant pattern types.
- **Improvements:** stratified sampling, domain-specific augmentations where valid, monitoring per cohort, more labeled OOD.

## 14. Scaling & improvements

- More data: larger/wider nets, better coverage of rare patterns.
- More compute: longer training, architecture search, distillation for latency.
- Multimodal / retrieval / agents: optional layers on top of core translation (metadata, defect DB lookup, etc.).

## 15. Behavioral / system thinking (talking points)

- **Hardest problem:** often alignment/QC — bad pairs cap model quality regardless of architecture.
- **Biggest trade-off:** pixel accuracy (high λ_L1) vs purely “realistic” GAN outputs.
- **Redesign:** domain-specific perceptual metrics, richer monitoring (population drift), active learning for labels.

---

# Part 2 — Mock practice questions

### Q1: Walk me through how data flows from a raw SEM image to a model prediction.

**A:** Raw SEM (304×304) → NCC template matching with optional Gaussian blur finds optimal crop → aligned SEM (250×250) → resize to 256×256 → normalize to [-1, 1] → pass through U-Net generator (`G_A` for SEM→Layout) → Tanh output in [-1, 1] → denormalize to [0, 255] uint8. The 256×256 size supports clean U-Net pooling (divisible by 2^n_down). Alignment is critical: without it, paired L1 supervision is misregistered and the generator cannot learn sharp mappings.

### Q2: Why did you choose L1 loss with high weight instead of a perceptual loss (e.g., VGG features)?

**A:** (1) Perceptual losses often use ImageNet-pretrained VGG on natural RGB images; SEM is single-channel with different statistics — features may be poorly calibrated. (2) For semiconductor use cases, geometric precision often matters more than “natural” appearance. L1 directly optimizes pixel alignment. The large λ_L1 vs the small effective adversarial weight deliberately prioritizes structural fidelity over pure realism.

### Q3: What happens if you remove cycle consistency loss?

**A:** The two generators are no longer constrained to be approximate inverses. With limited data, each generator can learn shortcuts: outputs that satisfy paired L1 and fool D on training pairs but fail to preserve information needed for reconstruction or generalization. Cycle loss forces approximate invertibility and reduces degenerate mappings.

### Q4: Explain spectral normalization intuitively.

**A:** Spectral normalization rescales weight matrices using their spectral norm, constraining the Lipschitz constant of the discriminator. That prevents D from developing arbitrarily sharp decision boundaries too quickly, which would give vanishing gradients to G. It bounds D’s “power” without removing useful gradient signal.

### Q5: Why not use a Transformer / ViT for the generator?

**A:** (1) Data scale: Transformers often need large datasets; this project has ~2k pairs. (2) Cost: self-attention at full 256×256 resolution is expensive. (3) U-Net skip connections are strong for pixel-level reconstruction. The codebase adds self-attention only at low resolution (few tokens) on top of U-Net rather than replacing it.

### Q6: How does PatchGAN improve over a global discriminator?

**A:** A global D outputs one score for the whole image, averaging over all locations. PatchGAN outputs a map of scores; each location corresponds to a local receptive field (~70×70 in this design). The generator gets localized feedback (“this region looks fake”), which helps high-frequency structure (edges, linewidths) that matter for SEM and layout.

### Q7: Why is the discriminator input two channels (concatenated input and output)?

**A:** It is a *conditional* discriminator: D judges whether the output is plausible *for that specific source image*, not whether it looks like a generic real layout or SEM. That ties adversarial learning to the correct cross-domain mapping.

### Q8: Explain the image buffer and why it helps.

**A:** A buffer stores recent fake images. D trains on a mix of current and past fakes instead of only the latest batch. That reduces mode oscillation: D is less likely to overfit to the newest G behavior, stabilizing the adversarial dynamics.

### Q9: What is “silent failure” and how does the system detect it?

**A:** Silent failure: the model returns plausible-looking outputs that are geometrically wrong, with normal latency and no errors — ops dashboards look healthy. Detection layers: drift detection (input shift), performance tracker when ground truth arrives (L1/SSIM/PSNR/IoU), and human review. Drift is an early warning; delayed labels are ground truth on quality.

### Q10: Why only geometric augmentation, not photometric?

**A:** SEM intensity carries physical information (contrast, tool settings). Random brightness/contrast can push the model toward invariance to signals it should use. Geometric transforms (flips, 90° rotations) preserve the pairing and are safe symmetries for many layout patterns.

### Q11: How would you handle class imbalance between line/space and via/contact patterns?

**A:** Options: stratified sampling or weighted sampling; per-pattern-type evaluation; train separate models or a router; oversample rare classes in the batch. The design doc notes training on homogeneous subsets or curated multi-dataset configs as practical mitigations.

### Q12: Explain the linear LR decay schedule.

**A:** Learning rate stays constant early, then linearly decays to zero in later epochs. That allows coarse learning first and finer convergence later, reducing oscillation when near a good basin. In config: e.g. linear decay starting at epoch 50 for 100 total epochs.

### Q13: Why use `.detach()` in feature matching loss?

**A:** Feature matching compares D’s features on fake vs real; real features are `.detach()` so gradients do not flow into the “real” path. Otherwise the loss could train D to make real and fake features identical, destroying discrimination. Only G should be updated to match fixed real feature targets.

### Q14: How does drift detection work and what are its limitations?

**A:** Reference statistics from training images (mean, std, edge density, histogram). Incoming images are compared via z-scores and histogram KL vs thresholds. Limitation: distributional shift is not the same as semantic failure — a new pattern type could look statistically similar and still fail. Performance tracking with labels is the backstop.

### Q15: What would you change about the fine-tuning strategy?

**A:** Possible upgrades: EMA of weights; elastic weight consolidation or replay; freeze early layers if feedback is tiny; validate on original-domain holdout during fine-tune to catch catastrophic forgetting; stronger promotion gates.

---

# Part 3 — Deep dive questions (MLE level)

### D1: Receptive field ~70×70 — how is it related to `n_layers`, and what if you change it?

**A:** RF grows with stacked strided convolutions and kernel sizes. Increasing `n_layers` generally increases RF so each output position “sees” a larger image patch — more global context, potentially less emphasis on very fine local texture. For SEM/layout, ~70×70 is a design choice balancing local edge quality vs broader context.

### D2: Kaiming initialization for LeakyReLU vs ReLU — what changes?

**A:** Kaiming init accounts for activation variance (including negative slope for LeakyReLU). The scaling differs slightly from pure ReLU. In practice PyTorch defaults match the specified nonlinearity when configured correctly.

### D3: Cycle reconstruction uses L1 — why not adversarial loss on the reconstruction?

**A:** The goal of cycle consistency is to recover the *same* input instance after round-trip translation, not merely to produce a plausible image in the source domain. L1 enforces pixel-level identity of the reconstruction; a GAN loss on reconstruction would be easier to satisfy with diverse plausible outputs.

### D4: Good L1 but poor IoU — how do you diagnose and fix?

**A:** Small systematic errors concentrated at edges can yield decent L1 but bad overlap after binarization. Diagnose with overlays and edge maps. Fixes: alignment QC, sharper losses near edges, post-processing, or adjusting IoU thresholding strategy to match the deployment definition of “correct.”

### D5: Drift detector uses KL on histograms — KL is asymmetric. Does direction matter?

**A:** Yes. KL(p‖q) measures surprise of p under q. For “does this input look like training?” you choose direction and smoothing consistently. Asymmetry means KL(q‖p) is not the same problem — interpretability and thresholds depend on the chosen direction and numerical stabilization.

### D6: Why freeze the discriminator during the generator update?

**A:** When training G, you want gradients through G to improve fakes; you should not simultaneously update D in a way that collapses the game in the same step. `requires_grad=False` on D during the G step also saves memory and compute.

### D7: ~2k pairs — when would you consider switching to diffusion?

**A:** Diffusion often needs much more data for stable conditional generation at this resolution, and inference is many steps slower. This task is closer to constrained translation with paired supervision than open-ended generative modeling — GAN + strong L1 + cycle is a reasonable engineering fit at this scale.

### D8: How would you scale to ~100k pairs and sub-10 ms inference?

**A:** Data: stratified sampling, curriculum. Model: larger capacity or distilled student. Serving: ONNX/TensorRT, batching, INT8 quantization, multi-GPU or replicated workers. Sub-10 ms depends on hardware and model size but these are the standard levers.

### D9: Feature matching averages across D layers — should earlier vs later layers be weighted differently?

**A:** Earlier layers lean toward edges and local statistics; later layers toward global texture and semantics. Uniform averaging is a baseline; if edges dominate metrology quality, up-weighting earlier layers is a reasonable experiment.

### D10: Forward translation looks good but cycle reconstruction is poor — what does that indicate?

**A:** The forward mapping may discard information still compatible with one-way L1 but needed for inversion. Increase cycle weight, consider identity loss if domains warrant it, or inspect whether paired supervision dominates too strongly over cycle.

### D11: When would monitoring fail to catch degradation?

**A:** Slow drift that stays under per-image z-score thresholds; insufficient labeled evaluations in the performance window; feedback latency so long that trends are invisible. Mitigations: population-level trend tests, more frequent spot checks, richer slice-based metrics.

### D12: Why does the discriminator often omit BatchNorm in the first layer (DCGAN-style)?

**A:** The first layer consumes raw structured inputs; BN can erase useful absolute intensity cues and adds batch-noise when batch size is small.

### D13: What was the hardest engineering problem?

**A:** A strong answer: alignment and data QC — misregistered pairs cap achievable quality regardless of model capacity; debugging “the model” before verifying pairs wastes time.

### D14: What is the biggest trade-off you made?

**A:** Emphasizing pixel accuracy (high λ_L1) vs pushing purely adversarial realism — appropriate for metrology-like translation.

### D15: If redesigning from scratch?

**A:** Domain-specific perceptual or feature metrics; richer monitoring (population drift); hybrid attention at multiple scales; active learning to spend human review budget efficiently.
