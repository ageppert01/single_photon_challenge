# Two-Stage Solution Proposal — Detailed Analysis

This document summarizes and analyzes the proposed two-stage pipeline for the Single Photon Challenge. Use it to align on design before deciding implementation order.

---

## High-level pipeline

```
1024 binary frames (800×800×3 each)
        ↓
   [ STAGE 1 ]  Alignment-based aggregation
        ↓
   Single intermediate image (strong starting point)
        ↓
   [ STAGE 2 ]  Generative reconstruction (diffusion / VAE / GAN)
        ↓
   Clean 800×800×3 submission image
```

**Design principle:** Stage 1 maximizes information preserved in a single image; Stage 2 refines that image into a clean, perceptually sharp result. Quality of Stage 1 directly limits what Stage 2 can achieve.

---

## Stage 1: Alignment-based aggregation

### Goal
Combine the 1,024 input frames into **one intermediate image** that is a strong starting point for Stage 2. This stage is critical: better starting point → easier and better refinement.

### Why not naive sum?
- **Naive sum:** Average (or sum) a subset of frames → more photons, better color, but **motion-induced blur**.
- Cause: scene not static — camera translation/zoom so the same pixel in different frames corresponds to different scene points → direct sum smears boundaries.
- Trade-off: more frames → less noise, more blur.

### Proposed approach (summary)

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| **Reference frame** | Last frame (index −1) | Matches competition causality: GT = last frame; reference = last frame keeps output in same “view” as evaluation. |
| **Processing order** | Reverse (from last backward) | Adjacent frames similar → alignment for frame *t* is good init for frame *t−1* → stable, efficient. |
| **Alignment** | Per-frame (or per batch): estimate **translation** and **scale** to align to reference | Scene changes driven by camera translation + zoom (from sample data); 2 DoF (+ scale) is a reasonable first model. |
| **Stopping** | Stop when overlap with reference becomes “minimal” | Avoid including frames that no longer share content with reference; prevents degradation from non-overlapping data. |
| **Output** | Sum (or average) of the **aligned subset** only | Retain as much photon count as possible while reducing blur. |

### Relation to prior work
- **Quanta Burst Photography (Ma et al., 2020)** [3]: Also stresses **motion estimation + compensation before merging** for binary single-photon sequences under motion. Your pipeline is conceptually aligned; differences to explore: motion model (e.g. homography vs translation+scale), merging strategy, handling of occlusion/disocclusion.

### Implementation questions to pin down later
- Motion model: translation only, or translation + scale (and possibly rotation)?
- Alignment method: classical (e.g. phase correlation, optical flow, feature-based) vs learned?
- Overlap criterion: threshold on alignment residual, correlation, or geometric overlap?
- Batching: align and merge in batches (e.g. 32 frames) for speed vs per-frame.
- Coordinate system: output in “reference” (last frame) coordinates → 800×800×3, compatible with GT and eval.

---

## Stage 2: Generative reconstruction

### Goal
Map the Stage 1 aggregated image → **clean, high-fidelity 800×800×3** image. During Stage 1 development, **naive sum** can be used as a stand-in for Stage 2 so the two stages can be developed and evaluated somewhat independently.

### Why generative?
- Classical (e.g. QBP) and regression (e.g. QUIVER) tend to **over-smooth** and lose high-frequency detail when optimizing pixel-level loss and averaging over uncertainty.
- Generative models can represent **distributions** over plausible clean images → potential to recover detail and texture that regression washes out.

### Three candidate approaches

| Approach | Role in your plan | Pros | Cons |
|----------|-------------------|------|------|
| **Diffusion** | Primary (dominant) | Models distribution over clean images; good fit for heavy noise + motion; U-Net backbone; can condition on binary frames (e.g. RNN over time). Novel for quanta burst reconstruction; QuDi [2] gives video-reconstruction precedent. | Slower inference (multi-step); need to choose conditioning (Stage 1 only vs Stage 1 + temporal features). |
| **VAE** | Baseline / hybrid | Stable training; smooth, globally coherent; efficient; possible preconditioner for diffusion. | Tends to over-smooth; Gaussian prior + pixel loss → weaker fine texture/edges. |
| **GAN** | Baseline | Single-pass, fast inference; adversarial loss → perceptual realism, sharp edges/textures. | Instability, mode collapse; risk of hallucination under extreme noise; need careful loss balance. |

### Diffusion design notes (from proposal)
- **Input:** Stage 1 aggregated image as structured initialization; optionally **conditioning** on features from the 1,024 binary frames (e.g. RNN over frames → hidden state used in denoising).
- **Process:** Iterative denoising (reverse noising); predict noise residuals; U-Net backbone.
- **Reference:** QuDi [2] for video reconstruction; re-implement and adapt ideas.

### Stage 2 implementation order (for discussion)
- Start with **naive sum** as Stage 2 placeholder to validate pipeline and metrics.
- Then add one generative method: diffusion (main) first, or VAE/GAN as a faster baseline.
- Conditioning (binary temporal stream vs Stage 1 only) can be added after a working diffusion model.

---

## Fit with competition

| Requirement | How the proposal fits |
|-------------|------------------------|
| **Input** | 1024 frames, 800×800×3 binary; causal (GT = last frame). | Stage 1 uses last frame as reference; can use all or a subset of frames; Stage 2 consumes one 800×800×3 image. |
| **Output** | 800×800×3 PNG per test sample. | Stage 2 output is 800×800×3; save as PNG; zip as `<scene>/<id>.png`. |
| **Metrics** | PSNR, MS-SSIM, LPIPS (mean + 5%, 1% quantiles). | LPIPS favors perceptually sharp results (generative Stage 2); PSNR/MS-SSIM favor fidelity (good Stage 1 + refinement). |
| **Train/val** | 50-scene train, 5-scene test (no public GT). | Stage 1: can train/validate alignment on train set (GT = last frame). Stage 2: train on (Stage1_output, GT) pairs; validate on hold-out scenes; test = inference only. |

---

## References (as in proposal)

- **[3] Quanta Burst Photography (Ma et al., 2020)** — Motion estimation and compensation before merging for high-speed binary single-photon sequences.
- **[2] QuDi** — Video reconstruction with diffusion; re-implement and take inspiration for conditioning and architecture.

*(Exact citations can be added when you have papers.)*

---

## Suggested discussion points for “what to implement first”

1. **Stage 1 first vs end-to-end baseline**
   - Option A: Implement Stage 1 (alignment + selective sum), keep Stage 2 = naive sum; measure PSNR/MS-SSIM/LPIPS vs naive sum only.
   - Option B: Implement full pipeline with naive sum for both stages, then swap in alignment-based Stage 1.

2. **Stage 1 scope for “beginning”**
   - Translation-only vs translation+scale.
   - Classical alignment (e.g. OpenCV / phase correlation) vs learned.
   - Single scene from sample dataset to debug load/unpack/alignment/sum.

3. **Stage 2 scope for “beginning”**
   - Naive sum only as placeholder, or
   - One simple generative baseline (e.g. small VAE or GAN) to validate training loop and `eval_single.py` on (GT, pred).

4. **Data**
   - Start with sample dataset (~3.5 GB) for development; move to full train/test when scaling.

Once you decide what you want to implement first (e.g. “Stage 1 alignment on sample data only” or “naive pipeline + eval, then add alignment”), we can break it into concrete tasks and file/script layout.
