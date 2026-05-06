# Stage 1: RNN-Based Temporal Integration — Design & Discussion

**Scope:** Stage 1 only. Implement and validate an RNN (e.g. ConvLSTM / Bi-ConvLSTM) that consumes the 1024 binary frames and outputs a **high-dimensional feature map** (final hidden state) to be used as a prior in Stage 2.

---

## 1. Your approach (summary)

| Step | Description |
|------|-------------|
| **Input** | 1024 binary frames, each 800×800×3 (or downsampled for the RNN). |
| **Model** | RNN over time — e.g. **ConvLSTM** or **Bidirectional ConvLSTM** (temporal + motion adaptive, better than a fixed sliding window). |
| **Processing** | Process frames 0 → 1022 **without saving** intermediate outputs; only update hidden state. |
| **Final step** | After passing the **last frame (1023)**, take the **final hidden state** as the Stage 1 output. |
| **Output** | **High-dimensional feature map** (not 3-channel yet): a learned “summary” of every photon seen (edges, brightness, motion, etc.). |
| **Stage 2 use** | Stage 2 takes this feature map and shrinks it to RGB (e.g. 1×1 or small conv to 3 channels) and refines into the final clean image. So Stage 1 output is a **prior** for Stage 2. |
| **Training detail** | Use `return_sequences=True` if you need per-timestep outputs for auxiliary losses; for the **summary** we only need the **last** hidden state (e.g. `return_sequences=False` or take the last step of `return_sequences=True`). |

---

## 2. Why this can be powerful for the competition

- **Learned temporal integration:** The model can learn *how* to merge 1024 noisy binary frames instead of hand-designed alignment + sum. It can adapt to motion, occlusion, and varying SNR across time.
- **Single rich prior for Stage 2:** One feature map that encodes structure, edges, and motion is exactly what a diffusion (or VAE/GAN) model can condition on. Better prior → better and more stable refinement.
- **Causal and consistent:** Processing 0 → 1023 and taking the state after the last frame matches the competition’s causal setup (GT = last frame). No “future” leakage if we use unidirectional RNN.
- **End-to-end trainable later:** Once Stage 2 is in place, you can backprop through the Stage 1 RNN from the final reconstruction loss (e.g. LPIPS + PSNR), so the “summary” can optimize for what Stage 2 needs.
- **Different from naive sum:** Naive sum has no notion of motion or structure; the RNN can, in principle, learn to emphasize consistent structure and suppress motion blur, which should help both pixel metrics (PSNR, MS-SSIM) and perceptual quality (LPIPS).

**Usefulness for winning:** Stage 1 is the bottleneck for information. If the RNN gives a sharper, more structured prior than naive sum (validated by your metrics below), Stage 2 has a much easier job. That combination (learned temporal prior + strong generative Stage 2) is a plausible path to top leaderboard.

---

## 3. Challenges and mitigations

| Challenge | Mitigation |
|-----------|------------|
| **Long sequence (1024)** | Gradient issues over 1024 steps. Use gradient clipping, possibly **chunked processing**: e.g. 64 frames per chunk, pass hidden state between chunks; effective length 16. Or 2-layer ConvLSTM with residual connections. |
| **Memory** | 1024 × 800×800×3 at full res is huge. Run ConvLSTM at **reduced resolution** (e.g. 100×100 or 200×200) via strided conv or pooling before the RNN; then the feature map is low-res. Stage 2 can take this + full-res naive sum, or upsample and refine. |
| **Binary, very noisy input** | Each frame is 0/1. Train Stage 1 with a **supervision signal**: e.g. small decoder head (conv layers) from final hidden state → 3-channel image, supervise to GT. That way the hidden state is explicitly trained to carry reconstructible information. |
| **Unidirectional vs bidirectional** | **Unidirectional** = strictly causal, matches “only past frames.” **Bidirectional** = at test time we have the full burst, so we can use both directions; the “final” summary is then a function of the whole sequence and can be richer. Recommendation: start **unidirectional** (causal); try **bidirectional** as an ablation (allowed since we have all 1024 at test time). |
| **What is “final hidden state”?** | In ConvLSTM, the **cell state** and **hidden state** at the last timestep are the natural summary. Usually we take the **hidden state** (output at last step); optionally concatenate cell state for more capacity. Output shape: e.g. `(B, C_hidden, H, W)` — this is the high-dim feature map. |

---

## 4. Validation before Stage 2 (your plan, refined)

**A. Visual inspection of the prior**

- **Visualize the final hidden state** to check that photons are integrated into coherent structure:
  - Reduce channels to 3 for display: e.g. 1×1 conv (C_hidden → 3) or PCA on the channel dimension, or show a few slices of channels as RGB.
  - **Success criterion:** Recognizable objects, sharp-ish boundaries, no obvious garbage. If it looks like a very noisy/blurry but coherent image, the RNN is doing something useful.

**B. Quantitative baseline: “Stage 1 as image” vs naive sum vs GT**

- Add a **small decoder head** (e.g. a few conv layers) that maps the final hidden state → 3-channel image (same resolution as the feature map, e.g. 100×100 or 200×200).
- **Upsample** that to 800×800 (e.g. bilinear or learned) to match GT resolution.
- Train **Stage 1 only** with loss = L1 or L2 (or L1 + perceptual) to GT.
- Then compare:
  - **Naive sum** (last 1024 frames averaged, optional tonemap) vs **GT** (PSNR, MS-SSIM, LPIPS).
  - **RNN output (decoder head + upsample)** vs **GT** (same metrics).
- **Success criteria:** RNN image is **better than naive sum** on at least one of PSNR / MS-SSIM / LPIPS, and **closer to GT** overall. That would show the RNN prior is a better starting point for Stage 2.

**C. Optional: Ablations**

- Unidirectional vs bidirectional.
- Chunk size (e.g. 64 vs 128 vs 256 frames per chunk).
- Resolution of the RNN (e.g. 100×100 vs 200×200).

---

## 5. Implementation recommendations (same lines, more concrete)

- **Backbone:** **ConvLSTM** is a good default. **Bidirectional ConvLSTM** is worth trying for a richer summary (both past and future context); ensure you take the combined final state (e.g. concat forward + backward hidden at last step).
- **return_sequences:** Use `return_sequences=False` if the implementation only needs the last output; that’s the “final hidden state.” If you add an auxiliary loss (e.g. predict next frame from hidden state), use `return_sequences=True` and take the last timestep for the main summary; avoid saving all 1024 outputs unless necessary (memory).
- **Chunked processing:** To save memory and stabilize gradients, process **chunks of frames** (e.g. 64), pass hidden/cell state to the next chunk, and use the **final** hidden state after the last chunk as the global summary. Same idea as “update till 1022 then use state after 1023,” but with a shorter effective sequence (e.g. 16 chunks of 64).
- **Resolution:** Run the RNN at **lower spatial resolution** (e.g. 200×200 or 100×100). Input: downsample each binary frame (e.g. average pooling or strided conv). Output: feature map at that resolution. For validation (B above), upsample the decoded image to 800×800 for comparison to GT; for Stage 2, either feed low-res feature map and fuse with full-res input, or upsample the feature map and refine at 800×800.
- **Training Stage 1 alone:** Minimize L1 (or L2) to GT; optionally add a **perceptual loss** (e.g. VGG or LPIPS) on the decoded image so the prior is tuned for perceptual quality. That should give a prior that is both pixel-accurate and structurally rich.
- **Stage 2 interface:** Stage 2 receives the **feature map** `(C_hidden, H, W)` (and optionally the naive-sum image). It can “shrink” channels to 3 with a conv and then refine (diffusion/VAE/GAN). So your picture — “high-dim feature map as prior, Stage 2 shrinks to RGB and refines” — is the right interface.

---

## 6. Summary

| Question | Answer |
|----------|--------|
| **Is the RNN approach useful for winning?** | Yes. A learned temporal prior that beats naive sum and is closer to GT gives Stage 2 a strong starting point; combined with a strong generative Stage 2, this is a competitive design. |
| **Power of the idea** | Learns *how* to integrate binary frames in a motion-adaptive way; outputs a structured, high-dim prior instead of a single blurry image; directly usable as conditioning for Stage 2. |
| **Validation before Stage 2** | (1) Visualize final hidden state (project to 3 ch or show channel slices). (2) Decoder head → image, upsample to 800×800, compare vs naive sum and vs GT with PSNR/MS-SSIM/LPIPS. |
| **Recommendations** | ConvLSTM or Bi-ConvLSTM; chunked processing (e.g. 64 frames); reduced spatial resolution; train Stage 1 with decoder head to GT; optional perceptual loss; then feed feature map to Stage 2. |

Next step: implement Stage 1 RNN (data load, ConvLSTM, final hidden state, decoder head for validation, metrics vs naive sum and GT) and run on sample data.
