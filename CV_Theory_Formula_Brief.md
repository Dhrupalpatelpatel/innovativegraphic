# CIS 5543 Computer Vision - Theory + Formula Brief

This sheet is made for fast revision: **what it means**, **why it matters**, and **the core formula**.

---

## 1) Image Filtering

### Theory (brief)
- Filtering applies a local operation on each pixel using neighboring pixels.
- Linear filters (box, Gaussian) are good for smoothing noise.
- Nonlinear filters (median, bilateral) are robust to outliers and preserve edges better.

### Key formulas
- Convolution:  
  `O(x,y) = sum_u sum_v K(u,v) * I(x-u, y-v)`
- Output size:  
  `H_out = floor((H + 2P - K)/S) + 1`  
  `W_out = floor((W + 2P - K)/S) + 1`
- Gaussian:  
  `G_sigma(x,y) = [1/(2*pi*sigma^2)] * exp(-(x^2+y^2)/(2*sigma^2))`
- Sharpening (unsharp):  
  `I_sharp = I + alpha*(I - I_smooth)`
- Median filter (nonlinear):  
  `O(i,j) = median{I(m,n) in neighborhood}`

### Memory line
**Linear smooths, median resists outliers, bilateral smooths without destroying edges.**

---

## 2) Gradients and Edges

### Theory (brief)
- Edges are locations of rapid intensity change.
- Derivatives detect edges but are noise-sensitive.
- Best practice: smooth first (Gaussian), then differentiate.

### Key formulas
- Gradient vector:  
  `grad(I) = [Ix, Iy]^T`
- Edge strength:  
  `|grad(I)| = sqrt(Ix^2 + Iy^2)`
- Edge direction:  
  `theta = atan2(Iy, Ix)`
- Derivative of Gaussian:  
  `d/dx (G*I) = (dG/dx) * I`

### Memory line
**Edge = high gradient; noisy image = smooth before gradient.**

---

## 3) CNN Basics

### Theory (brief)
- CNNs use local connectivity + weight sharing, so they are efficient for images.
- ReLU adds nonlinearity; pooling downsamples while keeping strong activations.
- Deep CNNs learn hierarchy: edges -> textures -> parts -> objects.

### Key formulas
- Conv params:  
  `params = (k_h*k_w*C_in + 1) * C_out`
- Conv MACs (approx):  
  `MACs ~= H_out * W_out * k_h * k_w * C_in * C_out`
- ReLU:  
  `ReLU(x) = max(0, x)`
- Receptive field recurrence:  
  `r_l = r_(l-1) + (k_l - 1)*j_(l-1)`  
  `j_l = j_(l-1)*s_l`

### Memory line
**CNN = shared local filters + nonlinear stacking for visual hierarchy.**

---

## 4) Vision Transformer (ViT)

### Theory (brief)
- ViT treats image patches like tokens in NLP.
- Self-attention lets each patch look at all patches (global context).
- ViT is powerful at scale but attention cost grows quickly with token count.

### Key formulas
- Number of tokens:  
  `N = (H/P) * (W/P)`
- Patch embedding:  
  `z_p = x_p E + b`
- Attention:  
  `Attention(Q,K,V) = softmax((QK^T)/sqrt(d_k)) V`
- Multi-head:  
  `MHA(X) = Concat(head_1,...,head_h) W_O`
- Encoder block (Pre-LN):  
  `X' = X + MHA(LN(X))`  
  `Y  = X' + MLP(LN(X'))`
- Complexity:  
  `Cost ~ O(N^2 * D)`

### Memory line
**ViT = patch tokens + attention; smaller patches mean much higher compute.**

---

## 5) Detection and Segmentation

### Theory (brief)
- Detection predicts object class + bounding box.
- Segmentation predicts labels/masks at pixel level.
- Evaluation balances overlap quality and classification quality.

### Key formulas
- IoU:  
  `IoU = Area(intersection) / Area(union)`
- Precision/Recall/F1:  
  `Precision = TP/(TP+FP)`  
  `Recall = TP/(TP+FN)`  
  `F1 = 2PR/(P+R)`
- AP and mAP:  
  `AP = integral_0^1 P(R) dR`  
  `mAP = average(AP over classes)`
- Dice:  
  `Dice = (2*sum p_i g_i + eps) / (sum p_i + sum g_i + eps)`  
  `L_Dice = 1 - Dice`
- DETR (set prediction):  
  `L_DETR = L_cls + lambda1*L1_box + lambda_g*L_GIoU`

### Memory line
**IoU checks overlap; mAP summarizes detector quality; Dice is segmentation-friendly overlap.**

---

## 6) Contrastive Learning (CLIP/BLIP)

### Theory (brief)
- Contrastive learning aligns related pairs and separates unrelated pairs.
- CLIP learns shared image-text embedding space.
- This enables zero-shot classification and retrieval.

### Key formulas
- Cosine similarity:  
  `sim(u,v) = (u^T v) / (||u|| ||v||)`
- InfoNCE:  
  `L_i = -log( exp(sim(z_i,z_i+)/tau) / sum_j exp(sim(z_i,z_j)/tau) )`
- CLIP symmetric loss:  
  `L_CLIP = 0.5 * (L_img->text + L_text->img)`

### Memory line
**Contrastive learning pulls positives together and pushes negatives apart.**

---

## 7) LLM Core

### Theory (brief)
- LLMs model text as next-token prediction.
- Training uses teacher forcing and cross-entropy.
- Inference samples tokens repeatedly from model probabilities.

### Key formulas
- Autoregressive factorization:  
  `p(x_1:T) = product_t p(x_t | x_<t)`
- Softmax:  
  `p_t(i) = exp(z_t,i) / sum_j exp(z_t,j)`
- NLL/CE:  
  `L_NLL = -(1/T) * sum_t log p(x_t | x_<t)`
- Perplexity:  
  `PPL = exp(L_NLL)`

### Memory line
**LLM training minimizes next-token surprise; lower perplexity means better language modeling.**

---

## 8) Vision-Language Models (VLMs)

### Theory (brief)
- VLMs combine image understanding and language reasoning.
- Two common designs:
  1) Cross-attention between separate vision and text streams.
  2) Unified token space where visual tokens join text tokens.

### Key formulas
- Cross-attention:  
  `CrossAttn(Q_text, K_img, V_img) = softmax((Q_text K_img^T)/sqrt(d_k)) V_img`
- Unified space projection:  
  `v_tilde = W_proj * v`

### Memory line
**VLM = vision features + language model, fused by attention or shared token space.**

---

## 9) Core Loss Functions (high-yield)

### Theory (brief)
- Choose loss based on task type: classification, regression, detection, metric learning.

### Key formulas
- Multiclass CE:  
  `L_CE = -sum_i y_i log p_i`  
  (one-hot: `L = -log p_correct`)
- Binary CE:  
  `L_BCE = -[y log p + (1-y) log(1-p)]`
- Focal loss:  
  `L_focal = -alpha_t * (1-p_t)^gamma * log(p_t)`
- Triplet loss:  
  `L_triplet = max(0, d(a,p) - d(a,n) + m)`
- Huber loss:  
  `if |e|<delta: 0.5e^2 else: delta(|e|-0.5delta)`

### Memory line
**Loss function defines what "good prediction" means for each task.**

---

## 10) Explainability

### Theory (brief)
- Explainability shows which input regions/tokens drove predictions.
- Helps debugging, trust, and failure analysis.

### Key formulas
- Saliency:  
  `S = | d(y^c) / dX |`
- Grad-CAM:  
  `alpha_k^c = (1/Z) * sum_i sum_j d(y^c)/d(A_ij^k)`  
  `L_GradCAM^c = ReLU(sum_k alpha_k^c A^k)`

### Memory line
**Saliency/Grad-CAM answer: "where did the model look before deciding?"**

---

## Final Rapid Revision (10 lines)

1. Convolution = weighted local sum.  
2. Gaussian smooths noise; median removes impulse noise.  
3. Gradients detect edges; smooth before derivatives.  
4. CNNs use local shared filters and build feature hierarchies.  
5. ViT uses patch tokens and global self-attention.  
6. Attention formula: softmax(QK^T/sqrt(d))V.  
7. Detection quality uses IoU/mAP; segmentation uses Dice/mIoU.  
8. CLIP aligns image-text with contrastive loss.  
9. LLMs model next-token probabilities; PPL = exp(NLL).  
10. VLMs fuse visual and language streams for multimodal reasoning.
