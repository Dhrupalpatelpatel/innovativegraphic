# CIS 5543 Computer Vision - Formula Cheat Sheet

Clean, exam-ready formulas with consistent notation and readable formatting.

---

## 0) Notation

- Image: \(I \in \mathbb{R}^{H \times W \times C}\)
- Kernel size: \(K\), stride: \(S\), padding: \(P\)
- Feature map: \(F\)
- Logits: \(z\), probabilities: \(p\)
- One-hot target label: \(y\)

---

## 1) Image Filtering

### 1.1 2D Convolution

\[
O(x,y)=\sum_{u}\sum_{v} K(u,v)\,I(x-u,y-v)
\]

Meaning: output pixel equals weighted sum of neighborhood pixels.

### 1.2 Output Size

\[
H_{\text{out}}=\left\lfloor \frac{H+2P-K}{S} \right\rfloor+1,\quad
W_{\text{out}}=\left\lfloor \frac{W+2P-K}{S} \right\rfloor+1
\]

### 1.3 Linearity and Shift-Invariance

\[
f(aI_1+bI_2)=a f(I_1)+b f(I_2)
\]
\[
f(T_{\Delta}I)=T_{\Delta}f(I)
\]

### 1.4 Gaussian Filter

\[
G_{\sigma}(x,y)=\frac{1}{2\pi\sigma^2}\exp\!\left(-\frac{x^2+y^2}{2\sigma^2}\right)
\]
\[
I_{\text{smooth}}=I*G_{\sigma}
\]
\[
G_{\sigma}(x,y)=g_{\sigma}(x)\,g_{\sigma}(y)
\]

Separable implementation reduces cost from \(O(HWk^2)\) to \(O(HW \cdot 2k)\).

### 1.5 Sharpening (Unsharp Mask)

\[
I_{\text{sharp}}=I+\alpha\left(I-I_{\text{smooth}}\right)
\]

If \(\alpha=1\):

\[
I_{\text{sharp}}=2I-I_{\text{smooth}}
\]

### 1.6 Gradient and Edge Strength

\[
\nabla I=\begin{bmatrix}I_x\\I_y\end{bmatrix},\quad
\|\nabla I\|=\sqrt{I_x^2+I_y^2},\quad
\theta=\operatorname{atan2}(I_y,I_x)
\]

### 1.7 Derivative of Gaussian

\[
\frac{\partial}{\partial x}(G_{\sigma}*I)=\left(\frac{\partial G_{\sigma}}{\partial x}\right)*I,\quad
\frac{\partial}{\partial y}(G_{\sigma}*I)=\left(\frac{\partial G_{\sigma}}{\partial y}\right)*I
\]

### 1.8 Median Filter (Nonlinear)

\[
O(i,j)=\operatorname{median}\{I(m,n):(m,n)\in\mathcal{N}_{ij}\}
\]

Not linear:

\[
f(I_1+I_2)\neq f(I_1)+f(I_2)
\]

### 1.9 Bilateral Filter

\[
O(p)=\frac{1}{W_p}\sum_{q\in\Omega}
\exp\!\left(-\frac{\|p-q\|^2}{2\sigma_s^2}\right)
\exp\!\left(-\frac{|I(p)-I(q)|^2}{2\sigma_r^2}\right)I(q)
\]
\[
W_p=\sum_{q\in\Omega}
\exp\!\left(-\frac{\|p-q\|^2}{2\sigma_s^2}\right)
\exp\!\left(-\frac{|I(p)-I(q)|^2}{2\sigma_r^2}\right)
\]

---

## 2) CNN Basics

### 2.1 Conv Layer Parameters

\[
\#\text{params}=(k_hk_wC_{\text{in}}+1)\,C_{\text{out}}
\]

### 2.2 Conv MACs (Approximate)

\[
\text{MACs}\approx H_{\text{out}}W_{\text{out}}\,k_hk_w\,C_{\text{in}}C_{\text{out}}
\]

### 2.3 ReLU

\[
\operatorname{ReLU}(x)=\max(0,x)
\]

### 2.4 Pooling Output Size

\[
H_{\text{out}}=\left\lfloor\frac{H+2P-K}{S}\right\rfloor+1,\quad
W_{\text{out}}=\left\lfloor\frac{W+2P-K}{S}\right\rfloor+1
\]

### 2.5 Receptive Field Recurrence

\[
r_l=r_{l-1}+(k_l-1)j_{l-1},\quad j_l=j_{l-1}s_l
\]

For stride 1 and repeated \(3 \times 3\) convs:

\[
r=1+2L
\]

---

## 3) Vision Transformer (ViT)

### 3.1 Number of Patch Tokens

\[
N=\frac{H}{P}\cdot\frac{W}{P}
\]

Example: \(224 \times 224\), \(P=16 \Rightarrow N=14 \times 14=196\).

### 3.2 Patch Embedding

\[
x_p\in\mathbb{R}^{P^2C},\quad z_p=x_pE+b,\quad E\in\mathbb{R}^{(P^2C)\times D}
\]

### 3.3 ViT Input Sequence

\[
Z_0=[z_{\text{cls}};z_1;\dots;z_N]+E_{\text{pos}}
\]

### 3.4 Scaled Dot-Product Attention

\[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
\]
\[
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V
\]

### 3.5 Multi-Head Attention

\[
\operatorname{MHA}(X)=\operatorname{Concat}(\text{head}_1,\dots,\text{head}_h)W_O
\]
\[
\text{head}_i=\operatorname{Attention}(XW_Q^{(i)},XW_K^{(i)},XW_V^{(i)})
\]

### 3.6 Encoder Block (Pre-LN)

\[
X' = X + \operatorname{MHA}(\operatorname{LN}(X))
\]
\[
Y = X' + \operatorname{MLP}(\operatorname{LN}(X'))
\]

### 3.7 Attention Complexity

\[
\text{Cost}\sim O(N^2D)
\]

If \(P \to P/2\), then \(N\to 4N\), so attention cost is about \(16\times\) larger.

---

## 4) Detection and Segmentation

### 4.1 Intersection over Union (IoU)

\[
\operatorname{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}
\]

### 4.2 Precision, Recall, F1

\[
\text{Precision}=\frac{TP}{TP+FP},\quad
\text{Recall}=\frac{TP}{TP+FN}
\]
\[
F1=\frac{2PR}{P+R}
\]

### 4.3 AP and mAP

\[
AP=\int_0^1 P(R)\,dR,\quad
mAP=\frac{1}{|C|}\sum_{c\in C}AP_c
\]

COCO:

\[
mAP@[0.5:0.95]=\frac{1}{|C||T|}\sum_{c}\sum_{t\in\{0.50,\dots,0.95\}} AP_{c,t}
\]

### 4.4 DETR Matching and Loss

\[
\hat{\sigma}=\arg\min_{\sigma}\sum_i \mathcal{L}_{\text{match}}(y_i,\hat{y}_{\sigma(i)})
\]
\[
\mathcal{L}_{\text{DETR}}=\sum_i\left(
\mathcal{L}_{\text{cls}}+
\lambda_1\|b_i-\hat b_{\sigma(i)}\|_1+
\lambda_g\,\mathcal{L}_{\text{GIoU}}
\right)
\]
\[
\mathcal{L}_{\text{GIoU}}=1-\operatorname{GIoU}
\]

### 4.5 Segmentation Losses

Pixel-wise cross-entropy:

\[
\mathcal{L}_{\text{CE}}=-\frac{1}{HW}\sum_{i=1}^{H}\sum_{j=1}^{W}\sum_{c=1}^{C} y_{ijc}\log p_{ijc}
\]

Dice:

\[
\operatorname{Dice}=\frac{2\sum_i p_i g_i+\epsilon}{\sum_i p_i+\sum_i g_i+\epsilon},\quad
\mathcal{L}_{\text{Dice}}=1-\operatorname{Dice}
\]

### 4.6 Panoptic Quality

\[
PQ=\frac{\sum_{(p,g)\in TP}\operatorname{IoU}(p,g)}
{|TP|+\frac{1}{2}|FP|+\frac{1}{2}|FN|}
\]

### 4.7 Bilinear Upsampling

\[
f(x,y)=(1-x)(1-y)f_{00}+x(1-y)f_{10}+(1-x)yf_{01}+xyf_{11}
\]

---

## 5) Contrastive Models (CLIP / BLIP)

### 5.1 Cosine Similarity

\[
\operatorname{sim}(u,v)=\frac{u^{\top}v}{\|u\|_2\|v\|_2}
\]

### 5.2 InfoNCE Loss

\[
\mathcal{L}_i=-\log\frac{\exp(\operatorname{sim}(z_i,z_i^+)/\tau)}
{\sum_{j=1}^{N}\exp(\operatorname{sim}(z_i,z_j)/\tau)}
\]

### 5.3 CLIP Symmetric Loss

\[
\mathcal{L}_{\text{CLIP}}=\frac{1}{2}
\left(\mathcal{L}_{\text{img}\to\text{text}}+\mathcal{L}_{\text{text}\to\text{img}}\right)
\]

### 5.4 Zero-Shot Class Probability

\[
p(y=j\mid x)=
\frac{\exp(\operatorname{sim}(f(x),g(t_j))/\tau)}
{\sum_k \exp(\operatorname{sim}(f(x),g(t_k))/\tau)}
\]

---

## 6) LLM Core

### 6.1 Autoregressive Factorization

\[
p_{\theta}(x_{1:T})=\prod_{t=1}^{T} p_{\theta}(x_t\mid x_{<t})
\]

### 6.2 Softmax Over Vocabulary

\[
p_t(i)=\frac{\exp(z_{t,i})}{\sum_j \exp(z_{t,j})}
\]

### 6.3 Next-Token NLL / Cross-Entropy

\[
\mathcal{L}_{\text{NLL}}=-\frac{1}{T}\sum_{t=1}^{T}\log p_{\theta}(x_t\mid x_{<t})
\]

### 6.4 Perplexity

\[
\operatorname{PPL}=\exp(\mathcal{L}_{\text{NLL}})
\]

### 6.5 Temperature Sampling

\[
p(i)\propto \exp\!\left(\frac{z_i}{T}\right)
\]

Lower \(T\): sharper distribution. Higher \(T\): more randomness.

---

## 7) Vision-Language Models

### 7.1 Cross-Attention Fusion

\[
\operatorname{CrossAttn}(Q_{\text{text}},K_{\text{img}},V_{\text{img}})
=\operatorname{softmax}\!\left(\frac{Q_{\text{text}}K_{\text{img}}^{\top}}{\sqrt{d_k}}\right)V_{\text{img}}
\]

### 7.2 Unified Token Space (LLaVA-style)

\[
\tilde v = W_{\text{proj}}v,\quad
[\tilde v_1,\dots,\tilde v_m,t_1,\dots,t_n]
\]

All tokens are processed by a single transformer.

---

## 8) Core Loss Functions

### 8.1 Multiclass Softmax + Cross-Entropy

\[
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}},\quad
\mathcal{L}_{\text{CE}}=-\sum_i y_i\log p_i
\]

For one-hot target:

\[
\mathcal{L}_{\text{CE}}=-\log p_{\text{correct}}
\]

### 8.2 Binary Cross-Entropy

\[
\mathcal{L}_{\text{BCE}}= -\left[y\log p+(1-y)\log(1-p)\right]
\]

### 8.3 Focal Loss

\[
\mathcal{L}_{\text{focal}}=-\alpha_t(1-p_t)^{\gamma}\log(p_t)
\]

### 8.4 Triplet Loss

\[
\mathcal{L}_{\text{triplet}}=\max\left(0,\ d(a,p)-d(a,n)+m\right)
\]

### 8.5 L1, L2, Huber

\[
\mathcal{L}_{L1}=|y-\hat y|,\quad
\mathcal{L}_{L2}=(y-\hat y)^2
\]

\[
\mathcal{L}_{\text{Huber}}=
\begin{cases}
\frac{1}{2}e^2, & |e|<\delta \\
\delta\left(|e|-\frac{1}{2}\delta\right), & |e|\ge\delta
\end{cases}
\quad\text{where } e=y-\hat y
\]

---

## 9) Explainability

### 9.1 Saliency

\[
S=\left|\frac{\partial y^c}{\partial X}\right|
\]

### 9.2 Grad-CAM

\[
\alpha_k^c=\frac{1}{Z}\sum_i\sum_j \frac{\partial y^c}{\partial A_{ij}^k}
\]
\[
L_{\text{Grad-CAM}}^c=\operatorname{ReLU}\!\left(\sum_k \alpha_k^c A^k\right)
\]

---

## 10) Ultra-Important Exam Facts

1. Median filter is **not linear**.  
2. DETR uses bipartite matching, so **no NMS**.  
3. ViT classification uses the **[CLS] token**.  
4. Attention cost is quadratic in token count \(N\).  
5. Derivatives amplify noise, so smooth with Gaussian first.  
6. CLIP uses symmetric image-text contrastive loss.  
7. Perplexity is \(\exp(\text{average NLL})\).
