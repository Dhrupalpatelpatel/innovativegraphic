# CIS 5543 Computer Vision - Formula Cheat Sheet (Plain Readable Version)

This version avoids LaTeX blocks so it is readable in any editor/browser.

---

## 0) Notation

- Image: I in R^(H x W x C)
- Kernel size: K, stride: S, padding: P
- Feature map: F
- Logits: z, probabilities: p
- One-hot target: y

---

## 1) Image Filtering

### 1.1 2D Convolution

O(x,y) = sum_u sum_v K(u,v) * I(x-u, y-v)

Meaning: output pixel is a weighted sum of neighborhood pixels.

### 1.2 Output Size

H_out = floor((H + 2P - K) / S) + 1  
W_out = floor((W + 2P - K) / S) + 1

### 1.3 Linearity and Shift-Invariance

f(aI1 + bI2) = a f(I1) + b f(I2)  
f(T_delta(I)) = T_delta(f(I))

### 1.4 Gaussian Filter

G_sigma(x,y) = [1 / (2*pi*sigma^2)] * exp(-(x^2 + y^2) / (2*sigma^2))

I_smooth = I * G_sigma

Separable Gaussian:
G_sigma(x,y) = g_sigma(x) * g_sigma(y)

Cost drop (separable): O(HWk^2) -> O(HW * 2k)

### 1.5 Sharpening (Unsharp Mask)

I_sharp = I + alpha * (I - I_smooth)

If alpha = 1:
I_sharp = 2I - I_smooth

### 1.6 Gradient and Edge Strength

grad(I) = [Ix, Iy]^T  
|grad(I)| = sqrt(Ix^2 + Iy^2)  
theta = atan2(Iy, Ix)

### 1.7 Derivative of Gaussian

d/dx (G_sigma * I) = (dG_sigma/dx) * I  
d/dy (G_sigma * I) = (dG_sigma/dy) * I

### 1.8 Median Filter (Nonlinear)

O(i,j) = median{ I(m,n) for (m,n) in neighborhood N_ij }

Not linear:
f(I1 + I2) != f(I1) + f(I2)

### 1.9 Bilateral Filter

O(p) = (1 / Wp) * sum_{q in Omega}
       exp(-||p-q||^2 / (2*sigma_s^2)) *
       exp(-|I(p)-I(q)|^2 / (2*sigma_r^2)) *
       I(q)

Wp = sum_{q in Omega}
     exp(-||p-q||^2 / (2*sigma_s^2)) *
     exp(-|I(p)-I(q)|^2 / (2*sigma_r^2))

---

## 2) CNN Basics

### 2.1 Conv Layer Parameters

params = (k_h * k_w * C_in + 1) * C_out

### 2.2 Conv MACs (Approx)

MACs ~= H_out * W_out * k_h * k_w * C_in * C_out

### 2.3 ReLU

ReLU(x) = max(0, x)

### 2.4 Pooling Output Size

H_out = floor((H + 2P - K) / S) + 1  
W_out = floor((W + 2P - K) / S) + 1

### 2.5 Receptive Field Recurrence

r_l = r_(l-1) + (k_l - 1) * j_(l-1)  
j_l = j_(l-1) * s_l

For stride 1 and repeated 3x3 convs:
r = 1 + 2L

---

## 3) Vision Transformer (ViT)

### 3.1 Number of Patch Tokens

N = (H/P) * (W/P)

Example: 224x224 image, P=16 -> N = 14x14 = 196

### 3.2 Patch Embedding

x_p in R^(P^2 * C)  
z_p = x_p E + b  
E in R^((P^2 * C) x D)

### 3.3 ViT Input Sequence

Z0 = [z_cls ; z1 ; ... ; zN] + E_pos

### 3.4 Scaled Dot-Product Attention

Q = X W_Q, K = X W_K, V = X W_V

Attention(Q,K,V) = softmax((Q K^T) / sqrt(d_k)) V

### 3.5 Multi-Head Attention

MHA(X) = Concat(head_1, ..., head_h) W_O

head_i = Attention(X W_Q^(i), X W_K^(i), X W_V^(i))

### 3.6 Encoder Block (Pre-LN)

X' = X + MHA(LN(X))  
Y  = X' + MLP(LN(X'))

### 3.7 Attention Complexity

Cost ~ O(N^2 * D)

If patch size halves (P -> P/2): N -> 4N, attention cost -> about 16x.

---

## 4) Detection and Segmentation

### 4.1 IoU

IoU(A,B) = Area(A intersection B) / Area(A union B)

### 4.2 Precision / Recall / F1

Precision = TP / (TP + FP)  
Recall    = TP / (TP + FN)  
F1        = 2PR / (P + R)

### 4.3 AP and mAP

AP = integral_0^1 P(R) dR  
mAP = (1 / |C|) * sum_c AP_c

COCO:
mAP@[0.5:0.95] = (1 / (|C||T|)) * sum_c sum_t AP_(c,t), t in {0.50,...,0.95}

### 4.4 DETR Matching and Loss

sigma_hat = argmin_sigma sum_i L_match(y_i, yhat_sigma(i))

L_DETR = sum_i [ L_cls + lambda1 * ||b_i - bhat_sigma(i)||_1 + lambda_g * L_GIoU ]

L_GIoU = 1 - GIoU

### 4.5 Segmentation Losses

Pixel CE:
L_CE = -(1/(HW)) * sum_i sum_j sum_c y_ijc * log(p_ijc)

Dice:
Dice = (2*sum_i p_i g_i + eps) / (sum_i p_i + sum_i g_i + eps)

L_Dice = 1 - Dice

### 4.6 Panoptic Quality

PQ = [sum_(p,g in TP) IoU(p,g)] / [|TP| + 0.5|FP| + 0.5|FN|]

### 4.7 Bilinear Upsampling

f(x,y) = (1-x)(1-y)f00 + x(1-y)f10 + (1-x)y f01 + x y f11

---

## 5) Contrastive Models (CLIP / BLIP)

### 5.1 Cosine Similarity

sim(u,v) = (u^T v) / (||u||_2 ||v||_2)

### 5.2 InfoNCE Loss

L_i = -log( exp(sim(z_i, z_i+)/tau) / sum_{j=1..N} exp(sim(z_i, z_j)/tau) )

### 5.3 CLIP Symmetric Loss

L_CLIP = 0.5 * (L_img->text + L_text->img)

### 5.4 Zero-Shot Class Probability

p(y=j|x) = exp(sim(f(x), g(t_j))/tau) / sum_k exp(sim(f(x), g(t_k))/tau)

---

## 6) LLM Core

### 6.1 Autoregressive Factorization

p_theta(x_1:T) = product_{t=1..T} p_theta(x_t | x_<t)

### 6.2 Softmax Over Vocabulary

p_t(i) = exp(z_t,i) / sum_j exp(z_t,j)

### 6.3 Next-Token NLL / CE

L_NLL = -(1/T) * sum_{t=1..T} log p_theta(x_t | x_<t)

### 6.4 Perplexity

PPL = exp(L_NLL)

### 6.5 Temperature Sampling

p(i) proportional to exp(z_i / T)

Lower T: sharper output; Higher T: more random output.

---

## 7) Vision-Language Models

### 7.1 Cross-Attention Fusion

CrossAttn(Q_text, K_img, V_img) =
softmax((Q_text K_img^T) / sqrt(d_k)) V_img

### 7.2 Unified Token Space (LLaVA style)

v_tilde = W_proj v

Combined tokens:
[v_tilde_1, ..., v_tilde_m, t_1, ..., t_n]

All tokens go through one transformer.

---

## 8) Core Loss Functions

### 8.1 Multiclass Softmax + CE

p_i = e^(z_i) / sum_j e^(z_j)

L_CE = -sum_i y_i log p_i

For one-hot target:
L_CE = -log(p_correct)

### 8.2 Binary CE

L_BCE = -[ y log p + (1-y) log(1-p) ]

### 8.3 Focal Loss

L_focal = -alpha_t * (1-p_t)^gamma * log(p_t)

### 8.4 Triplet Loss

L_triplet = max(0, d(a,p) - d(a,n) + m)

### 8.5 L1 / L2 / Huber

L1: L = |y - yhat|  
L2: L = (y - yhat)^2

Huber:
if |e| < delta:      L = 0.5 * e^2  
else:                L = delta * (|e| - 0.5*delta)
where e = y - yhat

---

## 9) Explainability

### 9.1 Saliency

S = | d(y^c) / dX |

### 9.2 Grad-CAM

alpha_k^c = (1/Z) * sum_i sum_j d(y^c)/d(A_ij^k)

L_GradCAM^c = ReLU( sum_k alpha_k^c A^k )

---

## 10) Ultra-Important Exam Facts

1. Median filter is NOT linear.  
2. DETR uses bipartite matching, so no NMS.  
3. ViT classification uses [CLS] token.  
4. Attention cost is quadratic in token count N.  
5. Derivatives amplify noise, so smooth with Gaussian first.  
6. CLIP uses symmetric image-text contrastive loss.  
7. Perplexity = exp(average NLL).
