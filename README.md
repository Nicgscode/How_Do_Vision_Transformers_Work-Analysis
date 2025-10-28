# How Do Vision Transformers Work? - Analysis
**DS 5690 Paper Presntation**

*Authors: Namuk Park & Songkuk Kim (Yonsei University, NAVER AI Lab)*  
*Published at ICLR 2022*

**Full Citation:**  
Park, N., & Kim, S. (2022). How Do Vision Transformers Work?. In International Conference on Learning Representations (ICLR 2022). arXiv:2202.06709v4 [cs.CV]
___
## Overview - Five-minute overview providing context, stating the problem the paper is addressing, characterizing the approach, and giving a brief account of how the problem was addressed
**The abstract says it all**: 

"The success of multi-head self-attentions (MSAs) for computer vision is now indisputable. However, little is known about how MSAs work. We present fundamental explanations to help better understand the nature of MSAs.";

"MSAs are low-pass filters, but Convs are high-pass filters. Therefore, MSAs and Convs are complementary.";

"We propose AlterNet, a model in which Convblocks at the end of a stage are replaced with MSA blocks."


# **Context**:
Following the success of the original Vision Transformer (ViT) introduced by [Dosovitskiy et al.](https://arxiv.org/pdf/2010.11929) in 2021, Multi-head Self-Attention (MSA) mechanisms have become ubiquitous in computer vision. By 2022, numerous variants—including [Swin Transformer](https://arxiv.org/pdf/2103.14030) and [PiT](https://arxiv.org/pdf/2103.16302) demonstrated that MSAs could match or exceed the performance of traditional Convolutional Neural Networks (CNNs) on various vision tasks. Despite this empirical success, the fundamental mechanisms explaining why MSAs work remained poorly understood. MSAs success has been attributed to "weak inductive bias" and "long-range dependency"—the ability to connect distant spatial locations in an image. Attributing MSAs success to those two traits conflict with common issues with MSAs such as the tendency to overfit training datasets, consequently leading to poor predictive performance in small data regime. 

# **Problems**: 
- MSAs are generally not defined well despite its ubiquitous success.
- What are listed as strengths for MSAs conflict with their weaknesses.
     - Specifically the "weak inductive bias" strength. If this is a benefit, why would MSAs struggle on small datasets?
- Local MSAs (small window MSAs) achieve better performance than global MSAs on small and large datasets.
- ViTs only outperform CNNs (Convolutional Neural Network) on large datasets.

# **Approach**:

This paper has three analytical approaches to addressing these problems. Most of which it will compare ViTs to [ResNeT](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)-A CNN image model
- **Loss Landscape Map Analysis**
  ![](Hessian_Eigenspace.png)
     - Looking at the Hessian (Second partial derivative) eigenvalues (how a matrix transforms space) of the loss landscape to measure local curvature and convexity.
     - Visualize loss landscapes using [filter normalization](https://arxiv.org/pdf/1712.09913) (calculated by taking the square root of the sum of the squares of all its elements)
     - Analyzes how local vs. global MSAs affect losses.
 
- **Fourier Domain Analysis**
     ![](Fourier_Domain_Analysis.png)
  - The authors examaine the feature map in the fourier domain
  - Meaasures high vs. low frequency accuracies

- **Feature Map Statistics**
       ![]()


# **Positives**:
   - Flatten lost landscapes (Due to data specificity, not long range dependency)
   - MSAs act as low pass filters
   - Play a key role in model's predictions if placed at the end of multi-stage neural networks.
- Convolutional Neural Networks (which acts as a high pass filter) is complimentary to MSAs as shown in their model **AlterNet**

_____
## Architecture overview

# AlterNet: Core Algorithm

## Algorithm 2: AlterNet Architecture Build-up Rule

**Input:** $\mathcal{B}$, baseline CNN architecture with $S$ stages (e.g., ResNet-50)

**Input:** $D \in \mathbb{N}$, dataset size

**Output:** $\mathcal{A}$, AlterNet architecture with Conv and MSA blocks

---

1. $\mathcal{A} \leftarrow \mathcal{B}$ ▷ Start with baseline CNN
2. $n_{\text{msa}} \leftarrow 0$ ▷ MSA block counter
3. **case** $D$ **of**
4. $\quad |D| \leq 50\text{K}$ (small data): $\text{max MSA} \leftarrow 4$ ▷ e.g., $CIFAR-100$
5. $\quad |D| \geq 1\text{M}$ (large data): $\text{max MSA} \leftarrow 6$ ▷ e.g., ImageNet
6. **end case**
7. **comment** Process stages from end to beginning
8. **for** $s = S$ **downto** $1$ **do**
9. $\quad \text{blocks in stage} \leftarrow \text{GetConvBlocks}(\mathcal{A}, s)$
10. $\quad$ **comment** KEY: Place MSA at end of stage (most important position)
11. $\quad$ **if** $n_{\text{msa}} < \text{max\_msa}$ **then**
12. $\quad \quad \text{last\_block} \leftarrow \text{blocks\_in\_stage}[|\text{blocks\_in\_stage}|]$
13. $\quad \quad \text{ConvertToMSA}(\text{last\_block})$ ▷ Replace last Conv in stage with MSA
14. $\quad \quad n_{\text{msa}} \leftarrow n_{\text{msa}} + 1$
15. $\quad$ **end if**
16. **end for**
17. **comment** Configure head counts per stage
18. **for** $s = 1$ **to** $S$ **do**
19. $\quad \text{heads}(s) \leftarrow [3, 6, 12, 24]_s$ ▷ Stage-wise head configuration
20. **end for**
21. **return** $\mathcal{A}$

---

## Algorithm 3: AlterNet Forward Pass (Single Stage)

**Input:** $X \in \mathbb{R}^{N \times H \times W \times C}$, input feature maps

**Input:** $L \in \mathbb{N}$, number of Conv blocks in stage

**Output:** $Y$, output feature maps after stage

**Components:** Conv blocks $\{\mathcal{C}_1, \ldots, \mathcal{C}_L\}$, MSA block $\mathcal{M}$, pooling $\mathcal{P}$

---

1. $Y \leftarrow X$
2. **comment** Apply Conv blocks sequentially
3. **for** $\ell = 1$ **to** $L$ **do**
4. $\quad Y \leftarrow \mathcal{C}_\ell(Y)$ ▷ Conv: learns high-frequency patterns
5. **end for**
6. **comment** KEY: Apply MSA at END of stage
7. **if** MSA exists in this stage **then**
8. $\quad Y \leftarrow \mathcal{M}(Y)$ ▷ MSA: spatial smoothing, variance reduction
9. **end if**
10. **comment** Prepare for next stage
11. **if** not last stage **then**
12. $\quad Y \leftarrow \mathcal{P}(Y)$ ▷ Pooling/subsampling
13. **end if**
14. **return** $Y$

---

## Algorithm 4: MSA as Spatial Smoothing (Theoretical Basis)

**Input:** Feature at position $i$: $e_i \in \mathbb{R}^{d}$

**Input:** All features: $E = [e_1, e_2, \ldots, e_N] \in \mathbb{R}^{N \times d}$

**Output:** Smoothed feature: $\tilde{e}_i \in \mathbb{R}^{d}$

---

1. **comment** Data-dependent ensemble of neighboring features
2. $q \leftarrow W_q e_i$ ▷ Query from current position
3. $K \leftarrow E W_k^T$ ▷ Keys from all N positions
4. $s \leftarrow q^T K^T / \sqrt{d}$ ▷ Similarity scores to all positions
5. $\alpha \leftarrow \text{Softmax}(s)$ ▷ Data-dependent attention weights
6. $V \leftarrow E W_v^T$ ▷ Values from all N positions
7. $\tilde{e}_i \leftarrow \alpha^T V$ ▷ **Weighted ensemble of all neighbors**
8. **return** $\tilde{e}_i$ ▷ Result: smoothed, lower-variance feature

---

## Core Design Principles

### Principle 1: MSA Placement Strategy
**Place MSA at end of each stage, not throughout:**
- Each stage acts as a mini-model with accumulated features
- MSA at stage-end ensembles outputs from all preceding Conv blocks
- Results in better feature aggregation than distributed MSAs

### Principle 2: Dataset-Dependent Configuration
| Dataset Size | MSA Blocks | Reason |
|---|---|---|
| Small (CIFAR-100) | 4 | Avoid non-convex loss landscape |
| Large (ImageNet) | 6 | Large data suppresses negative eigenvalues |

### Principle 3: Head Count Progression
$$\text{heads per stage} = [3, 6, 12, 24] \text{ for stages } [1, 2, 3, 4]$$

More heads → better loss landscape convexity in deeper stages

### Principle 4: MSA and Conv Complementarity

| Property | MSA (Low-Pass) | Conv (High-Pass) |
|---|---|---|
| Filter Type | Low-pass | High-pass |
| Variance Effect | **Reduces** | **Increases** |
| Loss Landscape | **Flattens** | **Sharpens** |
| Feature Bias | Shape-biased | Texture-biased |

**Key insight:** MSAs and Convs are complementary, not competing

---

## Why AlterNet Works

1. **Optimization:** MSA-flattened loss landscapes improve training
2. **Generalization:** Reduced feature variance → better test performance
3. **Data Efficiency:** Works on both small (CIFAR) and large (ImageNet) datasets
4. **Efficiency:** 20% faster training than ResNet

---

## References

- **Paper:** "How Do Vision Transformers Work?" (Park & Kim, ICLR 2022)
- **Code:** https://github.com/xxxnell/how-do-vits-work
