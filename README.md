# How Do Vision Transformers Work? - Analysis
**DS 5690 Paper Presntation**

*Authors: Namuk Park & Songkuk Kim (Yonsei University, NAVER AI Lab)*  
*Published at ICLR 2022*

**Full Citation:**  
Park, N., & Kim, S. (2022). How Do Vision Transformers Work?. In International Conference on Learning Representations (ICLR 2022). arXiv:2202.06709v4 [cs.CV]
___
# Overview
**The abstract says it all**: 

"The success of multi-head self-attentions (MSAs) for computer vision is now indisputable. However, little is known about how MSAs work. We present fundamental explanations to help better understand the nature of MSAs.";

"MSAs are low-pass filters, but Convs are high-pass filters. Therefore, MSAs and Convs are complementary.";

"We propose AlterNet, a model in which Convblocks at the end of a stage are replaced with MSA blocks."


## **Context**:
Following the success of the original Vision Transformer (ViT) introduced by [Dosovitskiy et al.](https://arxiv.org/pdf/2010.11929) in 2021, Multi-head Self-Attention (MSA) mechanisms have become ubiquitous in computer vision. By 2022, numerous variants—including [Swin Transformer](https://arxiv.org/pdf/2103.14030) and [PiT](https://arxiv.org/pdf/2103.16302) demonstrated that MSAs could match or exceed the performance of traditional Convolutional Neural Networks (CNNs) on various vision tasks. Despite this empirical success, the fundamental mechanisms explaining why MSAs work remained poorly understood. MSAs success has been attributed to "weak inductive bias" and "long-range dependency"—the ability to connect distant spatial locations in an image. Attributing MSAs success to those two traits conflict with common issues with MSAs such as the tendency to overfit training datasets, consequently leading to poor predictive performance in small data regime. 

## **Problems**: 
- MSAs are generally not defined well despite its ubiquitous success.
- What are listed as strengths for MSAs conflict with their weaknesses.
     - Specifically the "weak inductive bias" strength. If this is a benefit, why would **MSAs struggle on small datasets?**
- Local MSAs (small window MSAs) achieve better performance than global MSAs on small and large datasets.

## **Approach**:

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
       ![](Feature_map_Variance.png)
  - Tracks feature map variance across network layers
  - Performs lesion studies (removing individual layers) to measure importance
  - Analyzes representational similarity using [CKA](https://arxiv.org/pdf/2010.15327) (Centered Kernel Alignment)
 
## **How the problems were addressed**:
- MSAs are generally not defined well despite its ubiquitous success
  - "MSAs work by addressing themselves as a general form of spatial smoothing or an implementation of ensemble averaging for proximate data points"
    
- What are listed as strengths for MSAs conflict with their weaknesses
  - The paper redefines weak inductive bias as a **LIABILITY**, not a strength. ~ "A small patch size, or a weak inductive bias, produces negative eigenvalues" Meaning non-convex losses.
    
- Local MSAs (small window MSAs) achieve better performance than global MSAs on small and large datasets
  - ![](Window_Size_Comparison.png)
  - Smaller MSA windows have more convex losses as shown above.
    
## **MSA Positives**:
   - Flatten lost landscapes (Due to data specificity, not long range dependency)
   - MSAs act as low pass filters
   - Play a key role in model's predictions if placed at the end of multi-stage neural networks.
   - Convolutional Neural Networks (which acts as a high pass filter) is complimentary to MSAs as shown in their model **AlterNet**

_____
# Architecture overview
## AlterNet

_____
# Critical Analysis

## **What could have been developed further**:
- "If the added MSA block does not improve predictive performance, replace a Conv block located at the end of an earlier stage with an MSA block." (Build-up rule; Section 4.1)
- The above rule appears to be a rule the authors learned as they experiemented, but did not take the time to understand why architecturally this works well. They should have taken the time to find the theoretical reasoning behind this.
- My using this paper as a source, I would assume this is a rule because of the loss flattening ability of MSAs, but it is never explicitly stated in the reading.

## **How did the study stand years after?**:
- Data Specificity > Long-Range Dependency did not hold up with time

**Scaling trends contradict this**:
- Paper's argument: "Local MSAs (5×5) > Global MSAs (8×8)"
     - Therefore → Long-range dependency not important
- Reality: [ViT-22B](https://arxiv.org/pdf/2302.05442) (global attention)
     - Achieve state-of-the-art on many tasks
     - Long-range DOES matter at scale!

_____
## Impacts
- The paper reframed understading of MSAs. They would now be thought of as a generalized spatial smoothing rather than incorrectely seeing them as long-range dependency exploiters
- The Fourier analysis revealing MSAs as low-pass filters while CNNs are high-pass filters provides:
     - A mathematical characterization of architectural differences
     - Explanation for ViT frequency dependant advantages. 
     - Justification for hybrid architectures (Convs for texture/high-frequency, MSAs for shape/low-frequency)
- Shifted the Narrative from "Weak Inductive Bias" to "Complementary Mechanisms"
_____
## Code Demonstration

_____
# Citations
- **Paper:** "How Do Vision Transformers Work?" (Park & Kim, ICLR 2022)
- **Code:** https://github.com/xxxnell/how-do-vits-work
