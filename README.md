# How_Do_Vision_Transformers_Work? - Analysis
**DS 5690 Paper Presntation**

*Authors: Namuk Park & Songkuk Kim (Yonsei University, NAVER AI Lab)*  
*Published at ICLR 2022*

**Full Citation:**  
Park, N., & Kim, S. (2022). How Do Vision Transformers Work?. In International Conference on Learning Representations (ICLR 2022). arXiv:2202.06709v4 [cs.CV]

___
## Overview 
- 

_____
## Architecture overview
```python
### Multi-Head Self-Attention (MSA) Mechanism

```python
# MSA as Spatial Smoothing with Data-Specific Kernels

def multi_head_self_attention(X, num_heads, d_model):
    """
    Multi-Head Self-Attention mechanism

    Args:
        X: Input tensor of shape (batch, seq_len, d_model)
        num_heads: Number of attention heads
        d_model: Model dimension

    Returns:
        Output tensor of shape (batch, seq_len, d_model)
    """
    # d_k is dimension per head
    d_k = d_model // num_heads

    # Split into multiple heads
    # Q, K, V projections
    Q = linear_projection(X, d_model, d_model)  # (batch, seq_len, d_model)
    K = linear_projection(X, d_model, d_model)
    V = linear_projection(X, d_model, d_model)

    # Reshape for multi-head: (batch, num_heads, seq_len, d_k)
    Q = reshape(Q, (batch, seq_len, num_heads, d_k)).transpose(1, 2)
    K = reshape(K, (batch, seq_len, num_heads, d_k)).transpose(1, 2)
    V = reshape(V, (batch, seq_len, num_heads, d_k)).transpose(1, 2)

    # Compute attention scores
    # scores[i,j] = similarity between position i and j
    scores = matmul(Q, K.transpose(-2, -1)) / sqrt(d_k)  # (batch, num_heads, seq_len, seq_len)

    # KEY INSIGHT: Softmax creates DATA-SPECIFIC importance weights
    # π(x_i|x_j) - importance of position i for position j
    attention_weights = softmax(scores, dim=-1)  # (batch, num_heads, seq_len, seq_len)

    # Aggregate values with learned importance
    # z_j = Σ_i π(x_i|x_j) * V_i,j
    # This is SPATIAL SMOOTHING with trainable, data-specific kernels
    output = matmul(attention_weights, V)  # (batch, num_heads, seq_len, d_k)

    # Concatenate heads and project
    output = output.transpose(1, 2).reshape(batch, seq_len, d_model)
    output = linear_projection(output, d_model, d_model)

    return output


# Key difference from standard convolution:
# - Convolution: Fixed kernel, data-agnostic, channel-specific
# - MSA: Learned kernel PER INPUT, data-specific, channel-agnostic
```

### Local MSA (Swin-style)

```python
def local_multi_head_self_attention(X, num_heads, d_model, window_size):
    """
    Local MSA with restricted receptive field

    Args:
        X: Input feature map (batch, H, W, d_model)
        window_size: Size of local window (e.g., 7x7)

    Key Insight: Local MSA reduces degrees of freedom while
    maintaining data specificity - better than global MSA!
    """
    batch, H, W, d_model = X.shape

    # Partition into non-overlapping windows
    # windows: (batch * num_windows, window_size, window_size, d_model)
    windows = window_partition(X, window_size)

    # Reshape windows to sequence
    # (batch * num_windows, window_size^2, d_model)
    windows = windows.reshape(-1, window_size * window_size, d_model)

    # Apply MSA within each window only
    # This restricts long-range dependencies but maintains data specificity
    attention_output = multi_head_self_attention(windows, num_heads, d_model)

    # Reverse window partition
    output = window_reverse(attention_output, window_size, H, W)

    return output

# Paper's Key Finding:
# - Local MSA (window_size=7) > Global MSA
# - Even Local MSA (window_size=3) works well!
# - This proves: DATA SPECIFICITY matters, not long-range dependency
```

### AlterNet Architecture

```python
def alternet_stage(X, num_conv_blocks, num_msa_blocks, channels):
    """
    AlterNet: Alternating Convolution and MSA blocks

    Key Design Principle:
    - Place Conv blocks at the BEGINNING of stages (feature extraction)
    - Place MSA blocks at the END of stages (spatial aggregation)

    Args:
        X: Input feature map
        num_conv_blocks: Number of ResNet-style conv blocks
        num_msa_blocks: Number of MSA blocks (typically 1)
        channels: Number of channels
    """
    # Phase 1: Conv blocks for feature transformation
    # Convs are HIGH-PASS filters - amplify high-frequency details
    for i in range(num_conv_blocks):
        X = resnet_bottleneck_block(X, channels)

    # Phase 2: MSA blocks for spatial smoothing
    # MSAs are LOW-PASS filters - reduce high-frequency, ensemble predictions
    for i in range(num_msa_blocks):
        X = msa_block(X, channels, num_heads)

    return X


def alternet_model(input_image):
    """
    Full AlterNet model architecture

    Structure: 4 stages with progressive downsampling
    MSA heads increase with depth: [3, 6, 12, 24]

    Why this works:
    1. Convs diversify features (high-pass filtering)
    2. MSAs aggregate and smooth (low-pass filtering)
    3. Complementary behaviors lead to better representations
    """
    X = input_image  # (batch, 3, 224, 224)

    # Stem: Initial convolution
    X = conv_layer(X, 64, kernel=7, stride=2)  # (batch, 64, 112, 112)
    X = batch_norm(X)
    X = relu(X)
    X = max_pool(X, kernel=3, stride=2)  # (batch, 64, 56, 56)

    # Stage 1: 3 Conv blocks + 1 MSA block, heads=3
    X = alternet_stage(X, num_conv_blocks=3, num_msa_blocks=1, channels=256)
    X = downsample(X)  # (batch, 256, 28, 28)

    # Stage 2: 4 Conv blocks + 1 MSA block, heads=6
    X = alternet_stage(X, num_conv_blocks=4, num_msa_blocks=1, channels=512)
    X = downsample(X)  # (batch, 512, 14, 14)

    # Stage 3: 6 Conv blocks + 1 MSA block, heads=12
    X = alternet_stage(X, num_conv_blocks=6, num_msa_blocks=1, channels=1024)
    X = downsample(X)  # (batch, 1024, 7, 7)

    # Stage 4: 3 Conv blocks + 1 MSA block, heads=24
    # More heads in late stages = stronger spatial aggregation
    X = alternet_stage(X, num_conv_blocks=3, num_msa_blocks=1, channels=2048)

    # Classification head
    X = global_average_pooling(X)  # (batch, 2048)
    X = fully_connected(X, num_classes)  # (batch, num_classes)

    return X


# Key Differences from Standard Architectures:
# 
# vs. ResNet:
#   - AlterNet adds MSA blocks at stage ends
#   - MSAs provide spatial smoothing that Convs lack
# 
# vs. ViT:
#   - AlterNet uses mostly Conv blocks, fewer MSA blocks
#   - Works well even on small datasets (CIFAR)
# 
# vs. Swin:
#   - Similar philosophy but simpler
#   - Convs at beginning, MSAs at end of each stage
```
