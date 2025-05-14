# AIE External Kernel Library
## Norm
### LayerNorm
$$
\mu = \frac{1}{d} \sum_{i=1}^{d} x_i
$$

$$
\sigma = \sqrt{ \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2 + \epsilon }
$$

$$
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma}
$$

```python
import torch


def layernorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    x: shape [..., dim] - input tensor
    weight: shape [dim] - scale parameter γ
    eps: small constant for numerical stability
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    normalized = (x - mean) / torch.sqrt(var + eps)
    return normalized * weight


batch_size = 4
seq_len = 16
hidden_size = 1024

input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
weight = torch.randn(hidden_size, dtype=torch.float32)
output = layernorm(input_tensor, weight)
```

### RMSNorm
$$
\text{RMS}(x) = \sqrt{ \frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon }
$$

$$
\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\text{RMS}(x)}
$$

```python
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x, weight):
        norm = x.norm(dim=-1, keepdim=True)  # L2 norm along last dim
        rms = norm / (x.shape[-1] ** 0.5)
        return x / (rms + self.eps) * weight


batch_size = 4
seq_len = 16
hidden_size = 1024

input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)
weight = torch.randn(hidden_size, dtype=torch.float16)

rms_norm = RMSNorm()
output = rms_norm(input_tensor, weight)
print(output)
```