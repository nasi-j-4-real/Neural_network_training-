# Module Assignment: GPU Acceleration in Deep Learning

## 1. Conceptual Understanding
A CPU is designed for sequential processing with a few powerful cores, whereas a GPU contains thousands of smaller, specialized cores built for massive parallelism.

Deep learning training relies heavily on high-volume matrix multiplications and vector additions, 
which a GPU can compute simultaneously across thousands of data points instead of processing them one by one.

---

## 2. Environment Check
```python
import torch
print("CUDA Available:", torch.cuda.is_available())
print("GPU Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
