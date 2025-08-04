"""
IVFPQ (Inverted File Index Product Quantization) module for approximate nearest neighbor search.

This module implements IVFPQ for large-scale similarity search, designed to work with
single-cell embedding datasets. It supports:

- Product Quantization (PQ) for vector compression
- Inverted File Index (IVF) for dataset partitioning
- ONNX export for browser-side inference
- Parquet export for efficient data loading

References:
- https://towardsdatascience.com/similarity-search-with-ivfpq-9c6348fd4db3
- https://towardsdatascience.com/ivfpq-hnsw-for-billion-scale-similarity-search-89ff2f89d90e/
"""

from .pq import ProductQuantizer
from .ivf import InvertedFileIndex, IVFPQ

__version__ = "0.1.0"
__all__ = ["ProductQuantizer", "InvertedFileIndex", "IVFPQ"]
