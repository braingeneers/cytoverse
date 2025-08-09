"""
IVFPQ package - Inverted File Index with Product Quantization
"""

from .ivfpq import IVFPQ
from .pq import PQ, PQDistance
from .kmeans import KMeans

# For backward compatibility
InvertedFileIndex = IVFPQ

__all__ = [
    "IVFPQ",
    "PQ",
    "PQDistance",
    "KMeans",
]
