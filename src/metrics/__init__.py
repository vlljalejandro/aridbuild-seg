"""
src.metrics — evaluation metrics for the building-segmentation pipeline.

    SegmentationMetrics   pixel-level IoU, F1, precision, recall, accuracy,
                          accumulated as raw counts across an epoch so the
                          result is micro-averaged rather than an average of
                          per-batch scores.
"""

from .segmentation import SegmentationMetrics

__all__ = ["SegmentationMetrics"]
