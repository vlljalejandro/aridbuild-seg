"""
src.postprocess — inference post-processing.

    watershed     probability map + foreground mask -> instance labels
    polygonize    instance labels -> georeferenced shapely polygons

Applied outside the network and identically to every model: threshold the
probability map, split merged blobs with a probability-seeded watershed,
then vectorise.

Not included: georef.py, edge_merge.py and instance_merge.py, which stitched
instances across tile boundaries for national-scale inference. That is outside
the scope of this paper, and those modules assumed a z15 mosaic filename
convention that does not match the released tile index.
"""

from .watershed import apply_watershed, batch_watershed
from .polygonize import instance_labels_to_polygons

__all__ = [
    "apply_watershed",
    "batch_watershed",
    "instance_labels_to_polygons",
]
