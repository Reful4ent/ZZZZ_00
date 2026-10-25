"""P1 photometric — стенд глобального сравнения изображений (ТЗ 01, раздел 3.2).

Публичный API (ТЗ 01, раздел 6):
    prepare_image(path, color_profile)
    represent(image, method_config)
    compare(query_repr, reference_repr)
    score_pair(pair, config)
"""

from photometric.io import prepare_image
from photometric.methods import compare, represent
from photometric.scoring import score_pair

__all__ = ["prepare_image", "represent", "compare", "score_pair"]
__version__ = "1.0.0"
