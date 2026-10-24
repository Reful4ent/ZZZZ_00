import pytest

from metric.augment import resized_shape

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("h,w,short,want", [
    (480, 640, 256, (256, 341)),
    (640, 480, 256, (341, 256)),
    (256, 256, 256, (256, 256)),
    (50, 300, 256, (256, 1536)),
    (2, 3, 1, (1, 2)),
    (100, 150, 256, (256, 384)),
])
def test_resized_shape_table(h, w, short, want):
    assert resized_shape(h, w, short) == want
