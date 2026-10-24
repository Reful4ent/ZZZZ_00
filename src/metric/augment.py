"""Input pipeline helpers; keep TF-free at module level."""


def resized_shape(h: int, w: int, short: int) -> tuple[int, int]:
    """(H', W') after aspect-preserving resize; long side is integer round-half-up."""
    s = min(h, w)
    def grow(x: int) -> int:
        return short if x == s else (x * short + s // 2) // s
    return grow(h), grow(w)
