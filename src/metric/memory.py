"""Static memory accounting for P3 queues (formula Г2-149, spec §6.9). No TensorFlow."""


def queue_bytes(n_cls: int, K: int, d: int, bz: int = 32) -> int:
    return n_cls * K * d * bz // 8


def memory_breakdown(n_cls: int, K: int, d: int, n_params: int) -> dict:
    b = {
        "M_Q": queue_bytes(n_cls, K, d),
        "sums": n_cls * d * 8,  # S in float64
        "pointers": 2 * n_cls * 4,  # start, n int32
        "ins_step": n_cls * K * 8,  # int64 per slot
        "proxies": n_cls * d * 4,
        "weights": n_params * 4,
        "adam": 2 * (n_params + n_cls * d) * 4,  # two fp32 moments, proxies are trainable
    }
    b["total_static"] = sum(b.values())
    return b
