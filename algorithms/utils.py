import numpy as np


def manhattan_distance(r1: int, c1: int, r2: int, c2: int) -> int:
    return abs(r2 - r1) + abs(c2 - c1)


def step_toward(cur_r: int, cur_c: int, tgt_r: int, tgt_c: int, m: int) -> tuple[int, int]:
    dr = tgt_r - cur_r
    dc = tgt_c - cur_c

    dx = int(np.clip(dc, -m, m))
    rem = m - abs(dx)
    dy = int(np.clip(dr, -rem, rem))
    return dx, dy
