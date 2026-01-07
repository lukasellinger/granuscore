import os
from multiprocessing.pool import ThreadPool
from typing import Callable, Any

from tqdm import tqdm


def unique_normalized(xs):
    seen = {}
    for x in xs:
        k = x.strip().lower()
        if k not in seen:
            seen[k] = x.strip()
    return list(seen.values())


def map_with_progress(
    f: Callable,
    xs: list[Any],
    num_threads: int = os.cpu_count() or 10,
    pbar: bool = True,
):
    """
    Apply f to each element of xs, using a ThreadPool, and show progress.
    """
    pbar_fn = tqdm if pbar else lambda x, *args, **kwargs: x

    if os.getenv("debug"):
        return list(map(f, pbar_fn(xs, total=len(xs))))
    else:
        with ThreadPool(min(num_threads, len(xs))) as pool:
            return list(pbar_fn(pool.imap(f, xs), total=len(xs)))