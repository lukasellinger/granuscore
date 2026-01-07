import os
from multiprocessing.pool import ThreadPool
from typing import Sequence, Iterator, TypeVar, Callable, Any

import torch
from spacy.util import is_package
from tqdm import tqdm

T = TypeVar("T")


def ensure_spacy_model(name: str = "en_core_web_sm") -> None:
    if is_package(name):
        return

    from spacy.cli import download
    download(name)


def iter_batches(items: Sequence[T], batch_size: int) -> Iterator[Sequence[T]]:
    for i in range(0, len(items), batch_size):
        yield items[i: i + batch_size]


def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def map_with_progress(
    f: Callable,
    xs: list[Any],
    num_threads: int = os.cpu_count() or 10,
    pbar: bool = False,
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