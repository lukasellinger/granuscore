import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import requests
from importlib.resources import files
from platformdirs import user_cache_dir


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    subdir: str              # "faiss" or "models"
    url: str | None = None   # if None: must be bundled or already cached
    sha256: str | None = None


class ArtifactManager:
    """
    Resolution order:
      1) cache (and sha256 if provided)
      2) bundled assets inside the package -> copied to cache
      3) download from spec.url (public) -> cached
    """

    def __init__(self, app_name: str = "granuscore"):
        self.cache_root = Path(user_cache_dir(app_name))
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def ensure(self, spec: ArtifactSpec, force: bool = False) -> Path:
        dst_dir = self.cache_root / spec.subdir
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / spec.name

        # 1) cache hit
        if dst.exists() and not force:
            if spec.sha256 is None or _sha256(dst) == spec.sha256:
                return dst
            dst.unlink(missing_ok=True)

        # 2) bundled asset
        bundled = files("granuscore") / "../../assets" / spec.subdir / spec.name
        if bundled.is_file() and not force:
            self._copy_to_cache(bundled, dst)
            self._verify_hash_if_needed(dst, spec)
            return dst

        # 3) public download
        if not spec.url:
            raise FileNotFoundError(
                f"{spec.name} not found in cache and not bundled, and no public download URL provided."
            )

        self._download_public(spec.url, dst)
        self._verify_hash_if_needed(dst, spec)
        return dst

    def _verify_hash_if_needed(self, path: Path, spec: ArtifactSpec) -> None:
        if spec.sha256 is None:
            return
        got = _sha256(path)
        if got != spec.sha256:
            path.unlink(missing_ok=True)
            raise ValueError(
                f"Hash mismatch for {spec.name}. Expected {spec.sha256}, got {got}."
            )

    def _copy_to_cache(self, src_path: Path, dst: Path) -> None:
        tmp = dst.with_suffix(dst.suffix + ".part")
        with src_path.open("rb") as src, tmp.open("wb") as out:
            shutil.copyfileobj(src, out)
        tmp.replace(dst)

    def _download_public(self, url: str, dst: Path) -> None:
        tmp = dst.with_suffix(dst.suffix + ".part")

        with requests.get(url, stream=True, timeout=60, allow_redirects=True) as r:
            # Common mistake: GitHub "blob" links download HTML
            if r.status_code >= 400 and "github.com/" in url and "/blob/" in url:
                raise requests.HTTPError(
                    "You are using a GitHub 'blob' URL (HTML). Use a release asset URL "
                    "(.../releases/download/...) or a raw.githubusercontent.com URL."
                )

            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0)) or None

            try:
                from tqdm import tqdm  # type: ignore
                bar = tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {dst.name}")
            except Exception:
                bar = None

            try:
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        if bar is not None:
                            bar.update(len(chunk))
            finally:
                if bar is not None:
                    bar.close()

        os.replace(tmp, dst)