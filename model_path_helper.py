from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


def path_is_ascii(path: Path) -> bool:
    try:
        str(path).encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


def native_safe_model_path(model_path: Path) -> Path:
    """Return a model path that native Windows loaders can open reliably."""

    resolved = model_path.resolve()
    if path_is_ascii(resolved):
        return resolved

    cache_dir = Path(tempfile.gettempdir()) / "insight_cv_models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_path = cache_dir / resolved.name

    needs_copy = True
    if cached_path.exists():
        source_stat = resolved.stat()
        cached_stat = cached_path.stat()
        needs_copy = (
            source_stat.st_size != cached_stat.st_size
            or source_stat.st_mtime > cached_stat.st_mtime
        )
    if needs_copy:
        shutil.copy2(resolved, cached_path)

    if not path_is_ascii(cached_path):
        raise RuntimeError(
            "The temporary model path still contains non-ASCII characters. "
            "Move this project to an ASCII-only path such as C:\\insight."
        )
    return cached_path
