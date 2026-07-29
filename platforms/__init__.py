"""実行基盤の解決。**コアは特定の基盤の名前を持たない。**

名前から動的に読み込む。ここに基盤名を並べた表を置くと、
その時点でコアが基盤を知ってしまう。

    from platforms import load_backend
    backend = load_backend("local")     # 名前は利用者が渡す

`tests/test_platform_isolation.py` がこの分離を機構で守る。
"""

from __future__ import annotations

import importlib
from pathlib import Path

from .base import Backend, JobResult, JobSpec

__all__ = ["Backend", "JobSpec", "JobResult", "load_backend",
           "available_backends"]


def available_backends() -> list[str]:
    """`platforms/<name>/backend.py` があるものを列挙する。

    表を持たない。**足したら自動で見つかる。**
    """
    here = Path(__file__).resolve().parent
    return sorted(p.parent.name for p in here.glob("*/backend.py"))


def load_backend(name: str):
    """名前から基盤モジュールを読み込む。

    見つからないときは、**何が使えるのかを添えて**落とす。
    「使えません」だけでは利用者が次に何をすべきか分からない。
    """
    if name not in available_backends():
        raise ValueError(
            f"そのような実行基盤はありません: {name!r}。"
            f"使えるのは {available_backends()}")
    return importlib.import_module(f"{__name__}.{name}.backend")
