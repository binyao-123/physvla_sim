"""Mirror the S3-hosted assets referenced by tasks/123.usd and write tasks/123_local.usd."""

import re
import sys
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path

from pxr import Sdf, UsdUtils

S3_PREFIX = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
ROOT = Path(__file__).resolve().parents[1]
SRC_SCENE = ROOT / "tasks" / "123.usd"
DST_SCENE = ROOT / "tasks" / "123_local.usd"
MIRROR = ROOT / "assets" / "s3_mirror"

LAYER_EXTS = {".usd", ".usda", ".usdc"}
MDL_TEXTURE_RE = re.compile(r'"([^"]+\.(?:png|jpg|jpeg|exr|hdr|tga|dds|mdl))"', re.IGNORECASE)


def local_path(url: str) -> Path:
    return MIRROR / urllib.parse.unquote(url[len(S3_PREFIX):])


def resolve(ref: str, base_url: str) -> str | None:
    ref = ref.strip()
    if not ref or "<UDIM>" in ref or ref.startswith(("omniverse:", "file:")):
        return None
    if ref.startswith(S3_PREFIX):
        return ref
    if ref.startswith(("http://", "https://")) or ref.startswith("/"):
        return None
    if not ref.startswith("."):
        # Bare names such as OmniPBR.mdl are resolved from the MDL search path, not S3.
        if Path(ref).suffix.lower() == ".mdl" and "/" not in ref:
            return None
    return urllib.parse.urljoin(base_url, ref)


def download(url: str) -> bool:
    dst = local_path(url)
    if dst.is_file() and dst.stat().st_size > 0:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
    except Exception as exc:
        print(f"[WARN] {url}: {exc}")
        return False
    dst.write_bytes(data)
    print(f"[OK] {url} ({len(data) / 1024:.0f} KB)")
    return True


def layer_refs(path: Path) -> list[str]:
    layer = Sdf.Layer.FindOrOpen(str(path))
    if layer is None:
        return []
    refs: list[str] = []

    def _collect(p: str) -> str:
        refs.append(p)
        return p

    UsdUtils.ModifyAssetPaths(layer, _collect)
    return refs


def localize_layer(path: Path) -> None:
    layer = Sdf.Layer.FindOrOpen(str(path))
    if layer is None:
        return

    def _rewrite(p: str) -> str:
        if p.startswith(S3_PREFIX) and local_path(p).is_file():
            return str(local_path(p))
        return p

    UsdUtils.ModifyAssetPaths(layer, _rewrite)
    if layer.dirty:
        layer.Save()


def main() -> int:
    seeds = [p for p in layer_refs(SRC_SCENE) if p.startswith(S3_PREFIX)]
    print(f"[INFO] {len(seeds)} S3 references in {SRC_SCENE.name}")
    queue = deque(seeds)
    seen: set[str] = set()
    failed: list[str] = []
    while queue:
        url = queue.popleft()
        if url in seen:
            continue
        seen.add(url)
        if not download(url):
            failed.append(url)
            continue
        dst = local_path(url)
        suffix = dst.suffix.lower()
        if suffix in LAYER_EXTS:
            children = layer_refs(dst)
        elif suffix == ".mdl":
            children = MDL_TEXTURE_RE.findall(dst.read_text(errors="ignore"))
        else:
            children = []
        for ref in children:
            child = resolve(ref, url)
            if child and child not in seen:
                queue.append(child)

    print(f"[INFO] downloaded {len(seen) - len(failed)}/{len(seen)} files into {MIRROR}")
    missing_top = [u for u in seeds if not local_path(u).is_file()]
    if missing_top:
        print(f"[ERROR] top-level assets not downloaded; {DST_SCENE.name} not written:")
        for u in missing_top:
            print(f"  {u}")
        return 1

    if MIRROR.is_dir():
        for path in MIRROR.rglob("*"):
            if path.suffix.lower() in LAYER_EXTS:
                localize_layer(path)

    if DST_SCENE.is_file() and "--overwrite_scene" not in sys.argv:
        print(f"[INFO] {DST_SCENE} exists, kept as is (pass --overwrite_scene to regenerate from 123.usd)")
        return 0
    src = Sdf.Layer.FindOrOpen(str(SRC_SCENE))
    src.Export(str(DST_SCENE))
    localize_layer(DST_SCENE)
    print(f"[INFO] wrote {DST_SCENE}")
    if failed:
        print(f"[WARN] {len(failed)} dependent files failed (usually textures / built-in MDL); scene still loads.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
