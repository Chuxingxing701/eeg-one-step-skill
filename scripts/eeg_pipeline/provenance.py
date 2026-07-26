"""SHA256 manifests for frozen EEG workflow evidence."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json


def sha256_file(path: Path, chunk_size: int = 8_388_608) -> str:
    """Hash a file without loading it fully into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(paths: list[Path], destination: Path) -> Path:
    """Write an exclusive JSON manifest for existing files."""

    files = []
    for path in paths:
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        files.append(
            {
                "path": str(resolved),
                "size_bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    payload = {"algorithm": "sha256", "files": files}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
    return destination


def verify_manifest(manifest: Path) -> list[Path]:
    """Return exact missing, size-mismatched, or hash-mismatched paths."""

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("algorithm") != "sha256":
        raise ValueError("只支持sha256清单")
    mismatches: list[Path] = []
    for record in payload.get("files", []):
        path = Path(record["path"])
        if (
            not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            mismatches.append(path.resolve())
    return mismatches
