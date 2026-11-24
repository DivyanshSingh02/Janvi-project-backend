"""Explicit, one-time download of the FaceNet (InceptionResnetV1, VGGFace2) weights. YOU run it; the API never does.

    python -m scripts.download_recognition_model [--directory DIR] [--force] [--check | --check-only]

Where the file comes from: the release asset "20180402-114759-vggface2.pt" of the facenet-pytorch project on GitHub
(see WEIGHTS_URL). It is saved as facenet-vggface2.pt in RECOGNITION_MODEL_DIR (or --directory). The script prints
the size and the SHA-256 of the file it SAVED, computed from the file on disk. No checksum is hard-coded here: none
could be verified. Write the printed value down; compare it if you ever copy or re-download the file.

MTCNN's own (small) detector weights are not downloaded here: they ship inside the facenet-pytorch package.

LICENCE: these pretrained weights derive from the VGGFace2 dataset. The project's MIT licence covers its code, not
necessarily the dataset or the weights. Check the current terms of the dataset and of this checkpoint BEFORE any real
deployment (docs/recognition.md).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

WEIGHTS_URL = (
    "https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt"
)
WEIGHTS_FILENAME = "facenet-vggface2.pt"       # keep in sync with app/recognition/facenet.py
MIN_PLAUSIBLE_BYTES = 1_000_000                # a real checkpoint is about 100 MB; anything below is not one
CHUNK_BYTES = 1024 * 1024


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_weights(
    directory: Path, *, force: bool = False, timeout: float = 60.0, opener=None
) -> tuple[Path, int, str]:
    """Download into `directory`. Returns (path, size in bytes, SHA-256) of the SAVED file.
    Raises FileExistsError (nothing is replaced without force) or OSError. The file is written to a temporary name
    and moved into place only when complete, so an interrupted download never leaves a half file under the real name."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / WEIGHTS_FILENAME
    if destination.exists() and not force:
        raise FileExistsError("The weights file already exists; pass --force to replace it")

    descriptor, temporary_name = tempfile.mkstemp(prefix=".facenet-download-", suffix=".part", dir=directory)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        received = 0
        request = urllib.request.Request(WEIGHTS_URL, headers={"User-Agent": "missing-person-api-weights-download"})
        with (opener or urllib.request.urlopen)(request, timeout=timeout) as response, temporary.open("wb") as output:
            declared = response.headers.get("Content-Length") if getattr(response, "headers", None) else None
            while True:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                output.write(chunk)
                received += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        if declared is not None and declared.isdigit() and int(declared) != received:
            raise OSError("The download was incomplete")
        if received < MIN_PLAUSIBLE_BYTES:
            raise OSError("The downloaded file is too small to be the weights")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination, destination.stat().st_size, sha256_of(destination)


def _directory(argument: str | None) -> Path:
    if argument is not None:
        return Path(argument)
    from app.core.config import get_settings          # imported late: --help and --directory need no .env

    return Path(get_settings().recognition_model_dir)


def _load_check(path: Path) -> str:
    """Loads the file into the embedding network (needs torch and facenet-pytorch). Never raises."""
    try:
        from app.recognition.facenet import verify_weights_file

        version = verify_weights_file(path)
    except Exception as exc:                          # RecognitionUnavailableError or an import problem
        return f"failed: {exc}"
    return f"ok (facenet-pytorch {version}; the weights fit the network)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.download_recognition_model",
        description="Download the FaceNet VGGFace2 weights once. Never run by the API.",
    )
    parser.add_argument("--directory", default=None, help="Destination folder (default: RECOGNITION_MODEL_DIR)")
    parser.add_argument("--force", action="store_true", help="Replace an existing weights file")
    parser.add_argument("--timeout", type=float, default=60.0, help="Network timeout in seconds (default 60)")
    check = parser.add_mutually_exclusive_group()
    check.add_argument("--check", action="store_true", help="After saving, load the weights with PyTorch to check them")
    check.add_argument("--check-only", action="store_true", help="Do not download: check the existing file")
    args = parser.parse_args(argv)

    try:
        directory = _directory(args.directory)
    except Exception:
        print("Error: could not read RECOGNITION_MODEL_DIR from the settings; fix .env or pass --directory.")
        return 2

    if args.check_only:
        path = directory / WEIGHTS_FILENAME
        if not path.is_file():
            print("Error: there is no weights file in that folder.")
            return 2
    else:
        try:
            path, _size, _digest = download_weights(directory, force=args.force, timeout=args.timeout)
        except FileExistsError as exc:
            print(f"Error: {exc}")
            return 2
        except Exception as exc:                      # no secrets are involved: the type and text help to diagnose
            print(f"Error: the weights could not be downloaded or saved ({type(exc).__name__}: {exc}).")
            return 1
        print(f"Source: {WEIGHTS_URL}")

    print(f"Saved: {WEIGHTS_FILENAME}" if not args.check_only else f"File: {WEIGHTS_FILENAME}")
    print(f"Size: {path.stat().st_size} bytes")
    print(f"SHA-256: {sha256_of(path)}")
    print("Record the SHA-256 above. Check the licence of the VGGFace2 weights before real use (docs/recognition.md).")
    if args.check or args.check_only:
        outcome = _load_check(path)
        print(f"Load check: {outcome}")
        if outcome.startswith("failed"):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
