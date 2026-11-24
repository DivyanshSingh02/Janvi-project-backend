"""Measure how the configured recognition model scores same-person and different-person image pairs on YOUR dataset.

    python -m scripts.evaluate_recognition DATASET_FOLDER [--thresholds 0.5,0.6] [--target-far 0.01,0.001]
                                                          [--max-images-per-person N]

DATASET_FOLDER holds one sub-folder per person with that person's images (LFW has exactly this layout):

    DATASET_FOLDER/<person>/<image>.jpg

Use a public dataset whose licence allows this, or photos taken with explicit informed consent. NEVER use real case
photos, and never commit the dataset (the folder name evaluation-datasets/ is git-ignored).

What it does: every image goes through the same one-face rule as the API (extract_single_face). Images with no usable
face, with several faces, or that cannot be read are counted and skipped. Then every pair of usable images is scored
with cosine similarity: pairs of the same person ("same") and of different people ("different"). It prints both
distributions and, for some threshold candidates, how many different-person pairs would have been accepted (false
accepts) and how many same-person pairs would have been rejected (false rejects) at that threshold ("accept when the
similarity is >= the threshold").

What it does NOT do: it makes no accuracy claim and chooses no threshold for you. The numbers describe this dataset,
this model and these quality gates only. Pairs share images, so they are not independent samples and the rates have
no confidence interval; a rate of 0 means "none seen", not "impossible". Check the result per group of people and
per photo condition on your own data before trusting any threshold.

Privacy: the output is counts and numbers only: no folder names, no file names, no paths.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.recognition import (
    MultipleFacesError,
    NoFaceDetectedError,
    RecognitionUnavailableError,
    extract_single_face,
    get_recognition_service,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
BIN_COUNT = 4000                               # histogram of scores from -1 to 1: bins 0.0005 wide
BIN_EDGES = np.linspace(-1.0, 1.0, BIN_COUNT + 1)
BLOCK_ROWS = 256                               # rows of the score matrix computed at once (bounds memory)
DEFAULT_TARGET_FARS = (0.1, 0.01, 0.001, 0.0001)
THRESHOLD_DECIMALS = 4


# ---------------------------------------------------------------- argument parsing
def _number_list(value: str, low: float, high: float, what: str, *, exclusive: bool = False) -> list[float]:
    try:
        numbers = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"{what} must be comma-separated numbers") from None
    def inside(x: float) -> bool:
        return low < x < high if exclusive else low <= x <= high

    if not numbers or any(not math.isfinite(x) or not inside(x) for x in numbers):
        raise argparse.ArgumentTypeError(f"{what} must be finite numbers from {low} to {high}")
    return sorted(set(numbers))


def _threshold_values(value: str) -> list[float]:
    return _number_list(value, -1.0, 1.0, "thresholds")


def _far_values(value: str) -> list[float]:
    return _number_list(value, 0.0, 1.0, "target false-accept rates", exclusive=True)


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a whole number of at least 1") from None
    if number < 1:
        raise argparse.ArgumentTypeError("must be a whole number of at least 1")
    return number


# ---------------------------------------------------------------- collecting embeddings
def _image_files(person_dir: Path) -> list[Path]:
    return sorted(p for p in person_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _person_folders(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir())


@dataclass
class Collected:
    vectors: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    images: int = 0
    no_face: int = 0            # includes images that cannot be decoded: they have no usable face
    multiple_faces: int = 0
    unreadable: int = 0         # the file could not be read from disk
    people: int = 0             # folders with at least one image

    @property
    def usable(self) -> int:
        return len(self.vectors)

    @property
    def skipped(self) -> int:
        return self.no_face + self.multiple_faces + self.unreadable


def collect_embeddings(root: Path, recognition, max_images_per_person: int | None = None) -> Collected:
    """Raises RecognitionUnavailableError when the model fails: that must stop the run, not be counted as 'skipped'."""
    collected = Collected()
    label = 0
    for person_dir in _person_folders(root):
        files = _image_files(person_dir)
        if max_images_per_person is not None:
            files = files[:max_images_per_person]
        if not files:
            continue
        collected.people += 1
        for path in files:
            collected.images += 1
            try:
                data = path.read_bytes()
            except OSError:
                collected.unreadable += 1
                continue
            try:
                face = extract_single_face(recognition, data)
            except NoFaceDetectedError:
                collected.no_face += 1
                continue
            except MultipleFacesError:
                collected.multiple_faces += 1
                continue
            collected.vectors.append(face.embedding)
            collected.labels.append(label)
        label += 1
    return collected


# ---------------------------------------------------------------- scoring pairs
def iter_pair_scores(matrix: np.ndarray, labels: np.ndarray):
    """Yields (same_person_scores, different_person_scores) for blocks of rows; every unordered pair exactly once."""
    count = matrix.shape[0]
    columns = np.arange(count)[None, :]
    for start in range(0, count, BLOCK_ROWS):
        stop = min(count, start + BLOCK_ROWS)
        scores = np.clip(matrix[start:stop] @ matrix.T, -1.0, 1.0).astype(np.float64)
        later = columns > np.arange(start, stop)[:, None]
        same_label = labels[start:stop, None] == labels[None, :]
        yield scores[later & same_label], scores[later & ~same_label]


class Distribution:
    """Count, exact minimum / maximum / mean and a histogram (percentiles are exact to half a bin: 0.00025)."""

    def __init__(self) -> None:
        self.counts = np.zeros(BIN_COUNT, dtype=np.int64)
        self.n = 0
        self.total = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def add(self, scores: np.ndarray) -> None:
        if scores.size == 0:
            return
        self.counts += np.histogram(scores, bins=BIN_EDGES)[0]
        self.n += int(scores.size)
        self.total += float(scores.sum())
        self.minimum = min(self.minimum, float(scores.min()))
        self.maximum = max(self.maximum, float(scores.max()))

    def percentile(self, q: float) -> float:
        cumulative = np.cumsum(self.counts)
        index = min(int(np.searchsorted(cumulative, q / 100.0 * self.n, side="left")), BIN_COUNT - 1)
        return float((BIN_EDGES[index] + BIN_EDGES[index + 1]) / 2.0)

    def describe(self, label: str) -> str:
        if self.n == 0:
            return f"{label}: pairs=0 (not enough usable data)"
        return (
            f"{label}: pairs={self.n} min={self.minimum:.4f} p05={self.percentile(5):.4f} "
            f"median={self.percentile(50):.4f} mean={self.total / self.n:.4f} p95={self.percentile(95):.4f} "
            f"max={self.maximum:.4f}"
        )


def _scan(matrix: np.ndarray, labels: np.ndarray) -> tuple[Distribution, Distribution]:
    same, different = Distribution(), Distribution()
    for same_scores, different_scores in iter_pair_scores(matrix, labels):
        same.add(same_scores)
        different.add(different_scores)
    return same, different


def derive_candidates(same: Distribution, different: Distribution, target_fars) -> tuple[list, list]:
    """Threshold candidates from the histograms: (candidates, notes). A candidate is (description, threshold).
    - for each target false-accept rate f: the lowest threshold whose false-accept rate is <= f (only when the data
      can resolve it, i.e. there are at least 1/f different-person pairs);
    - the equal-error point: where the false-accept and false-reject rates are closest."""
    candidates: list[tuple[str, float]] = []
    notes: list[str] = []
    if same.n == 0 or different.n == 0:
        return candidates, notes
    # accepted_different[k] = different-person scores >= BIN_EDGES[k]; rejected_same[k] = same-person scores < BIN_EDGES[k]
    accepted_different = np.append(np.cumsum(different.counts[::-1])[::-1], 0)
    rejected_same = np.append(0, np.cumsum(same.counts))
    for far in target_fars:
        if different.n * far < 1:
            notes.append(
                f"target_far={far:g} not resolvable: {different.n} different-person pairs "
                f"(at least {math.ceil(1 / far)} are needed)"
            )
            continue
        index = int(np.argmax(accepted_different <= far * different.n))
        candidates.append((f"target_far={far:g}", float(BIN_EDGES[min(index, BIN_COUNT)])))
    far_curve = accepted_different / different.n
    frr_curve = rejected_same / same.n
    index = int(np.argmin(np.abs(far_curve - frr_curve)))
    candidates.append(("equal_error_point", float(BIN_EDGES[index])))
    return candidates, notes


def exact_counts(matrix: np.ndarray, labels: np.ndarray, thresholds: list[float]) -> dict[float, tuple[int, int]]:
    """(false accepts, false rejects) at each threshold, counted exactly on the scores themselves."""
    false_accepts = {t: 0 for t in thresholds}
    false_rejects = {t: 0 for t in thresholds}
    for same_scores, different_scores in iter_pair_scores(matrix, labels):
        for t in thresholds:
            false_accepts[t] += int((different_scores >= t).sum())
            false_rejects[t] += int((same_scores < t).sum())
    return {t: (false_accepts[t], false_rejects[t]) for t in thresholds}


# ---------------------------------------------------------------- the report
def build_report(collected: Collected, explicit_thresholds, target_fars) -> list[str]:
    lines = [
        f"images={collected.images} usable={collected.usable} skipped={collected.skipped} "
        f"skipped_no_face={collected.no_face} skipped_multiple_faces={collected.multiple_faces} "
        f"skipped_unreadable={collected.unreadable}"
    ]
    labels = np.asarray(collected.labels, dtype=np.int64)
    per_person = np.bincount(labels) if labels.size else np.array([], dtype=np.int64)
    lines.append(
        f"people={collected.people} people_with_usable_images={int((per_person > 0).sum())} "
        f"people_with_2_or_more_usable_images={int((per_person > 1).sum())}"
    )
    if collected.usable < 2:
        lines.append("Not enough usable images for any pair.")
        return lines + _closing()
    matrix = np.stack(collected.vectors).astype(np.float32)
    same, different = _scan(matrix, labels)
    lines.append(f"same_person_pairs={same.n} different_person_pairs={different.n}")
    lines.append(same.describe("same_person"))
    lines.append(different.describe("different_person"))
    if same.n == 0 or different.n == 0:
        lines.append("Threshold trade-off unavailable: both same-person and different-person pairs are required.")
        return lines + _closing()

    candidates, notes = derive_candidates(same, different, target_fars)
    if explicit_thresholds:
        candidates = [("given", t) for t in explicit_thresholds]
    rounded = sorted({round(t, THRESHOLD_DECIMALS) for _, t in candidates})
    counts = exact_counts(matrix, labels, rounded)
    lines.append("threshold candidates (accept as the same person when similarity >= threshold):")
    for description, threshold in candidates:
        t = round(threshold, THRESHOLD_DECIMALS)
        fa, fr = counts[t]
        lines.append(
            f"  {description} threshold={t:.4f} false_accept={fa}/{different.n} ({fa / different.n:.4f}) "
            f"false_reject={fr}/{same.n} ({fr / same.n:.4f})"
        )
    lines.extend(f"  {note}" for note in notes)
    return lines + _closing()


def _closing() -> list[str]:
    return [
        "These numbers describe this dataset, this model and these quality gates only. They are not accuracy claims,",
        "not an identity probability and not a recommended threshold. A match is a lead for human review.",
    ]


# ---------------------------------------------------------------- command line
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.evaluate_recognition",
        description="Measure same-person / different-person similarity distributions; makes no accuracy claim.",
    )
    parser.add_argument("dataset", type=Path, help="Folder with one sub-folder of images per person")
    parser.add_argument("--thresholds", type=_threshold_values, default=None,
                        help="Your own threshold candidates, comma-separated, from -1 to 1 (replaces the derived ones)")
    parser.add_argument("--target-far", type=_far_values, default=list(DEFAULT_TARGET_FARS),
                        help="False-accept rates to derive thresholds for (default 0.1,0.01,0.001,0.0001)")
    parser.add_argument("--max-images-per-person", type=_positive_int, default=None,
                        help="Use at most this many images per person (the first ones, sorted): bounds the run time")
    args = parser.parse_args(argv)

    if not args.dataset.is_dir():
        print("Error: dataset folder does not exist.")
        return 2
    if not any(_image_files(folder) for folder in _person_folders(args.dataset)):
        print("No images found. images=0 usable=0 skipped=0")
        return 0
    try:
        recognition = get_recognition_service()
        recognition.model_info()
        collected = collect_embeddings(args.dataset, recognition, args.max_images_per_person)
    except RecognitionUnavailableError:
        print("Error: the configured recognition model is unavailable (is RECOGNITION_BACKEND=facenet and are the "
              "weights and packages installed? See docs/recognition.md).")
        return 1
    for line in build_report(collected, args.thresholds, args.target_far):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
