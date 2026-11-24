"""Embedding vectors: storage format, normalisation and cosine similarity (numpy).

Stored format: float32, little-endian, exactly dimension * 4 bytes (the database CHECK enforces the length)."""
import numpy as np

_STORED_DTYPE = np.dtype("<f4")


class InvalidVectorError(ValueError):
    """Empty, not one-dimensional, containing NaN or infinity, or all zeros."""


def _checked(vector) -> np.ndarray:
    try:
        array = np.asarray(vector, dtype=np.float32)
    except (TypeError, ValueError):
        raise InvalidVectorError("Not a numeric vector") from None
    if array.ndim != 1 or array.size == 0:
        raise InvalidVectorError("A vector must be one-dimensional and not empty")
    if not np.isfinite(array).all():
        raise InvalidVectorError("A vector must not contain NaN or infinity")
    return array


def normalise(vector) -> np.ndarray:
    """The vector scaled to length 1 (float32). A zero vector has no direction, so it is rejected."""
    array = _checked(vector)
    norm = float(np.linalg.norm(array.astype(np.float64)))
    if not np.isfinite(norm) or norm == 0.0:
        raise InvalidVectorError("A zero vector cannot be normalised")
    return (array / norm).astype(np.float32)


def pack(vector) -> bytes:
    """The vector as float32 little-endian bytes (it is NOT normalised here: normalise first if you want that).
    A zero vector is refused: it can never be compared with anything."""
    array = _checked(vector)
    if not array.any():
        raise InvalidVectorError("A zero vector cannot be stored")
    return array.astype(_STORED_DTYPE).tobytes()


def unpack(data: bytes, dimension: int) -> np.ndarray:
    if dimension < 1 or len(data) != dimension * 4:
        raise InvalidVectorError("The stored bytes do not match the dimension")
    return _checked(np.frombuffer(data, dtype=_STORED_DTYPE, count=dimension).astype(np.float32))


def _unit64(vector) -> np.ndarray:
    """Like normalise(), but kept in float64 so the similarity does not pick up float32 rounding."""
    array = _checked(vector).astype(np.float64)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm == 0.0:
        raise InvalidVectorError("A zero vector cannot be normalised")
    return array / norm


def cosine_similarity(first, second) -> float:
    """1 for the same direction, 0 for orthogonal vectors, -1 for opposite ones."""
    a, b = _unit64(first), _unit64(second)
    if a.shape != b.shape:
        raise InvalidVectorError("The vectors have different dimensions")
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def cosine_similarities(matrix: np.ndarray, query) -> np.ndarray:
    """The cosine similarity of every row of `matrix` (n x dimension) to `query`, as float64. A row that is
    all zeros (or not finite) gets NaN, so the caller can skip it instead of ranking it."""
    q = _unit64(query)
    rows = np.asarray(matrix, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != q.shape[0]:
        raise InvalidVectorError("The matrix does not match the query dimension")
    norms = np.linalg.norm(rows, axis=1)
    usable = np.isfinite(norms) & (norms > 0)
    result = np.full(rows.shape[0], np.nan, dtype=np.float64)
    result[usable] = np.clip((rows[usable] @ q) / norms[usable], -1.0, 1.0)
    return result
