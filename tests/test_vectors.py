"""The vector helpers on their own (numpy only: no database, no HTTP)."""
import numpy as np
import pytest

from app.recognition import vectors


def test_identical_vectors_have_similarity_one():
    assert vectors.cosine_similarity([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0, abs=1e-9)


def test_orthogonal_vectors_have_similarity_zero():
    assert vectors.cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0, abs=1e-9)


def test_opposite_vectors_have_similarity_minus_one():
    assert vectors.cosine_similarity([1, 2, 3], [-1, -2, -3]) == pytest.approx(-1.0, abs=1e-9)


def test_the_length_of_a_vector_does_not_change_the_similarity():
    assert vectors.cosine_similarity([1, 0], [5, 0]) == pytest.approx(1.0, abs=1e-9)
    assert vectors.cosine_similarity([3, 4], [6, 8]) == pytest.approx(1.0, abs=1e-9)
    assert vectors.cosine_similarity([1, 0], [0.6, 0.8]) == pytest.approx(0.6, abs=1e-6)


def test_normalise_gives_a_float32_vector_of_length_one():
    unit = vectors.normalise([3, 4, 0, 0])
    assert unit.dtype == np.float32
    assert float(np.linalg.norm(unit)) == pytest.approx(1.0, abs=1e-6)
    assert unit.tolist() == pytest.approx([0.6, 0.8, 0.0, 0.0], abs=1e-6)


def test_zero_nan_and_infinite_vectors_are_rejected_everywhere():
    bad_vectors = [[0, 0, 0], [float("nan"), 1, 2], [1, float("inf"), 2], [1, float("-inf"), 2]]
    for bad in bad_vectors:
        with pytest.raises(vectors.InvalidVectorError):
            vectors.normalise(bad)
        with pytest.raises(vectors.InvalidVectorError):
            vectors.pack(bad)
        with pytest.raises(vectors.InvalidVectorError):
            vectors.cosine_similarity(bad, [1, 1, 1])
        with pytest.raises(vectors.InvalidVectorError):
            vectors.cosine_similarity([1, 1, 1], bad)


def test_empty_nested_and_non_numeric_vectors_are_rejected():
    for bad in ([], [[1, 2], [3, 4]], ["a", "b"], None):
        with pytest.raises(vectors.InvalidVectorError):
            vectors.normalise(bad)


def test_vectors_of_different_lengths_cannot_be_compared():
    with pytest.raises(vectors.InvalidVectorError):
        vectors.cosine_similarity([1, 0, 0], [1, 0])


def test_pack_and_unpack_round_trip_as_float32_little_endian():
    original = vectors.normalise([0.1, -0.2, 0.3, 0.4, -0.5])
    packed = vectors.pack(original)
    assert len(packed) == 5 * 4
    restored = vectors.unpack(packed, 5)
    assert restored.dtype == np.float32
    assert np.array_equal(restored, original)
    assert vectors.pack([1.0]) == b"\x00\x00\x80\x3f"          # 1.0 as little-endian float32


def test_unpack_refuses_bytes_that_do_not_match_the_dimension():
    packed = vectors.pack([1, 2, 3, 4])
    for dimension in (3, 5, 0, -4):
        with pytest.raises(vectors.InvalidVectorError):
            vectors.unpack(packed, dimension)
    with pytest.raises(vectors.InvalidVectorError):
        vectors.unpack(packed[:-1], 4)


def test_cosine_similarities_scores_every_row_and_gives_nan_for_an_unusable_row():
    matrix = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, 0, 0], [0.6, 0.8, 0]], dtype=np.float32)
    scores = vectors.cosine_similarities(matrix, [1, 0, 0])
    assert scores[0] == pytest.approx(1.0, abs=1e-6)
    assert scores[1] == pytest.approx(0.0, abs=1e-6)
    assert scores[2] == pytest.approx(-1.0, abs=1e-6)
    assert np.isnan(scores[3])                                   # a zero row is never ranked
    assert scores[4] == pytest.approx(0.6, abs=1e-6)


def test_cosine_similarities_refuses_a_matrix_of_the_wrong_width():
    with pytest.raises(vectors.InvalidVectorError):
        vectors.cosine_similarities(np.ones((2, 4), dtype=np.float32), [1, 0, 0])
    with pytest.raises(vectors.InvalidVectorError):
        vectors.cosine_similarities(np.ones(4, dtype=np.float32), [1, 0, 0, 0])
