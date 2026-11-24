"""The three Stage 6B scripts. None of these tests needs torch, a real model, a real dataset or the network."""
import hashlib
import re
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from fake_recognition import FakeRecognitionService
from sqlalchemy import select

from app.models.face_embedding import FaceEmbedding
from app.recognition import UnconfiguredRecognitionService
from scripts import download_recognition_model, evaluate_recognition, reembed_cases

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


# ------------------------------------------------------------------ argument handling
def test_download_script_help_does_not_download():
    with pytest.raises(SystemExit) as exc:
        download_recognition_model.main(["--help"])
    assert exc.value.code == 0


def test_reembed_script_help_does_not_load_a_model():
    with pytest.raises(SystemExit) as exc:
        reembed_cases.main(["--help"])
    assert exc.value.code == 0


def test_evaluation_script_help_does_not_load_a_model():
    with pytest.raises(SystemExit) as exc:
        evaluate_recognition.main(["--help"])
    assert exc.value.code == 0


def test_the_scripts_refuse_unknown_arguments():
    for script in (download_recognition_model, reembed_cases, evaluate_recognition):
        with pytest.raises(SystemExit) as exc:
            script.main(["--no-such-option"])
        assert exc.value.code == 2


def test_threshold_and_rate_parsers_accept_valid_values_and_reject_the_rest():
    assert evaluate_recognition._threshold_values("-0.5, 0.5,0.5") == [-0.5, 0.5]
    assert evaluate_recognition._far_values("0.01,0.001") == [0.001, 0.01]
    for parser, bad in (
        (evaluate_recognition._threshold_values, "0.5,1.2"), (evaluate_recognition._threshold_values, "abc"),
        (evaluate_recognition._threshold_values, ""), (evaluate_recognition._threshold_values, "nan"),
        (evaluate_recognition._far_values, "0"), (evaluate_recognition._far_values, "1"),
        (evaluate_recognition._positive_int, "0"), (evaluate_recognition._positive_int, "x"),
    ):
        with pytest.raises(Exception):
            parser(bad)
    assert evaluate_recognition._positive_int("3") == 3


# ------------------------------------------------------------------ evaluation script
def _dataset(root, make_image_bytes, fake):
    """Two fictional people with two images each, plus one image without a face and one with two faces.
    Folder and file names are canaries: they must never be printed."""
    layout = {
        "CANARY_PERSON_A": [[1, 0, 0, 0], [0.9, 0.1, 0, 0]],
        "CANARY_PERSON_B": [[0, 1, 0, 0], [0.1, 0.9, 0, 0]],
    }
    seed = 0
    for person, vectors in layout.items():
        (root / person).mkdir()
        for index, vector in enumerate(vectors):
            seed += 1
            data = make_image_bytes("PNG", seed=seed)
            (root / person / f"CANARY_FILE_{index}.png").write_bytes(data)
            fake.set_faces(data, vector)
    (root / "CANARY_PERSON_C").mkdir()
    no_face = make_image_bytes("PNG", seed=90)
    (root / "CANARY_PERSON_C" / "CANARY_NOFACE.png").write_bytes(no_face)
    two_faces = make_image_bytes("PNG", seed=91)
    (root / "CANARY_PERSON_C" / "CANARY_TWO.png").write_bytes(two_faces)
    fake.set_faces(two_faces, [1, 0, 0, 0], [0, 1, 0, 0])
    (root / "CANARY_PERSON_C" / "notes.txt").write_text("not an image")


def test_evaluation_empty_folder_prints_counts_only_and_never_loads_a_model(tmp_path, capsys, monkeypatch):
    def must_not_be_called():
        pytest.fail("the model must not be loaded for an empty dataset")

    monkeypatch.setattr(evaluate_recognition, "get_recognition_service", must_not_be_called)
    (tmp_path / "CANARY_EMPTY_PERSON").mkdir()
    assert evaluate_recognition.main([str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "No images found. images=0 usable=0 skipped=0" in output
    assert "CANARY" not in output


def test_evaluation_refuses_a_missing_folder_without_loading_a_model(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(evaluate_recognition, "get_recognition_service", lambda: pytest.fail("no model"))
    assert evaluate_recognition.main([str(tmp_path / "missing")]) == 2
    assert "dataset folder does not exist" in capsys.readouterr().out


def test_evaluation_reports_counts_and_measured_rates_and_prints_no_names(tmp_path, capsys, monkeypatch, make_image_bytes):
    fake = FakeRecognitionService()
    _dataset(tmp_path, make_image_bytes, fake)
    monkeypatch.setattr(evaluate_recognition, "get_recognition_service", lambda: fake)
    assert evaluate_recognition.main([str(tmp_path), "--thresholds", "0.0,0.15,0.5,0.999"]) == 0
    output = capsys.readouterr().out
    assert "images=6 usable=4 skipped=2 skipped_no_face=1 skipped_multiple_faces=1 skipped_unreadable=0" in output
    assert "people=3 people_with_usable_images=2 people_with_2_or_more_usable_images=2" in output
    assert "same_person_pairs=2 different_person_pairs=4" in output
    # Scores: same-person 0.9939 (x2); different-person 0.0, 0.1104, 0.1104, 0.2195.
    assert re.search(r"threshold=0\.0000 false_accept=4/4 \(1\.0000\) false_reject=0/2 \(0\.0000\)", output)
    assert re.search(r"threshold=0\.1500 false_accept=1/4 \(0\.2500\) false_reject=0/2 \(0\.0000\)", output)
    assert re.search(r"threshold=0\.5000 false_accept=0/4 \(0\.0000\) false_reject=0/2 \(0\.0000\)", output)
    assert re.search(r"threshold=0\.9990 false_accept=0/4 \(0\.0000\) false_reject=2/2 \(1\.0000\)", output)
    assert "not accuracy claims" in output
    assert "CANARY" not in output and "notes" not in output and str(tmp_path) not in output


def test_evaluation_derives_candidates_and_says_when_a_rate_cannot_be_resolved(tmp_path, capsys, monkeypatch, make_image_bytes):
    fake = FakeRecognitionService()
    _dataset(tmp_path, make_image_bytes, fake)
    monkeypatch.setattr(evaluate_recognition, "get_recognition_service", lambda: fake)
    assert evaluate_recognition.main([str(tmp_path), "--target-far", "0.5,0.0001"]) == 0
    output = capsys.readouterr().out
    assert "target_far=0.5 threshold=" in output
    assert "equal_error_point threshold=" in output
    assert "target_far=0.0001 not resolvable: 4 different-person pairs (at least 10000 are needed)" in output


def test_evaluation_limits_images_per_person(tmp_path, capsys, monkeypatch, make_image_bytes):
    fake = FakeRecognitionService()
    _dataset(tmp_path, make_image_bytes, fake)
    monkeypatch.setattr(evaluate_recognition, "get_recognition_service", lambda: fake)
    assert evaluate_recognition.main([str(tmp_path), "--max-images-per-person", "1"]) == 0
    output = capsys.readouterr().out
    assert "same_person_pairs=0 different_person_pairs=1" in output
    assert "Threshold trade-off unavailable" in output


def test_a_model_that_breaks_during_the_run_stops_it_instead_of_counting_skipped_images(
    tmp_path, capsys, monkeypatch, make_image_bytes
):
    fake = FakeRecognitionService()
    _dataset(tmp_path, make_image_bytes, fake)
    fake.fail_on_call = 2
    monkeypatch.setattr(evaluate_recognition, "get_recognition_service", lambda: fake)
    assert evaluate_recognition.main([str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert "recognition model is unavailable" in output
    assert "usable=" not in output and "CANARY" not in output


def test_evaluation_with_no_backend_configured_says_so(tmp_path, capsys, monkeypatch, make_image_bytes):
    _dataset(tmp_path, make_image_bytes, FakeRecognitionService())
    monkeypatch.setattr(evaluate_recognition, "get_recognition_service", UnconfiguredRecognitionService)
    assert evaluate_recognition.main([str(tmp_path)]) == 1
    assert "RECOGNITION_BACKEND=facenet" in capsys.readouterr().out


def test_pair_scores_cover_every_pair_once_even_across_blocks(monkeypatch):
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(7, 5)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    labels = np.array([0, 0, 1, 1, 1, 2, 3])
    monkeypatch.setattr(evaluate_recognition, "BLOCK_ROWS", 3)
    same, different = [], []
    for same_scores, different_scores in evaluate_recognition.iter_pair_scores(matrix, labels):
        same.extend(same_scores.tolist())
        different.extend(different_scores.tolist())
    expected_same, expected_different = [], []
    for i in range(7):
        for j in range(i + 1, 7):
            (expected_same if labels[i] == labels[j] else expected_different).append(float(matrix[i] @ matrix[j]))
    assert sorted(same) == pytest.approx(sorted(expected_same), abs=1e-6)
    assert sorted(different) == pytest.approx(sorted(expected_different), abs=1e-6)
    assert len(same) + len(different) == 21


def test_derived_thresholds_match_the_scores_they_were_derived_from():
    rng = np.random.default_rng(0)
    same_scores = np.clip(rng.normal(0.8, 0.05, 2000), -1, 1)
    different_scores = np.clip(rng.normal(0.1, 0.1, 20000), -1, 1)
    same, different = evaluate_recognition.Distribution(), evaluate_recognition.Distribution()
    same.add(same_scores)
    different.add(different_scores)
    candidates, notes = evaluate_recognition.derive_candidates(same, different, [0.01, 0.001])
    assert notes == []
    by_name = dict(candidates)
    step = 0.0005
    for far, name in ((0.01, "target_far=0.01"), (0.001, "target_far=0.001")):
        threshold = by_name[name]
        assert (different_scores >= threshold).mean() <= far                       # meets the target ...
        assert (different_scores >= threshold - step).mean() > far                 # ... and is the lowest edge that does
    eer = by_name["equal_error_point"]
    far_rate, frr_rate = (different_scores >= eer).mean(), (same_scores < eer).mean()
    assert abs(far_rate - frr_rate) < 0.01
    assert different.percentile(50) == pytest.approx(float(np.median(different_scores)), abs=0.001)
    assert same.minimum == float(same_scores.min()) and different.maximum == float(different_scores.max())


# ------------------------------------------------------------------ download script (no network)
class _FakeResponse:
    def __init__(self, data, declared=None, fail_after=None):
        self._data, self._pos, self._fail_after = data, 0, fail_after
        self.headers = {} if declared is None else {"Content-Length": str(declared)}

    def read(self, size):
        if self._fail_after is not None and self._pos >= self._fail_after:
            raise OSError("connection dropped")
        chunk = self._data[self._pos:self._pos + size]
        self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


PAYLOAD = b"fictional weights " * 100_000          # 1.8 MB: above the plausibility floor


def _serve(monkeypatch, response):
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout=None: response)


def test_download_saves_the_file_and_prints_its_size_and_sha256(tmp_path, capsys, monkeypatch):
    _serve(monkeypatch, _FakeResponse(PAYLOAD, declared=len(PAYLOAD)))
    assert download_recognition_model.main(["--directory", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    saved = tmp_path / download_recognition_model.WEIGHTS_FILENAME
    assert saved.read_bytes() == PAYLOAD
    assert f"Size: {len(PAYLOAD)} bytes" in output
    assert f"SHA-256: {hashlib.sha256(PAYLOAD).hexdigest()}" in output
    assert "github.com/timesler/facenet-pytorch" in output and "licence" in output
    assert [p.name for p in tmp_path.iterdir()] == [saved.name]                    # no temporary file left behind


def test_download_never_replaces_an_existing_file_without_force(tmp_path, capsys, monkeypatch):
    saved = tmp_path / download_recognition_model.WEIGHTS_FILENAME
    saved.write_bytes(b"original")
    _serve(monkeypatch, _FakeResponse(PAYLOAD))
    assert download_recognition_model.main(["--directory", str(tmp_path)]) == 2
    assert saved.read_bytes() == b"original"
    assert download_recognition_model.main(["--directory", str(tmp_path), "--force"]) == 0
    assert saved.read_bytes() == PAYLOAD
    capsys.readouterr()


def test_a_failed_incomplete_or_tiny_download_saves_nothing(tmp_path, capsys, monkeypatch):
    for response in (
        _FakeResponse(PAYLOAD, fail_after=500_000),             # connection drops half way
        _FakeResponse(PAYLOAD, declared=len(PAYLOAD) + 10),     # fewer bytes than announced
        _FakeResponse(b"<html>not the weights</html>"),         # far too small
    ):
        _serve(monkeypatch, response)
        assert download_recognition_model.main(["--directory", str(tmp_path)]) == 1
        assert "could not be downloaded or saved" in capsys.readouterr().out
        assert list(tmp_path.iterdir()) == []


def test_check_only_needs_an_existing_file_and_reports_an_unloadable_one(tmp_path, capsys):
    assert download_recognition_model.main(["--directory", str(tmp_path), "--check-only"]) == 2
    (tmp_path / download_recognition_model.WEIGHTS_FILENAME).write_bytes(PAYLOAD)
    assert download_recognition_model.main(["--directory", str(tmp_path), "--check-only"]) == 1    # not a checkpoint
    output = capsys.readouterr().out
    assert f"SHA-256: {hashlib.sha256(PAYLOAD).hexdigest()}" in output and "Load check: failed" in output


def test_no_checksum_is_hard_coded_anywhere_in_the_scripts():
    for script in SCRIPTS_DIR.glob("*.py"):
        assert not re.search(r"\b[0-9a-f]{64}\b", script.read_text(encoding="utf-8")), script.name


# ------------------------------------------------------------------ re-embedding script (needs the test database)
class _SessionContext:
    def __init__(self, db):
        self.db = db

    def __call__(self):
        return self

    def __enter__(self):
        return self.db

    def __exit__(self, *args):
        return False


@pytest.fixture()
def reembed_env(db, storage, monkeypatch, fake_recognition):
    monkeypatch.setattr(reembed_cases, "_session_context", _SessionContext(db))
    monkeypatch.setattr(reembed_cases, "get_storage", lambda: storage)
    monkeypatch.setattr(reembed_cases, "get_recognition_service", lambda: fake_recognition)
    return fake_recognition


def _approved_case(db, make_user, make_case, make_photos, fake, email, name, seed, vector=(1, 0, 0, 0)):
    owner = make_user(email=email)
    case = make_case(owner, name=name)
    photos = make_photos(owner, case, 1, first_seed=seed)
    fake.set_faces(photos[0].sha256, list(vector))
    case.status = "approved"                                  # set directly: no approval hook runs
    db.commit()
    return case, photos


def test_reembed_prints_counts_only_and_a_second_run_changes_nothing(
    db, make_user, make_case, make_photos, reembed_env, capsys
):
    _approved_case(db, make_user, make_case, make_photos, reembed_env, "canary-owner-1@example.test", "CANARY Case One", 11)
    _approved_case(db, make_user, make_case, make_photos, reembed_env, "canary-owner-2@example.test", "CANARY Case Two", 22)
    assert reembed_cases.main([]) == 0
    output = capsys.readouterr().out
    assert output.strip() == "cases=2 cases_skipped=0 cases_failed=0 photos_done=2 photos_failed=0 photos_pending=0"
    calls_after_first = len(reembed_env.calls)
    assert reembed_cases.main([]) == 0
    assert len(reembed_env.calls) == calls_after_first                           # nothing was processed again
    assert "CANARY" not in output and "canary" not in output


def test_only_approved_cases_are_processed(db, make_user, make_case, make_photos, reembed_env, capsys):
    case, _ = _approved_case(db, make_user, make_case, make_photos, reembed_env, "o1@example.test", "Fictional One", 31)
    pending_owner = make_user(email="o2@example.test")
    pending_case = make_case(pending_owner, name="Fictional Two")
    make_photos(pending_owner, pending_case, 1, first_seed=32)
    assert reembed_cases.main([]) == 0
    assert capsys.readouterr().out.startswith("cases=1 ")
    assert db.scalar(select(FaceEmbedding.id).where(FaceEmbedding.case_id == pending_case.id)) is None


def test_a_new_model_version_replaces_the_old_embeddings(db, make_user, make_case, make_photos, reembed_env, capsys):
    case, photos = _approved_case(db, make_user, make_case, make_photos, reembed_env, "o1@example.test", "Fictional One", 41)
    assert reembed_cases.main([]) == 0
    reembed_env.version = "2"                                                  # the backend / weights were switched
    reembed_env.set_faces(photos[0].sha256, [0, 1, 0, 0])
    assert reembed_cases.main([]) == 0
    capsys.readouterr()
    db.expire_all()
    rows = list(db.scalars(select(FaceEmbedding).where(FaceEmbedding.case_id == case.id)))
    assert [row.model_version for row in rows] == ["2"]                         # replaced, not duplicated


def test_a_problem_with_one_case_is_counted_and_the_others_are_still_processed(
    db, make_user, make_case, make_photos, reembed_env, capsys
):
    _approved_case(db, make_user, make_case, make_photos, reembed_env, "o1@example.test", "Fictional One", 51)
    _approved_case(db, make_user, make_case, make_photos, reembed_env, "o2@example.test", "Fictional Two", 52)
    _approved_case(db, make_user, make_case, make_photos, reembed_env, "o3@example.test", "Fictional Three", 53)
    reembed_env.crash_on_call = 2                                               # the second photo blows up
    assert reembed_cases.main([]) == 1
    assert capsys.readouterr().out.strip() == (
        "cases=3 cases_skipped=0 cases_failed=1 photos_done=2 photos_failed=0 photos_pending=0"
    )


def test_an_unavailable_model_stops_the_run_with_a_generic_message(
    db, make_user, make_case, make_photos, reembed_env, capsys
):
    _approved_case(db, make_user, make_case, make_photos, reembed_env, "o1@example.test", "Fictional One", 61)
    reembed_env.unavailable = True
    assert reembed_cases.main([]) == 1
    output = capsys.readouterr().out
    assert "recognition model is unavailable" in output and "cases=" not in output
    assert db.scalar(select(FaceEmbedding.id)) is None


def test_no_backend_configured_is_reported_not_crashed(monkeypatch, capsys):
    monkeypatch.setattr(reembed_cases, "get_recognition_service", UnconfiguredRecognitionService)
    assert reembed_cases.main([]) == 1
    assert "recognition model is unavailable" in capsys.readouterr().out
