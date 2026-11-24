"""Files that decide what can leak or break outside the code: .gitignore, the requirements files, .env.example."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _lines(name: str) -> list[str]:
    return (ROOT / name).read_text(encoding="utf-8").splitlines()


def test_model_weights_and_evaluation_datasets_are_git_ignored_on_their_own_lines():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "\\n" not in text                                    # a literal backslash-n once made these entries useless
    ignored = {line.strip() for line in text.splitlines()}
    assert {"models/recognition/", "evaluation-datasets/", ".env", "storage/"} <= ignored


def test_the_base_requirements_stay_light_and_keep_their_pins():
    base = _lines("requirements.txt")
    assert "numpy==2.1.3" in base and "Pillow==11.3.0" in base
    for line in base:
        assert not re.match(r"(?i)(torch|torchvision|facenet)", line), line


def test_the_optional_requirements_include_the_base_and_do_not_repin_numpy_or_pillow():
    optional = [line.strip() for line in _lines("requirements-facenet.txt") if line.strip() and not line.startswith("#")]
    assert optional[0] == "-r requirements.txt"
    names = {re.split(r"[=<>!~\[ ]", line, maxsplit=1)[0].lower() for line in optional[1:]}
    assert {"torch", "torchvision", "facenet-pytorch"} <= names
    assert not names & {"numpy", "pillow"}                      # never downgrade a base pin silently
    assert all("==" in line for line in optional[1:])           # everything is pinned exactly


def test_env_example_documents_the_recognition_settings_with_safe_values():
    values = dict(line.split("=", 1) for line in _lines(".env.example") if "=" in line and not line.startswith("#"))
    assert values["RECOGNITION_BACKEND"] == "none"
    assert values["RECOGNITION_MODEL_DIR"]
    assert int(values["RECOGNITION_MIN_FACE_SIZE"]) >= 12
    assert 0 < float(values["RECOGNITION_MIN_DETECTION_PROBABILITY"]) <= 1
    assert values["SECRET_KEY"] == "CHANGE_ME" and values["POSTGRES_PASSWORD"] == "CHANGE_ME"
