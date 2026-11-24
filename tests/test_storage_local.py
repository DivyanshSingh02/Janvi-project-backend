from pathlib import Path

import pytest

from app.core.config import get_settings
from app.storage import (
    InvalidStorageKeyError,
    LocalStorage,
    StorageObjectNotFoundError,
    StorageWriteError,
    get_storage,
)

BAD_KEYS = [
    "../x", "..", "a/../../b", "a/..", "/abs", "/etc/passwd", "a\\b", "C:/x", "C:\\x", "",
    "a//b", "a/", "./a", "a/./b", "a b", "a\x00b", "con", "nul.jpg", "a/COM1.txt", "name.",
    "a?b", "a:b", "é.jpg", "x" * 256,
]
OPERATIONS = {
    "save": lambda store, key: store.save(key, b"data", "image/jpeg"),
    "read": lambda store, key: store.read(key),
    "delete": lambda store, key: store.delete(key),
    "exists": lambda store, key: store.exists(key),
}


@pytest.fixture()
def root(tmp_path) -> Path:
    return tmp_path / "storage-root"


@pytest.fixture()
def store(root) -> LocalStorage:
    return LocalStorage(root)


def test_save_read_exists_and_delete_round_trip(store, root):
    key = "cases/3f2a9c10/abc-DEF_1.jpg"
    assert store.exists(key) is False
    store.save(key, b"fictional image bytes", "image/jpeg")
    assert store.exists(key) is True
    assert store.read(key) == b"fictional image bytes"
    assert (root / "cases" / "3f2a9c10" / "abc-DEF_1.jpg").is_file()
    store.delete(key)
    assert store.exists(key) is False


def test_saving_again_replaces_the_content(store):
    store.save("a/b.jpg", b"first", "image/jpeg")
    store.save("a/b.jpg", b"second", "image/jpeg")
    assert store.read("a/b.jpg") == b"second"


def test_deleting_a_missing_key_does_nothing(store):
    store.delete("never/stored.jpg")
    store.delete("never/stored.jpg")


def test_reading_a_missing_key_raises_a_storage_error(store):
    with pytest.raises(StorageObjectNotFoundError):
        store.read("never/stored.jpg")


def test_every_operation_rejects_unsafe_keys_and_touches_nothing(store, root):
    for key in BAD_KEYS:
        for name, operation in OPERATIONS.items():
            with pytest.raises(InvalidStorageKeyError):
                operation(store, key)
    assert list(root.iterdir()) == []                    # nothing was created inside the root ...
    assert list(root.parent.iterdir()) == [root]         # ... and nothing outside it


def test_a_resolved_path_outside_the_root_is_rejected(root, store):
    with pytest.raises(InvalidStorageKeyError):
        LocalStorage._ensure_inside(root.resolve(), root.resolve().parent / "elsewhere.jpg")
    assert LocalStorage._ensure_inside(root.resolve(), root.resolve() / "inside.jpg")


def test_a_write_is_atomic_and_leaves_no_temporary_files(store, root, monkeypatch):
    store.save("a/b.jpg", b"original", "image/jpeg")
    assert sorted(path.name for path in (root / "a").iterdir()) == ["b.jpg"]

    def failing_replace(source, destination):
        raise OSError("simulated failure")

    monkeypatch.setattr("app.storage.local.os.replace", failing_replace)
    with pytest.raises(StorageWriteError):
        store.save("a/b.jpg", b"replacement that never lands", "image/jpeg")
    monkeypatch.undo()

    assert store.read("a/b.jpg") == b"original"                          # the old content is intact
    assert sorted(path.name for path in (root / "a").iterdir()) == ["b.jpg"]   # no .tmp-*.part left behind


def test_errors_never_contain_paths_or_keys(store, root):
    key = "cases/private-key-part/file.jpg"
    messages = []
    with pytest.raises(StorageObjectNotFoundError) as missing:
        store.read(key)
    messages.append(str(missing.value))
    with pytest.raises(InvalidStorageKeyError) as invalid:
        store.read("../" + key)
    messages.append(str(invalid.value))
    (root / "blocked").write_bytes(b"a file where a directory is needed")
    with pytest.raises(StorageWriteError) as failed:
        store.save("blocked/inside.jpg", b"x", "image/jpeg")
    messages.append(str(failed.value))
    for message in messages:
        assert str(root) not in message
        assert "private-key-part" not in message
        assert str(root.parent) not in message


def test_get_storage_uses_the_configured_folder():
    storage = get_storage()
    assert isinstance(storage, LocalStorage)
    storage.save("probe/probe.bin", b"1", "application/octet-stream")
    assert (Path(get_settings().storage_dir) / "probe" / "probe.bin").is_file()


def test_get_storage_returns_the_same_instance_every_time():
    assert get_storage() is get_storage()
