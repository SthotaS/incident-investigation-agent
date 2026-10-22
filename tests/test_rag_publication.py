"""Private-data checks for atomic publication and competing local index builds."""

import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from filelock import FileLock
from incident_server import rag


class TinyEmbeddings:
    def __init__(self, offset=0):
        self.offset = offset

    def passage_embed(self, documents):
        for position, _ in enumerate(documents):
            vector = np.zeros(rag.EMBEDDING_DIMENSION, dtype=np.float32)
            vector[position + self.offset] = 1
            yield vector

    def query_embed(self, query):
        vector = np.zeros(rag.EMBEDDING_DIMENSION, dtype=np.float32)
        vector[self.offset] = 1
        yield vector


@pytest.fixture
def local_index(tmp_path, monkeypatch):
    """Do not open the user's database, real index, embedding model, or API."""
    seeds = rag.load_seed_cases()
    monkeypatch.setattr(rag, "load_cases", lambda: seeds)
    monkeypatch.setattr(rag, "_embedding_model", lambda *args, **kwargs: TinyEmbeddings())
    monkeypatch.setenv("INCIDENT_RAG_DIR", str(tmp_path))
    rag._cached_index.cache_clear()
    yield tmp_path, seeds
    rag._cached_index.cache_clear()


def test_build_publishes_complete_generation(local_index):
    directory, seeds = local_index
    rag.build_index(directory)
    manifest = rag.GenerationManifest.model_validate_json(
        (directory / rag.CURRENT_FILE).read_bytes()
    )
    generation = directory / rag.GENERATIONS_DIR / manifest.generation
    assert (generation / rag.INDEX_FILE).is_file()
    assert (generation / rag.METADATA_FILE).is_file()
    assert not (directory / rag.INDEX_FILE).exists()
    loaded = rag.IncidentIndex.load(directory)
    assert len(loaded.cases) == len(seeds)
    assert loaded.search("settlement HTTP 503").matches[0].incident_id == "INC-09876"


@pytest.mark.parametrize("failed_name", [rag.METADATA_FILE, rag.CURRENT_FILE])
def test_failed_publish_preserves_previous_complete_generation(
    local_index, monkeypatch, failed_name
):
    directory, _ = local_index
    rag.build_index(directory)
    old_manifest = (directory / rag.CURRENT_FILE).read_bytes()
    old_fingerprint = rag.IncidentIndex.load(directory).metadata.fingerprint
    original_write = rag._atomic_write
    monkeypatch.setattr(rag, "_embedding_model", lambda *args, **kwargs: TinyEmbeddings(1))

    def failed_write(path, content):
        if path.name == failed_name:
            raise OSError("Synthetic interrupted publication")
        original_write(path, content)

    monkeypatch.setattr(rag, "_atomic_write", failed_write)
    with pytest.raises(rag.RagIndexError, match="Index build failed"):
        rag.build_index(directory)
    assert (directory / rag.CURRENT_FILE).read_bytes() == old_manifest
    assert rag.IncidentIndex.load(directory).metadata.fingerprint == old_fingerprint


def test_corpus_change_during_embedding_is_not_published(local_index, monkeypatch):
    directory, seeds = local_index
    rag.build_index(directory)
    old_manifest = (directory / rag.CURRENT_FILE).read_bytes()
    changed = False
    current = seeds

    def changing_model(*args, **kwargs):
        nonlocal current, changed
        if not changed:
            changed = True
            current = (
                *seeds,
                seeds[0].model_copy(
                    update={
                        "incident_id": "INC-00124",
                        "reference": "synthetic://incidents/INC-00124",
                    }
                ),
            )
        return TinyEmbeddings()

    monkeypatch.setattr(rag, "load_cases", lambda: current)
    monkeypatch.setattr(rag, "_embedding_model", changing_model)
    with pytest.raises(rag.RagIndexError, match="changed during"):
        rag.build_index(directory)
    assert (directory / rag.CURRENT_FILE).read_bytes() == old_manifest
    # The newer database correctly makes the previous generation stale; rebuilding fixes it.
    with pytest.raises(rag.RagIndexError, match="stale"):
        rag.IncidentIndex.load(directory)
    rag.build_index(directory)
    assert len(rag.IncidentIndex.load(directory).cases) == 6


def test_build_lock_is_taken_before_loading_cases(local_index, monkeypatch):
    directory, _ = local_index
    snapshots = []
    monkeypatch.setattr(rag, "load_cases", lambda: snapshots.append("read") or ())
    monkeypatch.setattr(rag, "BUILD_LOCK_TIMEOUT_SECONDS", 0.02)
    with (
        FileLock(str(directory / rag.BUILD_LOCK_FILE)),
        pytest.raises(rag.RagIndexError, match="Another index build"),
    ):
        rag.build_index(directory)
    assert snapshots == []
    assert not (directory / rag.CURRENT_FILE).exists()


def test_competing_builds_are_serialized(local_index, monkeypatch):
    directory, seeds = local_index
    first_entered = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    snapshots = []
    model_loads = []

    def load_cases():
        snapshots.append(threading.current_thread().name)
        return seeds

    def model_factory(*args, **kwargs):
        model_loads.append(threading.current_thread().name)
        if len(model_loads) == 1:
            first_entered.set()
            assert release_first.wait(3)
        return TinyEmbeddings()

    def second_build():
        second_started.set()
        return rag.build_index(directory)

    monkeypatch.setattr(rag, "load_cases", load_cases)
    monkeypatch.setattr(rag, "_embedding_model", model_factory)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(rag.build_index, directory)
        assert first_entered.wait(3)
        second = pool.submit(second_build)
        assert second_started.wait(3)
        # The second worker has started, but cannot take its corpus snapshot yet.
        assert len(snapshots) == 1
        release_first.set()
        assert first.result(timeout=3) == directory
        assert second.result(timeout=3) == directory
    assert len(model_loads) == 2
    assert len(snapshots) == 4
    assert len(rag.IncidentIndex.load(directory).cases) == 5


def install_legacy_pair(directory):
    metadata, binary = rag._index_files(directory)
    shutil.copyfile(metadata, directory / rag.METADATA_FILE)
    shutil.copyfile(binary, directory / rag.INDEX_FILE)
    (directory / rag.CURRENT_FILE).unlink()


def test_existing_legacy_index_loads_without_rebuild(local_index):
    directory, _ = local_index
    rag.build_index(directory)
    install_legacy_pair(directory)
    old_metadata = (directory / rag.METADATA_FILE).read_bytes()
    old_binary = (directory / rag.INDEX_FILE).read_bytes()
    assert len(rag.IncidentIndex.load(directory).cases) == 5
    rag.build_index(directory)
    assert (directory / rag.METADATA_FILE).read_bytes() == old_metadata
    assert (directory / rag.INDEX_FILE).read_bytes() == old_binary
    assert (directory / rag.CURRENT_FILE).is_file()


@pytest.mark.parametrize(
    "manifest",
    [
        b"not JSON",
        b"{}",
        b'{"generation":"../../outside"}',
        b'{"generation":"/tmp/outside"}',
        b'{"generation":"' + b"a" * 32 + b'","extra":true}',
        b" " * 4097,
    ],
)
def test_bad_manifest_never_falls_back_to_valid_legacy(local_index, manifest):
    directory, _ = local_index
    rag.build_index(directory)
    install_legacy_pair(directory)
    (directory / rag.CURRENT_FILE).write_bytes(manifest)
    with pytest.raises(rag.RagIndexError, match="damaged"):
        rag.IncidentIndex.load(directory)
    with pytest.raises(rag.RagIndexError, match="invalid"):
        rag.get_index()


@pytest.mark.parametrize("which", ["manifest", "generations", "generation", "binary", "metadata"])
def test_generation_rejects_symbolic_links(local_index, which):
    directory, _ = local_index
    rag.build_index(directory)
    metadata, binary = rag._index_files(directory)
    target = {
        "manifest": directory / rag.CURRENT_FILE,
        "generations": directory / rag.GENERATIONS_DIR,
        "generation": metadata.parent,
        "binary": binary,
        "metadata": metadata,
    }[which]
    moved = directory / f"moved-{which}"
    target.rename(moved)
    target.symlink_to(moved, target_is_directory=moved.is_dir())
    with pytest.raises(rag.RagIndexError, match="damaged"):
        rag.IncidentIndex.load(directory)


def test_new_manifest_changes_cached_index_signature(local_index, monkeypatch):
    directory, _ = local_index
    rag.build_index(directory)
    first = rag.get_index()
    monkeypatch.setattr(rag, "_embedding_model", lambda *args, **kwargs: TinyEmbeddings(1))
    rag.build_index(directory)
    second = rag.get_index()
    assert second is not first
    assert second.metadata.fingerprint != first.metadata.fingerprint


def test_actual_index_read_is_bounded(local_index):
    directory, _ = local_index
    rag.build_index(directory)
    metadata, _ = rag._index_files(directory)
    metadata.write_bytes(b"x" * (rag.MAX_METADATA_BYTES + 1))
    with pytest.raises(rag.RagIndexError, match="damaged"):
        rag.IncidentIndex.load(directory)


def test_build_lock_is_shared_with_another_process(local_index, monkeypatch):
    directory, _ = local_index
    ready = directory / "child-ready"
    child_code = (
        "import sys; from pathlib import Path; from filelock import FileLock; "
        "lock = FileLock(sys.argv[1]); lock.acquire(timeout=2); "
        "Path(sys.argv[2]).write_text('ready'); sys.stdin.read(1); lock.release()"
    )
    monkeypatch.setattr(rag, "BUILD_LOCK_TIMEOUT_SECONDS", 0.02)
    with subprocess.Popen(
        [sys.executable, "-c", child_code, str(directory / rag.BUILD_LOCK_FILE), str(ready)],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env={"PATH": os.defpath},
    ) as child:
        try:
            deadline = time.monotonic() + 3
            while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
                threading.Event().wait(0.01)
            assert ready.exists(), "The private lock-holder process did not start"
            with pytest.raises(rag.RagIndexError, match="Another index build"):
                rag.build_index(directory)
        finally:
            child.communicate(input=b"x", timeout=3)
    assert child.returncode == 0
    assert not (directory / rag.CURRENT_FILE).exists()
    rag.build_index(directory)
    assert len(rag.IncidentIndex.load(directory).cases) == 5


def test_build_rejects_a_symbolic_link_for_its_lock(local_index):
    directory, _ = local_index
    elsewhere = directory / "unrelated"
    elsewhere.write_text("preserve this file")
    (directory / rag.BUILD_LOCK_FILE).symlink_to(elsewhere)
    with pytest.raises(rag.RagIndexError, match="lock must not"):
        rag.build_index(directory)
    assert elsewhere.read_text() == "preserve this file"
    assert not (directory / rag.CURRENT_FILE).exists()
