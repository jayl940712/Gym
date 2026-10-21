# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run these tests in the isolated scoring venv; no Gym imports are required."""

import importlib.util
import json
import multiprocessing
from pathlib import Path

import pytest


np = pytest.importorskip("numpy")
pytest.importorskip("cv2")
Image = pytest.importorskip("PIL.Image")
ImageDraw = pytest.importorskip("PIL.ImageDraw")
SPEC = importlib.util.spec_from_file_location(
    "native_scorer", Path(__file__).resolve().parents[3] / "responses_api_agents/interaction2code_agent/scorer.py"
)
scorer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scorer)


def _hold_scoring_slot(limit, connection, release):
    with scorer.scoring_slot(limit):
        connection.send("entered")
        release.wait(10)


def test_scoring_slots_bound_process_concurrency(tmp_path, monkeypatch):
    monkeypatch.setattr(scorer, "MODEL_CACHE", tmp_path / "models")
    context = multiprocessing.get_context("fork")
    received, sent = context.Pipe(duplex=False)
    release = context.Event()
    release.set()
    process = context.Process(target=_hold_scoring_slot, args=(2, sent, release))
    try:
        with scorer.scoring_slot(2), scorer.scoring_slot(2):
            process.start()
            assert not received.poll(0.3)
        assert received.poll(5)
        assert received.recv() == "entered"
        process.join(5)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
            process.join(5)


def test_killed_scorer_releases_slot(tmp_path, monkeypatch):
    monkeypatch.setattr(scorer, "MODEL_CACHE", tmp_path / "models")
    context = multiprocessing.get_context("fork")
    first_received, first_sent = context.Pipe(duplex=False)
    second_received, second_sent = context.Pipe(duplex=False)
    hold, release = context.Event(), context.Event()
    release.set()
    first = context.Process(target=_hold_scoring_slot, args=(1, first_sent, hold))
    second = context.Process(target=_hold_scoring_slot, args=(1, second_sent, release))
    try:
        first.start()
        assert first_received.poll(5)
        assert first_received.recv() == "entered"
        second.start()
        assert not second_received.poll(0.3)
        first.kill()
        first.join(5)
        assert second_received.poll(5)
        assert second_received.recv() == "entered"
        second.join(5)
        assert second.exitcode == 0
    finally:
        for process in (first, second):
            if process.is_alive():
                process.kill()
                process.join(5)


def test_scoring_slot_rejects_nonpositive_limit():
    with pytest.raises(ValueError, match="positive"), scorer.scoring_slot(0):
        pass


def test_real_clip_and_ocr_with_provisioned_weights(tmp_path, monkeypatch):
    cache = Path(__file__).resolve().parents[1] / ".cache/scorer/models"
    if not (cache / "clip/ViT-B-32.pt").exists():
        pytest.skip("Provision the scoring runtime to test real metric weights")
    monkeypatch.setattr(scorer, "MODEL_CACHE", cache)
    scorer.clip_model.cache_clear()
    scorer.ocr_reader.cache_clear()
    path = tmp_path / "blank.png"
    Image.new("RGB", (64, 64), "white").save(path)
    assert scorer.clip_similarity(path, path) == pytest.approx(1.0, abs=1e-6)
    assert scorer.image_text(path) == ""
    scorer.clip_model.cache_clear()
    scorer.ocr_reader.cache_clear()


def test_native_ssim_and_first_red_contour(tmp_path):
    image = Image.new("RGB", (100, 100), "white")
    ImageDraw.Draw(image).rectangle((10, 20, 30, 40), outline="red", width=3)
    path = tmp_path / "reference.png"
    image.save(path)
    assert scorer.ssim_similarity(path, path) == 1
    # The first contour is the outer annotation box; preserve integer center rules.
    assert scorer.annotated_position(path) == (0.2, 0.3)
    Image.new("RGB", (100, 100), "white").save(path)
    with pytest.raises(ValueError, match="No red"):
        scorer.annotated_position(path)


@pytest.mark.parametrize(
    "reference,candidate,expected",
    [("", "", 1.0), ("", "text", None), ("text", "", None), (" abcdef", " abcdef", 1.0)],
)
def test_native_character_bleu_and_empty_text(monkeypatch, reference, candidate, expected):
    values = iter([reference, candidate])
    monkeypatch.setattr(scorer, "image_text", lambda _, **kwargs: next(values))
    assert scorer.text_similarity(Path("a"), Path("b")) == expected


def test_native_changed_region_excludes_maximum_pixel(tmp_path):
    before, after, crop = tmp_path / "before.png", tmp_path / "after.png", tmp_path / "crop.png"
    Image.new("RGB", (100, 100), "white").save(before)
    image = Image.open(before)
    ImageDraw.Draw(image).rectangle((10, 20, 30, 40), fill="black")
    image.save(after)
    assert scorer.interaction_crop(before, after, crop) == (20.0, 30.0)
    assert Image.open(crop).size == (20, 20)


def test_different_dimensions_do_not_depend_on_git_worktree():
    a = np.zeros((10, 5), dtype=np.uint8)
    b = np.zeros((15, 5), dtype=np.uint8)
    lo, hi = scorer.difference_bounds(a, b)
    assert 1 <= lo < hi
    assert scorer.difference_bounds(a, np.zeros((15, 8))) == (0, 15)
    with pytest.raises(ValueError, match="no changed"):
        scorer.difference_bounds(a, a)


@pytest.mark.parametrize(
    "case,flag", [("valid", True), ("same", False), ("missing", False), ("many", False), ("html", False)]
)
def test_native_flag_and_metric_fields(tmp_path, monkeypatch, case, flag):
    dataset, task, artifacts = tmp_path / "data", tmp_path / "task", tmp_path / "artifacts"
    reference = dataset / "source/1"
    reference.mkdir(parents=True)
    task.mkdir()
    artifacts.mkdir()
    Image.new("RGB", (100, 100), "white").save(artifacts / "0_source.png")
    image = Image.open(artifacts / "0_source.png")
    if case != "same":
        ImageDraw.Draw(image).rectangle((10, 20, 30, 40), fill="black")
    image.save(artifacts / "1_20_30_click.png")
    names = ["0_source.png", "1_20_30_click.png"]
    if case == "missing":
        names = names[:1]
    if case == "many":
        image.save(artifacts / "2_40_60_click.png")
        names.append("2_40_60_click.png")
    (artifacts / "render.json").write_text(json.dumps({"status": "ok", "screenshots": names}))
    (artifacts / "index.html").write_text("<html>" if case == "html" else "<html></html>")
    refs = {"after": "after.png", "before_mark": "before_mark.png", "after_mark": "after_mark.png", "crop": "crop.png"}
    (task / "task.json").write_text(json.dumps({"page_id": 1, "references": refs}))
    monkeypatch.setattr(scorer, "clip_similarity", lambda a, b: 0.8)
    monkeypatch.setattr(scorer, "ssim_similarity", lambda a, b: 0.7)
    monkeypatch.setattr(scorer, "text_similarity", lambda a, b: 0.6)
    monkeypatch.setattr(scorer, "annotated_position", lambda _: (0.2, 0.3))
    result = scorer.evaluate(dataset=dataset, task=task, artifacts=artifacts)
    assert result["interaction_flag"] is flag
    assert result["reward"] == (0.8 if flag else 0)
    assert set(result["interaction"]) == {
        "clip_similarity",
        "structure_similarity",
        "text_similarity",
        "position_similarity",
        "position_similarity_after",
    }
    # Full-page CLIP remains populated for invalid interactions, matching upstream.
    assert result["full_page"]["clip_similarity"] == (0 if case == "missing" else 0.8)


def test_reference_ocr_cache_uses_content_and_preserves_exact_text(tmp_path, monkeypatch):
    monkeypatch.setattr(scorer, "MODEL_CACHE", tmp_path / "models")
    monkeypatch.setattr(scorer, "ocr_cache_profile", lambda *args: "profile-a")
    calls = []
    monkeypatch.setattr(scorer, "_image_text", lambda p: calls.append(p.read_bytes()) or " Exact OCR Text")
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    first.write_bytes(b"same image")
    second.write_bytes(first.read_bytes())
    assert scorer.image_text(first, cache=True) == " Exact OCR Text"
    assert scorer.image_text(second, cache=True) == " Exact OCR Text"
    assert len(calls) == 1
    second.write_bytes(b"changed image")
    scorer.image_text(second, cache=True)
    assert len(calls) == 2
    scorer.image_text(first)
    assert len(calls) == 3  # Generated images still get fresh OCR.
    monkeypatch.setattr(scorer, "ocr_cache_profile", lambda *args: "profile-b")
    scorer.image_text(first, cache=True)
    assert len(calls) == 4
    monkeypatch.setenv("I2C_OCR_CACHE", "0")
    scorer.image_text(first, cache=True)
    assert len(calls) == 5


@pytest.mark.parametrize("text", ["", " Hello"])
def test_corrupt_reference_cache_is_recomputed(tmp_path, monkeypatch, text):
    monkeypatch.setattr(scorer, "MODEL_CACHE", tmp_path / "models")
    monkeypatch.setattr(scorer, "ocr_cache_profile", lambda *args: "profile")
    calls = []
    monkeypatch.setattr(scorer, "_image_text", lambda _: calls.append(1) or text)
    image = tmp_path / "image.png"
    image.write_bytes(b"PNG")
    assert scorer.image_text(image, cache=True) == text
    cache = next((tmp_path / "models/ocr-text").glob("*/*.json"))
    cache.write_text("invalid json")
    assert scorer.image_text(image, cache=True) == text
    assert scorer.image_text(image, cache=True) == text
    assert len(calls) == 2


def test_reference_ocr_cache_falls_back_when_unwritable(tmp_path, monkeypatch):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x")
    monkeypatch.setattr(scorer, "MODEL_CACHE", blocker)
    monkeypatch.setattr(scorer, "ocr_cache_profile", lambda *args: "profile")
    monkeypatch.setattr(scorer, "_image_text", lambda _: " text")
    assert scorer.image_text(blocker, cache=True) == " text"


def test_workers_share_one_reference_ocr_computation(tmp_path, monkeypatch):
    monkeypatch.setattr(scorer, "MODEL_CACHE", tmp_path / "models")
    monkeypatch.setattr(scorer, "ocr_cache_profile", lambda *args: "profile")
    count = tmp_path / "calls"
    image = tmp_path / "image.png"
    image.write_bytes(b"PNG")

    def compute(_):
        with count.open("a") as stream:
            stream.write("computed\n")
        return " same text"

    monkeypatch.setattr(scorer, "_image_text", compute)
    context = multiprocessing.get_context("fork")
    processes = [context.Process(target=scorer.image_text, args=(image,), kwargs={"cache": True}) for _ in range(3)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(10)
        assert process.exitcode == 0
    assert count.read_text() == "computed\n"


def test_ocr_profile_invalidates_for_weights_versions_threads_and_source(tmp_path, monkeypatch):
    source = tmp_path / "scorer.py"
    source.write_text("source-v1")
    weights = tmp_path / "easyocr"
    weights.mkdir()
    checkpoint = weights / "english_g2.pth"
    checkpoint.write_bytes(b"weights-v1")
    monkeypatch.setattr(scorer, "__file__", str(source))
    monkeypatch.setattr(scorer.importlib.metadata, "version", lambda _: "version-1")

    def fingerprint(threads="2"):
        scorer.ocr_cache_profile.cache_clear()
        return scorer.ocr_cache_profile(tmp_path, threads)

    initial = fingerprint()
    checkpoint.write_bytes(b"weights-v2")
    changed_weights = fingerprint()
    assert changed_weights != initial
    assert fingerprint("4") != changed_weights
    monkeypatch.setattr(scorer.importlib.metadata, "version", lambda _: "version-2")
    changed_version = fingerprint()
    assert changed_version != changed_weights
    source.write_text("source-v2")
    assert fingerprint() != changed_version
    scorer.ocr_cache_profile.cache_clear()
