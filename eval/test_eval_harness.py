"""Unit and regression tests for acoustic detector evaluation harness."""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pytest

from eval.class_map import YamnetClassMap
from eval.classifier import YamnetClassifier
from eval.detector_pipeline import AudioDetectorPipeline
from eval.synthetic_audio import SyntheticAudioGenerator, GroundTruthEvent
from eval.eval_harness import EvaluationHarness
from eval.manifest import EvaluationManifest, ManifestRecording


def test_class_map_parsing():
    cm = YamnetClassMap()
    assert len(cm) == 521
    # Verify label 1 is intact with commas preserved
    assert cm.get_label(1) == "Child speech, kid speaking"

    tightened = cm.get_indices("tightened")
    assert tightened == [421, 422, 423, 424]
    assert cm.get_label(421) == "Gunshot, gunfire"
    assert cm.get_label(422) == "Machine gun"
    assert cm.get_label(423) == "Fusillade"
    assert cm.get_label(424) == "Artillery fire"

    legacy = cm.get_indices("legacy")
    assert legacy == [420, 421, 422, 423, 424, 425, 428, 460]
    # Bang and Explosion are in legacy but excluded from tightened
    assert 420 in legacy and 420 not in tightened
    assert 460 in legacy and 460 not in tightened
    assert cm.get_label(460) == "Bang"


def test_yamnet_model_io():
    classifier = YamnetClassifier()
    assert len(classifier.input_details) == 1
    assert len(classifier.output_details) == 1
    assert classifier.input_details[0]["shape"].tolist() == [15600]
    assert classifier.output_details[0]["shape"].tolist() == [1, 521]

    # Test sine wave classification
    t = np.linspace(0, 15600 / 16000.0, 15600, endpoint=False)
    sine_440 = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    res = classifier.classify(sine_440)
    assert res is not None
    assert res.top_label == "Sine wave"
    assert res.top_score > 0.90


def test_tier1_activity_gate():
    pipeline = AudioDetectorPipeline()

    # 1. Fill history buffer with quiet ambient (RMS ~30)
    quiet_frame = np.random.normal(0, 30.0, 2048).astype(np.int16)
    for _ in range(35):  # fill > 15600 16k samples
        pipeline.process_frame(quiet_frame, 100.0)

    assert not pipeline.current_gate_open

    # 2. Impulsive burst
    impulse_frame = np.copy(quiet_frame)
    impulse_frame[100:150] = 25000  # peak > 0.25 (76% full scale)
    pipeline.process_frame(impulse_frame, 200.0)
    assert pipeline.current_gate_open

    # 3. Mic clipping frame
    clipped_frame = np.copy(quiet_frame)
    clipped_frame[50:100] = 32767  # 50 / 2048 > 0.005 clip fraction
    pipeline.process_frame(clipped_frame, 300.0)
    assert pipeline.current_gate_open


def test_window_boundary_coverage():
    generator = SyntheticAudioGenerator()
    pipeline = AudioDetectorPipeline(
        classifier_mode="tightened", window_strategy="retrospective"
    )
    harness = EvaluationHarness(pipeline=pipeline)

    pcm, events = generator.generate_boundary_evaluation_suite()
    metrics = harness.evaluate_stream(pcm, events)

    assert metrics.threat_events_count == len(events)
    assert metrics.event_recall_pct == 100.0
    assert metrics.true_positives == len(events)
    assert metrics.false_negatives == 0
    # Latency should be within acceptable bounds (< 250ms)
    assert metrics.latency_p50_ms < 250.0
    assert metrics.latency_p90_ms < 300.0


def test_hard_negative_door_slam_rejection():
    # Verify directly on model inference: legacy treats Explosion / Bang as gunshot,
    # while tightened mode strictly excludes them.
    generator = SyntheticAudioGenerator(sample_rate=16000)
    slam_16k = generator.generate_door_slam(duration_sec=0.975, peak_level=28000.0)

    clf = YamnetClassifier(mode="legacy")
    res_leg = clf.classify(slam_16k)
    # Legacy includes Explosion as gunshot
    assert res_leg.best_gunshot_label == "Explosion"
    assert res_leg.gunshot_confidence > 0.15

    # Tightened mode rejects Explosion/Bang from gunshot confidence calculation
    clf.set_mode("tightened")
    res_tight = clf.classify(slam_16k)
    assert res_tight.best_gunshot_label != "Explosion"
    assert res_tight.best_gunshot_label != "Bang"
    # Legacy gunshot score is strictly higher or equal due to including Explosion/Bang
    assert res_leg.gunshot_confidence >= res_tight.gunshot_confidence

    # Speech and dropped book have virtually 0 gunshot confidence
    shout_16k = generator.generate_speech_shout(duration_sec=0.975, peak_level=20000.0)
    res_shout = clf.classify(shout_16k)
    assert res_shout.gunshot_confidence < 0.01

    book_16k = generator.generate_dropped_book(duration_sec=0.975, peak_level=24000.0)
    res_book = clf.classify(book_16k)
    assert res_book.gunshot_confidence < 0.01


def test_manifest_roundtrip(tmp_path):
    manifest_path = str(tmp_path / "test_manifest.json")
    evt = GroundTruthEvent(
        event_id="evt_01",
        label="gunshot",
        start_sec=1.5,
        end_sec=2.0,
        is_threat=True,
        boundary_offset_ms=50.0,
        metadata={"caliber": "9mm"},
    )
    rec = ManifestRecording(
        recording_id="rec_01",
        audio_path="/fake/path.wav",
        duration_sec=10.0,
        sample_rate=44100,
        environment="firing_range",
        events=[evt],
        is_synthetic=True,
    )
    m = EvaluationManifest([rec])
    m.save_json(manifest_path)

    loaded = EvaluationManifest.load_json(manifest_path)
    assert len(loaded.recordings) == 1
    loaded_rec = loaded.recordings[0]
    assert loaded_rec.recording_id == "rec_01"
    assert len(loaded_rec.events) == 1
    assert loaded_rec.events[0].label == "gunshot"
    assert loaded_rec.events[0].boundary_offset_ms == 50.0
