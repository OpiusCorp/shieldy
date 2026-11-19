"""TensorFlow-dependent unit tests verifying TFLite model contract, gating, and tail scheduling."""

import os
import math
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pytest

from eval.class_map import YamnetClassMap
from eval.classifier import YamnetClassifier
from eval.detector_pipeline import AudioDetectorPipeline
from eval.synthetic_audio import SyntheticAudioGenerator
from eval.eval_harness import EvaluationHarness


def test_yamnet_model_tensor_contract():
    """Verify bundled yamnet.tflite input [15600] and output [1, 521] shapes."""
    classifier = YamnetClassifier()
    assert len(classifier.input_details) == 1
    assert len(classifier.output_details) == 1
    assert classifier.input_details[0]["shape"].tolist() == [15600]
    assert classifier.output_details[0]["shape"].tolist() == [1, 521]

    # Verify pure sine tone is classified as 'Sine wave' with high confidence
    t = np.linspace(0, 15600 / 16000.0, 15600, endpoint=False)
    sine_440 = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    res = classifier.classify(sine_440)
    assert res is not None
    assert res.top_label == "Sine wave"
    assert res.top_score > 0.90


def test_class_map_label_parsing():
    """Verify class map CSV parsing with quotes, commas, and gunshot subset selection."""
    cm = YamnetClassMap()
    assert len(cm) == 521
    assert cm.get_label(1) == "Child speech, kid speaking"

    tightened = cm.get_indices("tightened")
    assert tightened == [421, 422, 423, 424]

    legacy = cm.get_indices("legacy")
    assert 420 in legacy  # Explosion
    assert 460 in legacy  # Bang
    assert 420 not in tightened
    assert 460 not in tightened


def test_tier1_activity_gate_mechanics():
    """Verify Tier 1 activity gate on quiet noise, impulsive bursts, and clipping."""
    pipeline = AudioDetectorPipeline()

    quiet_frame = np.random.normal(0, 30.0, 2048).astype(np.int16)
    for _ in range(35):
        pipeline.process_frame(quiet_frame, 100.0)

    assert not pipeline.current_gate_open

    # Impulsive frame
    impulse_frame = np.copy(quiet_frame)
    impulse_frame[100:150] = 25000
    pipeline.process_frame(impulse_frame, 200.0)
    assert pipeline.current_gate_open

    # Mic clipping frame
    clipped_frame = np.copy(quiet_frame)
    clipped_frame[50:100] = 32767
    pipeline.process_frame(clipped_frame, 300.0)
    assert pipeline.current_gate_open


def test_hard_negative_rejection():
    """Verify tightened mode rejects Explosion/Bang from gunshot confidence calculation."""
    generator = SyntheticAudioGenerator(sample_rate=16000, seed=42)
    slam_16k = generator.generate_door_slam(duration_sec=0.975, peak_level=28000.0)

    clf = YamnetClassifier(mode="legacy")
    res_leg = clf.classify(slam_16k)
    assert res_leg.best_gunshot_label == "Explosion"
    # Legacy includes Explosion as gunshot, exceeding the 0.10 production threshold
    assert res_leg.gunshot_confidence >= 0.10

    clf.set_mode("tightened")
    res_tight = clf.classify(slam_16k)
    assert res_tight.best_gunshot_label != "Explosion"
    assert res_tight.best_gunshot_label != "Bang"
    assert res_leg.gunshot_confidence >= res_tight.gunshot_confidence

    # Speech formants produce 0 gunshot confidence
    shout_16k = generator.generate_speech_shout(duration_sec=0.975, peak_level=20000.0)
    res_shout = clf.classify(shout_16k)
    assert res_shout.gunshot_confidence < 0.01


def test_post_impulse_tail_scheduling():
    """Verify that post-impulse tail mode waits for tail arrival before classifying."""
    generator = SyntheticAudioGenerator(sample_rate=44100, seed=42)
    pipeline = AudioDetectorPipeline(
        classifier_mode="tightened",
        timing_mode="post_impulse_tail",
        tail_wait_ms=300.0,
    )
    harness = EvaluationHarness(pipeline=pipeline)

    pcm, events = generator.generate_boundary_evaluation_suite()
    metrics = harness.evaluate_stream(pcm, events)

    assert metrics.threat_events_count == len(events)
    assert metrics.event_recall_pct == 100.0
    assert metrics.true_positives == len(events)
    assert metrics.false_negatives == 0
    # Latency should reflect the tail wait (~300ms + frame buffer read)
    assert metrics.latency_p50_ms >= 200.0


def test_scheduler_impulse_then_silence_latches_inference():
    """Verify that a brief 5ms impulse latches pending tail wait even after gate closes."""
    pipeline = AudioDetectorPipeline(
        classifier_mode="tightened",
        timing_mode="post_impulse_tail",
        tail_wait_ms=300.0,
    )

    # Pre-fill history buffer with 1.0s of quiet audio
    quiet_frame = np.zeros(2048, dtype=np.int16)
    for i in range(25):
        pipeline.process_frame(quiet_frame, 100.0 + (i * 46.4))

    # Brief impulsive spike in one frame at t=1500ms
    impulse_frame = np.copy(quiet_frame)
    impulse_frame[10:30] = 28000
    alert_during_impulse = pipeline.process_frame(impulse_frame, 1500.0)

    # Inference should NOT run immediately; pending_tail_active should be latched
    assert alert_during_impulse is None
    assert pipeline.pending_tail_active is True
    assert math.isclose(pipeline.tail_target_time_ms, pipeline.first_impulse_at_ms + 300.0, rel_tol=1e-5)

    # Following frames are completely silent (gate closes)
    pipeline.process_frame(quiet_frame, 1550.0)
    assert pipeline.current_gate_open is False  # Gate closed!
    assert pipeline.pending_tail_active is True  # Latched through gate closure!

    pipeline.process_frame(quiet_frame, 1650.0)
    assert pipeline.pending_tail_active is True

    # Advance to t=1850ms (> 1800ms tail target)
    alert_after_tail = pipeline.process_frame(quiet_frame, 1850.0)
    # Latch was serviced!
    assert pipeline.pending_tail_active is False


def test_scheduler_sustained_impulses_timeout_bounded():
    """Verify that sustained impulses do not push timeout deadline indefinitely."""
    pipeline = AudioDetectorPipeline(
        classifier_mode="tightened",
        timing_mode="post_impulse_tail",
        tail_wait_ms=300.0,
        max_sustained_wait_ms=450.0,
    )

    quiet_frame = np.zeros(2048, dtype=np.int16)
    for i in range(25):
        pipeline.process_frame(quiet_frame, 100.0 + (i * 46.4))

    # First impulse at t=2000ms
    impulse_frame = np.copy(quiet_frame)
    impulse_frame[10:30] = 28000
    pipeline.process_frame(impulse_frame, 2000.0)

    initial_deadline = pipeline.tail_deadline_ms
    assert math.isclose(initial_deadline, pipeline.first_impulse_at_ms + 450.0, rel_tol=1e-5)

    # 10 continuous subsequent impulses every 40ms
    for step in range(1, 10):
        t = 2000.0 + (step * 40.0)
        pipeline.process_frame(impulse_frame, t)
        # Deadline MUST remain anchored to first impulse (2600ms)
        assert pipeline.tail_deadline_ms == initial_deadline
