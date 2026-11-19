"""Model-independent unit tests for scoring, event matching, and corpus repeatability.
Can run in any standard Python environment without TensorFlow.
"""

from typing import List
import numpy as np
import pytest

from eval.synthetic_audio import (
    SyntheticAudioGenerator,
    GroundTruthEvent,
    compute_audio_sha256,
)
from eval.eval_harness import EvaluationHarness, MetricSummary
from eval.detector_pipeline import DetectionAlert, AudioDetectorPipeline
from eval.manifest import EvaluationManifest, ManifestRecording


def test_identical_corpus_seed_hashing():
    """Verify that SyntheticAudioGenerator with fixed seed produces identical bitstreams."""
    gen1 = SyntheticAudioGenerator(sample_rate=44100, seed=42)
    pcm1, events1 = gen1.generate_boundary_evaluation_suite()
    sha1 = compute_audio_sha256(pcm1)

    gen2 = SyntheticAudioGenerator(sample_rate=44100, seed=42)
    pcm2, events2 = gen2.generate_boundary_evaluation_suite()
    sha2 = compute_audio_sha256(pcm2)

    assert sha1 == sha2
    assert np.array_equal(pcm1, pcm2)
    assert len(events1) == len(events2)


def test_one_to_one_alert_matching_overlapping_events():
    """Verify that a single alert cannot be claimed by two overlapping threat events."""
    harness = EvaluationHarness()

    events = [
        GroundTruthEvent(
            event_id="evt_01",
            label="gunshot",
            start_sec=1.0,
            end_sec=1.5,
            is_threat=True,
        ),
        GroundTruthEvent(
            event_id="evt_02",
            label="gunshot",
            start_sec=1.2,  # Overlaps with evt_01
            end_sec=1.7,
            is_threat=True,
        ),
    ]

    # Exactly ONE alert at 1.3s
    single_alert = DetectionAlert(
        timestamp_ms=1300.0,
        trigger_latency_ms=100.0,
        impulse_onset_ms=1200.0,
        confidence=0.85,
        top_label="Gunshot, gunfire",
        gunshot_label="Gunshot, gunfire",
        gate_reason="PEAK",
        effective_threshold=0.10,
        clip_fraction=0.0,
    )

    metrics = harness._score_results(
        total_duration_sec=5.0, gt_events=events, alerts=[single_alert]
    )

    # evt_01 matches alert (1 TP), evt_02 has no remaining alert (1 FN)
    assert metrics.threat_events_count == 2
    assert metrics.true_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.false_positives == 0
    assert metrics.event_recall_pct == 50.0


def test_duplicate_alerts_policy():
    """Verify that multiple alerts in one event count as 1 TP and N duplicate alerts."""
    harness = EvaluationHarness()

    events = [
        GroundTruthEvent(
            event_id="evt_01",
            label="gunshot",
            start_sec=1.0,
            end_sec=1.5,
            is_threat=True,
        )
    ]

    # Two alerts inside the same event window
    alerts = [
        DetectionAlert(
            timestamp_ms=1100.0,
            trigger_latency_ms=100.0,
            impulse_onset_ms=1000.0,
            confidence=0.85,
            top_label="Gunshot, gunfire",
            gunshot_label="Gunshot, gunfire",
            gate_reason="PEAK",
            effective_threshold=0.10,
            clip_fraction=0.0,
        ),
        DetectionAlert(
            timestamp_ms=1400.0,
            trigger_latency_ms=400.0,
            impulse_onset_ms=1000.0,
            confidence=0.75,
            top_label="Gunshot, gunfire",
            gunshot_label="Gunshot, gunfire",
            gate_reason="RMS",
            effective_threshold=0.10,
            clip_fraction=0.0,
        ),
    ]

    metrics = harness._score_results(
        total_duration_sec=5.0, gt_events=events, alerts=alerts
    )

    assert metrics.threat_events_count == 1
    assert metrics.true_positives == 1
    assert metrics.false_negatives == 0
    assert metrics.false_positives == 0
    assert metrics.duplicate_alerts == 1
    assert metrics.event_recall_pct == 100.0


def test_zero_events_and_zero_alerts():
    """Test boundary edge case: audio stream with 0 threat events and 0 alerts."""
    harness = EvaluationHarness()
    metrics = harness._score_results(
        total_duration_sec=10.0, gt_events=[], alerts=[]
    )

    assert metrics.threat_events_count == 0
    assert metrics.true_positives == 0
    assert metrics.false_negatives == 0
    assert metrics.false_positives == 0
    assert metrics.duplicate_alerts == 0
    assert metrics.event_recall_pct == 100.0
    assert metrics.false_alerts_per_hour == 0.0
    assert metrics.latency_p50_ms == 0.0


def test_wake_historical_rejection_rules():
    """Verify that classify_wake_historical strictly validates timestamp ranges."""
    # Dummy mock pipeline without loading TFLite
    pipeline = AudioDetectorPipeline.__new__(AudioDetectorPipeline)
    pipeline.SAMPLE_RATE = 44100
    pipeline.MODEL_SAMPLE_RATE = 16000
    pipeline.HISTORY_SAMPLES = 64000
    pipeline.history_samples_written = 64000
    pipeline.classifier = None

    # 1. Reject future timestamp
    assert pipeline.classify_wake_historical(event_timestamp_ms=5000.0, current_time_ms=4000.0) is None

    # 2. Reject expired timestamp (> 4000 ms old for 64000 samples @ 16k)
    assert pipeline.classify_wake_historical(event_timestamp_ms=1000.0, current_time_ms=6000.0) is None


def test_scheduler_early_peer_wake_request():
    """Verify early peer request returning PENDING_TAIL rather than voting NO."""
    pipeline = AudioDetectorPipeline.__new__(AudioDetectorPipeline)
    pipeline.SAMPLE_RATE = 44100
    pipeline.MODEL_SAMPLE_RATE = 16000
    pipeline.HISTORY_SAMPLES = 64000

    # Event at t=1000ms, peer request arrives early at t=1100ms, tail wait=300ms (needs t=1300ms)
    status, sched_time = pipeline.handle_peer_wake_request(
        event_timestamp_ms=1000.0, current_time_ms=1100.0, tail_ms=300.0
    )
    assert status == "PENDING_TAIL"
    assert sched_time == 1300.0

    # At t=1300ms, status transitions to READY
    status_ready, ready_time = pipeline.handle_peer_wake_request(
        event_timestamp_ms=1000.0, current_time_ms=1300.0, tail_ms=300.0
    )
    assert status_ready == "READY"
    assert ready_time == 1300.0


def test_scheduler_stale_impulse_expiration():
    """Verify stale impulse memory is cleared when elapsed time exceeds IMPULSE_MEMORY_MS."""
    pipeline = AudioDetectorPipeline.__new__(AudioDetectorPipeline)
    pipeline.SAMPLE_RATE = 44100
    pipeline.MODEL_SAMPLE_RATE = 16000
    pipeline.HISTORY_SAMPLES = 64000
    pipeline.history_buffer = np.zeros(64000, dtype=np.float32)
    pipeline.history_write_index = 0
    pipeline.history_samples_written = 64000
    pipeline.history_end_time_ms = 0.0
    pipeline.resample_method = "box_filter"
    pipeline.IMPULSE_MEMORY_MS = 3000.0
    pipeline.IMPULSE_ONSET_PEAK_NORMALIZED = 0.15
    pipeline.CLIP_MARGIN = 64
    pipeline.CLIP_FRACTION_GATE = 0.005
    pipeline.CLIP_FRACTION_SEVERE = 0.02
    pipeline.detection_threshold = 200.0
    pipeline.EMA_ALPHA = 0.4
    pipeline.smoothed_amplitude = 0.0
    pipeline.timing_mode = "baseline"
    pipeline.TRIGGER_COOLDOWN_MS = 5000.0
    pipeline.MODEL_INFERENCE_INTERVAL_MS = 400.0
    pipeline.last_trigger_time = -5000.0
    pipeline.last_model_inference = -400.0
    pipeline.pending_tail_active = False
    pipeline.classifier = None
    pipeline.MODEL_INPUT_SAMPLES = 15600
    pipeline.IMPULSIVE_PEAK_RMS_RATIO = 3.0

    # Stale impulse at 1000ms
    pipeline.last_impulse_at_ms = 1000.0

    # Quiet frame arrives at 5000ms (> 3000ms later)
    quiet_frame = np.zeros(2048, dtype=np.int16)
    pipeline.process_frame(quiet_frame, 5000.0)

    # Stale impulse must be cleared
    assert pipeline.last_impulse_at_ms == 0.0


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
    assert loaded.recordings[0].recording_id == "rec_01"
    assert loaded.recordings[0].events[0].label == "gunshot"
