# EchoShield Acoustic Detector Evaluation & Scoring Harness

## Overview

This package provides a reproducible, end-to-end audio replay, evaluation, and scoring harness for the EchoShield acoustic gunshot detection pipeline. It simulates the two-tier mobile detector implemented in [`AudioSensorService.kt`](file:///Users/nick/shieldy/app/src/main/java/com/echoshield/echonode/service/AudioSensorService.kt) and [`LegacyTfLiteGunshotClassifier.kt`](file:///Users/nick/shieldy/app/src/main/java/com/echoshield/echonode/service/LegacyTfLiteGunshotClassifier.kt), executing inference directly against the bundled [`yamnet.tflite`](file:///Users/nick/shieldy/app/src/main/assets/yamnet.tflite) neural network asset and [`yamnet_class_map.csv`](file:///Users/nick/shieldy/app/src/main/assets/yamnet_class_map.csv).

> [!IMPORTANT]
> **Synthetic Checks vs. Field Gunshot Accuracy**:
> The benchmark suites below use calibrated synthetic acoustic waveforms to systematically verify pipeline timing, boundary coverage, and negative rejection under identical bitstreams. They do **not** represent field accuracy on physical firearms. Field evaluation requires labeled acoustic field recordings loaded via the manifest interface.

---

## Metric Definitions & Policy

1. **Event Recall**:
   $$\text{Recall} = \frac{\text{True Positives}}{\text{Total Threat Events}} \times 100\%$$
   - **One-to-One Matching Policy**: Each threat event can be claimed by at most one primary alert within its active window `[start_sec, end_sec + max_latency_allowance]`.
   - **Duplicate Alert Policy**: Additional alerts that fire during the same active threat event window are recorded as `duplicate_alerts` (neither double-counted as TPs nor penalised as FPs).
2. **False Alerts Per Hour**:
   $$\text{Rate} = \frac{\text{False Positives}}{\text{Negative Audio Hours}}$$
   - Evaluated on a 46.0s hard-negative stress corpus containing 7 challenging transient sounds (door slams, claps, pops, shouts).
   - Because transients occur every 6 seconds, the hourly rate reflects continuous exposure to high-density transient impulse noise.
3. **Detection Latency**:
   $$\text{Latency} = t_{\text{trigger}} - t_{\text{ground\_truth\_impulse\_onset}}$$
   - Measures elapsed wall-clock milliseconds from the exact sample of impulse onset to detection firing.
   - Includes frame buffering delay (~46.4ms for 2048-sample reads at 44.1kHz), plus the explicit tail collection delay (~300ms) when post-impulse tail scheduling is enabled.
4. **Execution Mode**:
   - **TFLite Model**: All configurations execute real float32 inference using the bundled `app/src/main/assets/yamnet.tflite` model (`[15600]` input, `[1, 521]` output). Zero mocks are used for ML classification.
   - **Pipeline Simulation**: The capture loop, ring buffer, and tail-wait scheduling simulate `AudioSensorService.kt` and `DetectionScheduler.kt` in Python; they evaluate algorithmic timing, buffering, and label selection, but do not represent physical on-device Android execution.

---

## Directory & Test Structure

```text
eval/
├── __init__.py                # Package initialization with environment guards
├── requirements.txt           # Explicit dependencies (numpy, scipy, tensorflow, soundfile, pytest)
├── class_map.py               # Robust CSV class map parser (handles quotes/commas)
├── classifier.py              # TFLite YAMNet wrapper (matches [15600] in, [1, 521] out)
├── detector_pipeline.py       # Full AudioSensorService simulation (Tier 1 gate + Tier 2 ML + tail wait)
├── synthetic_audio.py         # Deterministic seeded synthetic generator & SHA-256 hasher
├── eval_harness.py            # Streaming audio replay, one-to-one matching, and scoring engine
├── run_eval.py                # Command-line benchmark runner with identical-corpus replay
├── test_scoring.py            # Model-independent tests (scoring, matching, duplicate policy, hashing)
├── test_inference.py          # TensorFlow-dependent tests (TFLite shapes, activity gate, tail timing)
├── test_eval_harness.py       # Master test suite runner
├── sample_manifest.json       # Example labeled recording manifest
└── raw_benchmark_results.json # Persisted raw output JSON from identical corpus replay
```

---

## Runnable Commands

### 1. Model-Independent Scoring Tests (Runs without TensorFlow)
```bash
python3 -m pytest eval/test_scoring.py -s -o cache_dir=/tmp/pytest_cache -v
```

### 2. Full Test Suite (Including TensorFlow TFLite Model Checks)
```bash
python3 -m pytest eval/test_inference.py -s -o cache_dir=/tmp/pytest_cache -v
```

### 3. Run Comparative Evaluation Matrix (Identical Corpus Replay)
```bash
python3 -m eval.run_eval --compare
```

### 4. Run Single Mode with JSON Output
```bash
python3 -m eval.run_eval --mode tightened --output eval/raw_benchmark_results.json
```

---

## Comparative Matrix (Identical Corpus Replay, Seed=42)

- **Corpus Seed**: `42`
- **Boundary Suite SHA-256**: `ec7537d470054658d5cf65c26b63ca00bbcd920257c770c0ba13689f92ae0755` (40.0s)
- **Hard Negatives SHA-256**: `3fb30f80bce382dcdaeead65b533f81e01861788c03792b0c36cb077d85e7834` (46.0s)
- **Tested Detector Head SHA**: Dynamic git provenance via `--commit-sha` (resolves to current `git rev-parse HEAD`)

| Configuration | Event Recall | False Alert Rate | Hard Neg FPs | Latency p50 | Latency p90 |
|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline Classifier** (Bang/Explosion/Cap gun, immediate) | 100.0% (6/6) | 391.3 / hr | 5 FPs | 26.4 ms | 43.5 ms |
| **Tightened Classifier** (Gunshots only `[421-424]`, immediate) | 100.0% (6/6) | 313.0 / hr | 4 FPs | 26.4 ms | 43.5 ms |
| **Tightened + Post-Impulse Tail Wait** (PR #6 timing) | 100.0% (6/6) | **78.3 / hr** | **1 FP** | 356.4 ms | 398.0 ms |
| **Tightened + Tail Wait + Polyphase Resampling** | 83.3% (5/6) | **0.0 / hr** | **0 FPs** | 368.4 ms | 398.8 ms |

### Hard Negative Breakdown by Event Type

| Negative Sound Type | Tested Events | Baseline (Legacy) FPs | Tightened FPs | Tightened + Tail Wait FPs |
|---|:---:|:---:|:---:|:---:|
| **door_slam** | 2 | 2 FPs (Explosion) | 1 FP | **0 FPs** |
| **hand_clap** | 2 | 2 FPs | 2 FPs | 1 FP |
| **balloon_pop** | 1 | 1 FP | 1 FP | **0 FPs** |
| **dropped_book** | 1 | 0 FPs | 0 FPs | 0 FPs |
| **speech_shout** | 1 | 0 FPs | 0 FPs | 0 FPs |

---

## Field Manifest Schema

For physical field evaluation on recorded datasets (UrbanSound8K, AudioSet, firing range recordings), format files according to [`eval/sample_manifest.json`](file:///Users/nick/shieldy/eval/sample_manifest.json):

```json
{
  "version": "1.0",
  "recordings": [
    {
      "recording_id": "indoor_range_9mm_01",
      "audio_path": "/path/to/recording.wav",
      "duration_sec": 30.5,
      "sample_rate": 44100,
      "environment": "indoor_corridor",
      "events": [
        {
          "event_id": "shot_01",
          "label": "gunshot",
          "start_sec": 4.12,
          "end_sec": 4.65,
          "is_threat": true
        }
      ]
    }
  ]
}
```
