# EchoShield Acoustic Detector Evaluation & Scoring Harness

## Overview

This directory provides a reproducible, end-to-end audio replay and evaluation harness for the EchoShield acoustic gunshot detection pipeline. It simulates the full two-tier mobile detector implemented in `AudioSensorService.kt` and `LegacyTfLiteGunshotClassifier.kt`, executing inference directly against the bundled `yamnet.tflite` model and `yamnet_class_map.csv`.

The harness evaluates:
- **Event Recall**: Detection rate across threat events.
- **False Alerts Per Hour**: False alarm rate normalized across negative audio hours.
- **Detection Latency**: Milliseconds from impulse onset to detection trigger (p50, p90, max).
- **Rolling-Window Boundary Coverage**: Probing impulses placed at boundary offsets (0ms, 50ms, 195ms, 390ms, 405ms, 950ms) across the 400ms inference hop grid.
- **Hard Negative Rejection**: Stress tests against door slams, hand claps, balloon pops, dropped books, and loud shouting.
- **Resampling & Windowing Comparison**: Comparing the production box-filter vs polyphase anti-aliased resampling, and latest vs retrospective impulse-centered windowing.

---

## Directory Structure

```text
eval/
├── __init__.py                # Package initialization with environment guards
├── class_map.py               # Robust CSV class map parser (handles quotes/commas)
├── classifier.py              # TFLite YAMNet wrapper (matches [15600] in, [1, 521] out)
├── detector_pipeline.py       # Full AudioSensorService simulation (Tier 1 gate + Tier 2 ML)
├── synthetic_audio.py         # Calibrated synthetic acoustic generators & test suites
├── eval_harness.py            # Streaming audio replay, window matching, and scoring engine
├── run_eval.py                # Command-line benchmark runner with matrix output
├── test_eval_harness.py       # Pytest regression suite
└── sample_manifest.json       # Example labeled recording manifest
```

---

## Runnable Commands

### 1. Run Automated Pytest Regression Suite
```bash
python3 -m pytest eval/test_eval_harness.py -s -o cache_dir=/tmp/pytest_cache -v
```

### 2. Run Comparative Evaluation Matrix
Compares Legacy (Bang/Explosion/Cap gun) vs Tightened class selection, along with retrospective windowing and polyphase resampling:
```bash
python3 -m eval.run_eval --compare
```

### 3. Run Single Mode Evaluation with JSON Output
```bash
python3 -m eval.run_eval --mode tightened --output eval/benchmark_results.json
```

### 4. Run Evaluation on a Labeled Manifest
```bash
python3 -m eval.run_eval --manifest eval/sample_manifest.json
```

---

## Evaluation Results Summary

### Comparative Performance Matrix

| Configuration | Event Recall | False Alert Rate | Hard Neg FPs | Latency p50 | Latency p90 |
|---|:---:|:---:|:---:|:---:|:---:|
| **Legacy Classifier** (Bang/Explosion/Cap gun) | 100.0% | 313.0 / hr | 4 FPs | 26.4 ms | 43.5 ms |
| **Tightened Classifier** (Gunshot Only: 421-424) | 100.0% | 313.0 / hr | 3 FPs | 26.4 ms | 43.5 ms |
| **Tightened + Retrospective Window** | 100.0% | 313.0 / hr | 3 FPs | 26.4 ms | 43.5 ms |
| **Tightened + Retrospective + Polyphase** | 100.0% | **234.8 / hr** | **2 FPs** | 26.4 ms | 43.5 ms |

### Hard Negative Breakdown

| Negative Sound Type | Tested Events | Legacy False Alerts | Tightened False Alerts |
|---|:---:|:---:|:---:|
| **door_slam** | 2 | **1 FP** (Explosion: 0.185) | **0 FPs** (Rejected: < 0.10) |
| **dropped_book** | 1 | 0 FPs | 0 FPs |
| **speech_shout** | 1 | 0 FPs | 0 FPs |
| **hand_clap** | 2 | 2 FPs | 2 FPs (high broadband energy) |
| **balloon_pop** | 1 | 1 FP | 1 FP |

### Key Findings:
1. **Class Selection**: Legacy class mapping matched `[420, 421, 422, 423, 424, 425, 428, 460]`. It caused heavy door slams to trigger false alerts under the "Explosion" label. Tightening to `[421, 422, 423, 424]` immediately prevents generic low-frequency door thuds from alarming.
2. **Resampling Filter**: The existing naive box filter in `AudioSensorService.kt.appendHistory` (`sum / count` of 2-3 samples) causes high-frequency aliasing into the 0-8 kHz band. Anti-aliased polyphase resampling reduces false alert rate by 25%.
3. **Window Timing & Centering**: Local sentinel detection currently classifies `snapshotHistoryAt(currentTime)`. If an impulse arrives right before a 400ms inference hop, the snapshot catches only the onset. Centering the window retrospectively on `lastImpulseAtMs + 500ms` guarantees the reverberant tail is captured for ML inference.

---

## Manifest Schema for Labeled Field Recordings

To evaluate on real recorded datasets (e.g. firing range captures, UrbanSound8K, AudioSet), provide a JSON manifest following this schema:

```json
{
  "version": "1.0",
  "description": "Evaluation manifest for field audio",
  "recordings": [
    {
      "recording_id": "indoor_range_9mm_01",
      "audio_path": "/path/to/recording.wav",
      "duration_sec": 30.5,
      "sample_rate": 44100,
      "environment": "indoor_corridor",
      "is_synthetic": false,
      "events": [
        {
          "event_id": "shot_01",
          "label": "gunshot",
          "start_sec": 4.12,
          "end_sec": 4.65,
          "is_threat": true,
          "boundary_offset_ms": 120.0,
          "metadata": {
            "weapon": "9mm handgun",
            "distance_m": 12.0
          }
        },
        {
          "event_id": "slam_01",
          "label": "door_slam",
          "start_sec": 15.30,
          "end_sec": 15.90,
          "is_threat": false
        }
      ]
    }
  ]
}
```

---

## Runtime & Dataset Blockers

1. **Java Runtime Missing on Host**:
   - `java -version` returns `Unable to locate a Java Runtime`.
   - Android Gradle checks (`./gradlew test`) cannot execute directly on this host machine until JDK 17+ is installed.
   - The Python evaluation harness circumvents this blocker by running the exact TFLite interpreter and model weights directly.
2. **Real Audio Asset Storage**:
   - The repository contains `yamnet.tflite` model weights and class maps, but no real acoustic WAV/FLAC audio files are checked in (due to git repository size considerations).
   - Real model validation requires downloading or linking a labeled test corpus into the manifest format.
3. **Synthetic Checks vs Measured Acoustic Model Accuracy**:
   - **Synthetic Checks**: Validate pipeline mechanics, activity gate RMS/peak formulas, clipping detection, EMA responsiveness, ring-buffer ring-indexing, inference rate-limiting, and timing latency.
   - **Measured Model Accuracy**: Runs the trained neural network (`yamnet.tflite`) weights over acoustic features. Synthetic impulses test whether the model activates on shockwave/blast patterns vs formants; final real-world precision/recall figures require labeled physical recordings.
