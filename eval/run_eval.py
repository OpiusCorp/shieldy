"""Command line runner for acoustic detector evaluation harness."""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
import sys
from typing import Dict, Any
import numpy as np

from eval.classifier import YamnetClassifier
from eval.class_map import YamnetClassMap
from eval.detector_pipeline import AudioDetectorPipeline
from eval.synthetic_audio import SyntheticAudioGenerator, compute_audio_sha256
from eval.eval_harness import EvaluationHarness, MetricSummary
from eval.manifest import EvaluationManifest


def format_table(header: list, rows: list) -> str:
    col_widths = [len(h) for h in header]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    hdr = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(header)) + " |"
    lines = [sep, hdr, sep]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row))
            + " |"
        )
    lines.append(sep)
    return "\n".join(lines)


def evaluate_pregenerated_corpus(
    classifier: YamnetClassifier,
    classifier_mode: str,
    timing_mode: str,
    resample_method: str,
    boundary_pcm: np.ndarray,
    boundary_events: list,
    hn_pcm: np.ndarray,
    hn_events: list,
) -> Dict[str, MetricSummary]:
    pipeline = AudioDetectorPipeline(
        classifier=classifier,
        classifier_mode=classifier_mode,
        timing_mode=timing_mode,
        resample_method=resample_method,
    )
    harness = EvaluationHarness(pipeline=pipeline)

    # 1. Boundary Suite Replay
    boundary_metrics = harness.evaluate_stream(boundary_pcm, boundary_events)

    # 2. Hard Negatives Suite Replay
    hn_metrics = harness.evaluate_stream(hn_pcm, hn_events)

    return {"boundary": boundary_metrics, "hard_negatives": hn_metrics}


def main():
    parser = argparse.ArgumentParser(description="EchoShield Detector Evaluation Harness")
    parser.add_argument("--manifest", type=str, help="Path to labeled evaluation manifest JSON")
    parser.add_argument(
        "--mode",
        choices=["tightened", "legacy"],
        default="tightened",
        help="Classifier class selection mode",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparative evaluation (tightened vs legacy, latest vs retrospective window)",
    )
    parser.add_argument("--output", type=str, help="Save evaluation summary to JSON file")
    args = parser.parse_args()

    print("======================================================================")
    print("        ECHOSHIELD ACOUSTIC DETECTOR EVALUATION & SCORING HARNESS      ")
    print("======================================================================")

    class_map = YamnetClassMap()
    classifier = YamnetClassifier(class_map=class_map, mode=args.mode)
    generator = SyntheticAudioGenerator(seed=42)

    # Pre-generate corpus ONCE so all configurations replay identical inputs
    print("[Corpus Generation] Generating deterministic test corpus (seed=42)...")
    boundary_pcm, boundary_events = generator.generate_boundary_evaluation_suite()
    hn_pcm, hn_events = generator.generate_hard_negatives_suite()

    b_sha = compute_audio_sha256(boundary_pcm)
    hn_sha = compute_audio_sha256(hn_pcm)
    print(f"-> Boundary Suite: {len(boundary_pcm)/44100:.1f}s, SHA256: {b_sha[:16]}...")
    print(f"-> Hard Negatives: {len(hn_pcm)/44100:.1f}s, SHA256: {hn_sha[:16]}...")

    if args.manifest:
        print(f"\n[Manifest Evaluation] Loading manifest: {args.manifest}")
        manifest = EvaluationManifest.load_json(args.manifest)
        pipeline = AudioDetectorPipeline(classifier=classifier, classifier_mode=args.mode)
        harness = EvaluationHarness(pipeline=pipeline)

        for rec in manifest.recordings:
            print(f"-> Processing recording: {rec.recording_id} ({rec.duration_sec:.1f}s)")
            audio = rec.load_audio()
            metrics = harness.evaluate_stream(audio, rec.events)
            print(f"   Recall: {metrics.event_recall_pct:.1f}% ({metrics.true_positives}/{metrics.threat_events_count})")
            print(f"   False Alerts/Hr: {metrics.false_alerts_per_hour:.1f}")
            print(f"   Latency p50: {metrics.latency_p50_ms:.1f}ms, p90: {metrics.latency_p90_ms:.1f}ms")
        return

    if args.compare:
        print("\n--- COMPARATIVE EVALUATION MATRIX (IDENTICAL CORPUS REPLAY) ---")
        configs = [
            ("Baseline Classifier (Bang/Expl/Cap)", "legacy", "baseline", "box_filter"),
            ("Tightened Classifier (Gunshot Only)", "tightened", "baseline", "box_filter"),
            ("Tightened + Post-Impulse Tail Wait", "tightened", "post_impulse_tail", "box_filter"),
            ("Tightened + Tail Wait + Polyphase", "tightened", "post_impulse_tail", "polyphase"),
        ]

        summary_rows = []
        for name, mode, timing, resample in configs:
            classifier.set_mode(mode)
            results = evaluate_pregenerated_corpus(
                classifier=classifier,
                classifier_mode=mode,
                timing_mode=timing,
                resample_method=resample,
                boundary_pcm=boundary_pcm,
                boundary_events=boundary_events,
                hn_pcm=hn_pcm,
                hn_events=hn_events,
            )
            b_m = results["boundary"]
            hn_m = results["hard_negatives"]

            summary_rows.append(
                [
                    name,
                    f"{b_m.event_recall_pct:.1f}%",
                    f"{hn_m.false_alerts_per_hour:.1f}/hr",
                    f"{hn_m.false_positives} FPs",
                    f"{b_m.latency_p50_ms:.1f}ms",
                    f"{b_m.latency_p90_ms:.1f}ms",
                ]
            )

        hdr = [
            "Configuration",
            "Event Recall",
            "False Alert Rate",
            "Hard Neg FPs",
            "Latency p50",
            "Latency p90",
        ]
        print("\n" + format_table(hdr, summary_rows) + "\n")

        # Print Hard Negative Breakdown for Legacy vs Tightened
        print("\n--- HARD NEGATIVES FALSE ALERT BREAKDOWN ---")
        classifier.set_mode("legacy")
        leg_hn = evaluate_pregenerated_corpus(
            classifier, "legacy", "baseline", "box_filter", boundary_pcm, boundary_events, hn_pcm, hn_events
        )["hard_negatives"]
        classifier.set_mode("tightened")
        tight_hn = evaluate_pregenerated_corpus(
            classifier, "tightened", "baseline", "box_filter", boundary_pcm, boundary_events, hn_pcm, hn_events
        )["hard_negatives"]

        hn_rows = []
        all_labels = set(leg_hn.hard_negative_breakdown.keys()) | set(tight_hn.hard_negative_breakdown.keys())
        for lbl in sorted(all_labels):
            leg_fp = leg_hn.hard_negative_breakdown.get(lbl, {}).get("false_alerts", 0)
            tight_fp = tight_hn.hard_negative_breakdown.get(lbl, {}).get("false_alerts", 0)
            tot = leg_hn.hard_negative_breakdown.get(lbl, {}).get("total_events", 0)
            hn_rows.append([lbl, str(tot), f"{leg_fp} FPs", f"{tight_fp} FPs"])

        print(format_table(["Negative Sound Type", "Tested Events", "Legacy False Alerts", "Tightened False Alerts"], hn_rows))

        if args.output:
            matrix_data = []
            for name, mode, timing, resample in configs:
                classifier.set_mode(mode)
                res = evaluate_pregenerated_corpus(
                    classifier=classifier,
                    classifier_mode=mode,
                    timing_mode=timing,
                    resample_method=resample,
                    boundary_pcm=boundary_pcm,
                    boundary_events=boundary_events,
                    hn_pcm=hn_pcm,
                    hn_events=hn_events,
                )
                b = res["boundary"]
                hn = res["hard_negatives"]
                matrix_data.append({
                    "configuration": name,
                    "classifier_mode": mode,
                    "timing_mode": timing,
                    "resample_method": resample,
                    "event_recall_pct": b.event_recall_pct,
                    "threat_events_count": b.threat_events_count,
                    "true_positives": b.true_positives,
                    "false_negatives": b.false_negatives,
                    "false_alerts_per_hour": hn.false_alerts_per_hour,
                    "false_positives": hn.false_positives,
                    "duplicate_alerts": b.duplicate_alerts + hn.duplicate_alerts,
                    "total_negative_hours": hn.total_negative_hours,
                    "latency_p50_ms": b.latency_p50_ms,
                    "latency_p90_ms": b.latency_p90_ms,
                    "inference_failures": b.inference_failures + hn.inference_failures,
                })

            out_data = {
                "benchmark_type": "synthetic_calibrated_identical_corpus",
                "corpus_seed": 42,
                "boundary_corpus_sha256": b_sha,
                "hard_negatives_corpus_sha256": hn_sha,
                "tested_detector_head_sha": "af6564b",
                "model_executed": "yamnet.tflite (unquantized float32, bundled in app/src/main/assets)",
                "class_map_source": "app/src/main/assets/yamnet_class_map.csv",
                "latency_definition": "trigger_timestamp_ms - ground_truth_impulse_onset_ms (includes frame capture buffer read and post-impulse tail wait if enabled)",
                "comparative_matrix": matrix_data,
                "hard_negatives_breakdown": {
                    lbl: {
                        "tested_events": leg_hn.hard_negative_breakdown.get(lbl, {}).get("total_events", 0),
                        "legacy_false_alerts": leg_hn.hard_negative_breakdown.get(lbl, {}).get("false_alerts", 0),
                        "tightened_baseline_false_alerts": tight_hn.hard_negative_breakdown.get(lbl, {}).get("false_alerts", 0),
                    }
                    for lbl in sorted(all_labels)
                },
            }
            with open(args.output, "w") as f:
                json.dump(out_data, f, indent=2)
            print(f"\nSaved comparative metrics to {args.output}")

        return

    # Single mode benchmark
    print(f"\n[Single Benchmark] Running mode={args.mode}...")
    results = evaluate_pregenerated_corpus(
        classifier=classifier,
        classifier_mode=args.mode,
        timing_mode="baseline",
        resample_method="box_filter",
        boundary_pcm=boundary_pcm,
        boundary_events=boundary_events,
        hn_pcm=hn_pcm,
        hn_events=hn_events,
    )
    b_m = results["boundary"]
    hn_m = results["hard_negatives"]

    print("\n--- RESULTS SUMMARY ---")
    print(f"Event Recall:             {b_m.event_recall_pct:.1f}% ({b_m.true_positives}/{b_m.threat_events_count} threats detected)")
    print(f"False Alerts Per Hour:    {hn_m.false_alerts_per_hour:.1f} alerts/hr (on negative stream)")
    print(f"Detection Latency:        p50={b_m.latency_p50_ms:.1f}ms, p90={b_m.latency_p90_ms:.1f}ms, max={b_m.latency_max_ms:.1f}ms")

    print("\n--- ROLLING-WINDOW BOUNDARY OFFSET BREAKDOWN ---")
    b_rows = []
    for offset_key, data in b_m.boundary_metrics.items():
        b_rows.append([
            offset_key,
            f"{data['detected']}/{data['total']}",
            f"{data['recall_pct']:.1f}%",
            f"{data['avg_latency_ms']:.1f}ms",
        ])
    print(format_table(["Impulse Boundary Offset", "Detected", "Recall", "Avg Latency"], b_rows))

    if args.output:
        out_data = {
            "mode": args.mode,
            "event_recall_pct": b_m.event_recall_pct,
            "false_alerts_per_hour": hn_m.false_alerts_per_hour,
            "latency_p50_ms": b_m.latency_p50_ms,
            "latency_p90_ms": b_m.latency_p90_ms,
            "boundary_metrics": b_m.boundary_metrics,
            "hard_negative_breakdown": hn_m.hard_negative_breakdown,
        }
        with open(args.output, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"\nSaved metrics output to {args.output}")


if __name__ == "__main__":
    main()
