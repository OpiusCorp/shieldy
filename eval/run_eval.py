"""Command line runner for acoustic detector evaluation harness."""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
import sys
from typing import Dict, Any

from eval.classifier import YamnetClassifier
from eval.class_map import YamnetClassMap
from eval.detector_pipeline import AudioDetectorPipeline
from eval.synthetic_audio import SyntheticAudioGenerator
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


def run_benchmark(
    generator: SyntheticAudioGenerator,
    classifier: YamnetClassifier,
    classifier_mode: str,
    window_strategy: str = "latest",
    resample_method: str = "box_filter",
) -> Dict[str, MetricSummary]:
    pipeline = AudioDetectorPipeline(
        classifier=classifier,
        classifier_mode=classifier_mode,
        window_strategy=window_strategy,
        resample_method=resample_method,
    )
    harness = EvaluationHarness(pipeline=pipeline)

    # 1. Boundary Suite
    boundary_pcm, boundary_events = generator.generate_boundary_evaluation_suite()
    boundary_metrics = harness.evaluate_stream(boundary_pcm, boundary_events)

    # 2. Hard Negatives Suite
    hn_pcm, hn_events = generator.generate_hard_negatives_suite()
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
    generator = SyntheticAudioGenerator()

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
        print("\n--- COMPARATIVE EVALUATION MATRIX ---")
        configs = [
            ("Legacy Classifier (Bang/Expl/Cap)", "legacy", "latest", "box_filter"),
            ("Tightened Classifier (Gunshot Only)", "tightened", "latest", "box_filter"),
            ("Tightened + Retrospective Window", "tightened", "retrospective", "box_filter"),
            ("Tightened + Retrospective + Polyphase", "tightened", "retrospective", "polyphase"),
        ]

        summary_rows = []
        for name, mode, win, resample in configs:
            classifier.set_mode(mode)
            results = run_benchmark(
                generator,
                classifier,
                classifier_mode=mode,
                window_strategy=win,
                resample_method=resample,
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
        leg_hn = run_benchmark(generator, classifier, "legacy")["hard_negatives"]
        classifier.set_mode("tightened")
        tight_hn = run_benchmark(generator, classifier, "tightened")["hard_negatives"]

        hn_rows = []
        all_labels = set(leg_hn.hard_negative_breakdown.keys()) | set(tight_hn.hard_negative_breakdown.keys())
        for lbl in sorted(all_labels):
            leg_fp = leg_hn.hard_negative_breakdown.get(lbl, {}).get("false_alerts", 0)
            tight_fp = tight_hn.hard_negative_breakdown.get(lbl, {}).get("false_alerts", 0)
            tot = leg_hn.hard_negative_breakdown.get(lbl, {}).get("total_events", 0)
            hn_rows.append([lbl, str(tot), f"{leg_fp} FPs", f"{tight_fp} FPs"])

        print(format_table(["Negative Sound Type", "Tested Events", "Legacy False Alerts", "Tightened False Alerts"], hn_rows))
        return

    # Single mode benchmark
    print(f"\n[Single Benchmark] Running mode={args.mode}...")
    results = run_benchmark(generator, classifier, classifier_mode=args.mode)
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
