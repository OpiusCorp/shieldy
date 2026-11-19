"""Evaluation and scoring harness for acoustic gunshot detection."""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from eval.synthetic_audio import GroundTruthEvent
from eval.detector_pipeline import AudioDetectorPipeline, DetectionAlert


@dataclass
class MetricSummary:
    total_audio_sec: float
    total_negative_hours: float
    threat_events_count: int
    true_positives: int
    false_negatives: int
    false_positives: int
    event_recall_pct: float
    false_alerts_per_hour: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_min_ms: float
    latency_max_ms: float
    boundary_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    hard_negative_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    alerts: List[Dict[str, Any]] = field(default_factory=list)


class EvaluationHarness:
    def __init__(
        self,
        pipeline: Optional[AudioDetectorPipeline] = None,
        chunk_size: int = 2048,
        max_latency_allowance_ms: float = 1500.0,
    ):
        self.pipeline = pipeline or AudioDetectorPipeline()
        self.chunk_size = chunk_size
        self.max_latency_allowance_ms = max_latency_allowance_ms

    def evaluate_stream(
        self, audio_pcm16: np.ndarray, ground_truth_events: List[GroundTruthEvent]
    ) -> MetricSummary:
        """
        Replay continuous 44.1 kHz PCM16 audio through the detector pipeline,
        simulate real-time frame reads, and score against ground truth events.
        """
        self.pipeline.reset()
        sample_rate = self.pipeline.SAMPLE_RATE
        total_samples = len(audio_pcm16)
        total_duration_sec = total_samples / sample_rate

        curr_sample = 0
        simulated_time_ms = 0.0

        all_alerts: List[DetectionAlert] = []

        while curr_sample < total_samples:
            end_sample = min(total_samples, curr_sample + self.chunk_size)
            frame = audio_pcm16[curr_sample:end_sample]
            frame_duration_ms = len(frame) * 1000.0 / sample_rate
            simulated_time_ms += frame_duration_ms

            alert = self.pipeline.process_frame(frame, simulated_time_ms)
            if alert:
                all_alerts.append(alert)

            curr_sample = end_sample

        return self._score_results(total_duration_sec, ground_truth_events, all_alerts)

    def _score_results(
        self,
        total_duration_sec: float,
        gt_events: List[GroundTruthEvent],
        alerts: List[DetectionAlert],
    ) -> MetricSummary:
        threat_events = [e for e in gt_events if e.is_threat]
        non_threat_events = [e for e in gt_events if not e.is_threat]

        tp = 0
        fn = 0
        latencies = []
        matched_alert_indices = set()

        # Score threat events
        boundary_stats: Dict[str, Dict[str, Any]] = {}

        for evt in threat_events:
            start_ms = evt.start_sec * 1000.0
            end_ms = (evt.end_sec * 1000.0) + self.max_latency_allowance_ms

            # Find matching alerts
            matching = [
                (idx, a)
                for idx, a in enumerate(alerts)
                if start_ms <= a.timestamp_ms <= end_ms
            ]

            b_key = (
                f"{evt.boundary_offset_ms:.0f}ms"
                if evt.boundary_offset_ms is not None
                else "default"
            )
            if b_key not in boundary_stats:
                boundary_stats[b_key] = {"total": 0, "detected": 0, "latencies": []}
            boundary_stats[b_key]["total"] += 1

            if matching:
                tp += 1
                first_idx, first_alert = matching[0]
                matched_alert_indices.add(first_idx)
                latency = max(0.0, first_alert.timestamp_ms - start_ms)
                latencies.append(latency)

                boundary_stats[b_key]["detected"] += 1
                boundary_stats[b_key]["latencies"].append(latency)
            else:
                fn += 1

        # Calculate False Positives (alerts not matched to any threat event)
        fp_alerts = [
            a for idx, a in enumerate(alerts) if idx not in matched_alert_indices
        ]
        fp = len(fp_alerts)

        # Hard negative breakdown
        hn_breakdown: Dict[str, Dict[str, Any]] = {}
        for non_threat in non_threat_events:
            lbl = non_threat.label
            if lbl not in hn_breakdown:
                hn_breakdown[lbl] = {"total_events": 0, "false_alerts": 0}
            hn_breakdown[lbl]["total_events"] += 1

            s_ms = non_threat.start_sec * 1000.0
            e_ms = (non_threat.end_sec * 1000.0) + self.max_latency_allowance_ms
            for fa in fp_alerts:
                if s_ms <= fa.timestamp_ms <= e_ms:
                    hn_breakdown[lbl]["false_alerts"] += 1

        # Calculate time metrics
        threat_duration_sec = sum(
            max(0.0, e.end_sec - e.start_sec) for e in threat_events
        )
        negative_duration_sec = max(1.0, total_duration_sec - threat_duration_sec)
        negative_hours = negative_duration_sec / 3600.0

        recall_pct = (tp / max(1, (tp + fn))) * 100.0 if (tp + fn) > 0 else 0.0
        false_alerts_per_hr = fp / negative_hours

        # Latency statistics
        if latencies:
            lat_mean = float(np.mean(latencies))
            lat_p50 = float(np.percentile(latencies, 50))
            lat_p90 = float(np.percentile(latencies, 90))
            lat_min = float(np.min(latencies))
            lat_max = float(np.max(latencies))
        else:
            lat_mean = lat_p50 = lat_p90 = lat_min = lat_max = 0.0

        # Boundary summary
        boundary_summary = {}
        for k, v in boundary_stats.items():
            tot = v["total"]
            det = v["detected"]
            lats = v["latencies"]
            boundary_summary[k] = {
                "total": tot,
                "detected": det,
                "recall_pct": (det / max(1, tot)) * 100.0,
                "avg_latency_ms": float(np.mean(lats)) if lats else 0.0,
            }

        return MetricSummary(
            total_audio_sec=total_duration_sec,
            total_negative_hours=negative_hours,
            threat_events_count=len(threat_events),
            true_positives=tp,
            false_negatives=fn,
            false_positives=fp,
            event_recall_pct=recall_pct,
            false_alerts_per_hour=false_alerts_per_hr,
            latency_mean_ms=lat_mean,
            latency_p50_ms=lat_p50,
            latency_p90_ms=lat_p90,
            latency_min_ms=lat_min,
            latency_max_ms=lat_max,
            boundary_metrics=boundary_summary,
            hard_negative_breakdown=hn_breakdown,
            alerts=[
                {
                    "timestamp_ms": a.timestamp_ms,
                    "confidence": a.confidence,
                    "top_label": a.top_label,
                    "gunshot_label": a.gunshot_label,
                    "gate_reason": a.gate_reason,
                    "latency_ms": a.trigger_latency_ms,
                }
                for a in alerts
            ],
        )
