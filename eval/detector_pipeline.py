"""Simulation of the Android AudioSensorService two-tier detection pipeline."""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from eval.classifier import YamnetClassifier, ClassificationResult


@dataclass
class DetectionAlert:
    timestamp_ms: float
    trigger_latency_ms: float
    impulse_onset_ms: float
    confidence: float
    top_label: str
    gunshot_label: str
    gate_reason: str
    effective_threshold: float
    clip_fraction: float


class AudioDetectorPipeline:
    SAMPLE_RATE = 44100
    MODEL_SAMPLE_RATE = 16000
    MODEL_INPUT_SAMPLES = 15600  # ~0.975 seconds

    EMA_ALPHA = 0.4
    PEAK_THRESHOLD_NORMALIZED = 0.25
    IMPULSIVE_PEAK_RMS_RATIO = 3.0
    CLIP_MARGIN = 64
    CLIP_FRACTION_GATE = 0.005
    CLIP_FRACTION_SEVERE = 0.02
    CLIPPED_CONFIDENCE_THRESHOLD = 0.06
    GUNSHOT_CONFIDENCE_THRESHOLD = 0.10
    MIN_CONSECUTIVE_DETECTIONS = 1
    MODEL_INFERENCE_INTERVAL_MS = 400.0
    TRIGGER_COOLDOWN_MS = 5000.0

    HISTORY_SECONDS = 4
    HISTORY_SAMPLES = 16000 * 4  # 64,000 floats
    IMPULSE_MEMORY_MS = 3000.0
    IMPULSE_ONSET_PEAK_NORMALIZED = 0.15
    WINDOW_TAIL_MS = 500.0

    def __init__(
        self,
        classifier: Optional[YamnetClassifier] = None,
        detection_threshold: float = 200.0,
        classifier_mode: str = "tightened",
        resample_method: str = "box_filter",
        window_strategy: str = "latest",  # "latest" (sentinel default) or "retrospective"
    ):
        self.classifier = classifier or YamnetClassifier(mode=classifier_mode)
        self.classifier_mode = classifier_mode
        self.detection_threshold = detection_threshold
        self.resample_method = resample_method
        self.window_strategy = window_strategy

        # Detector internal state
        self.history_buffer = np.zeros(self.HISTORY_SAMPLES, dtype=np.float32)
        self.history_write_index = 0
        self.history_samples_written = 0
        self.history_end_time_ms = 0.0

        self.last_impulse_at_ms = 0.0
        self.smoothed_amplitude = 0.0
        self.consecutive_high_confidence = 0
        self.current_gate_open = False
        self.last_trigger_time = -self.TRIGGER_COOLDOWN_MS
        self.last_model_inference = -self.MODEL_INFERENCE_INTERVAL_MS

        self.alerts: List[DetectionAlert] = []

    def reset(self):
        self.history_buffer.fill(0.0)
        self.history_write_index = 0
        self.history_samples_written = 0
        self.history_end_time_ms = 0.0
        self.last_impulse_at_ms = 0.0
        self.smoothed_amplitude = 0.0
        self.consecutive_high_confidence = 0
        self.current_gate_open = False
        self.last_trigger_time = -self.TRIGGER_COOLDOWN_MS
        self.last_model_inference = -self.MODEL_INFERENCE_INTERVAL_MS
        self.alerts.clear()

    def process_frame(
        self, frame_pcm16: np.ndarray, current_time_ms: float
    ) -> Optional[DetectionAlert]:
        """
        Process a single buffer frame of 44.1 kHz 16-bit PCM samples.
        Matches Android AudioSensorService.processAudioStream loop.
        """
        read_size = len(frame_pcm16)
        if read_size <= 0:
            return None

        # 1. Compute acoustic metrics
        samples_float = frame_pcm16.astype(np.float64)
        instant_rms = math.sqrt(np.mean(samples_float ** 2))
        abs_samples = np.abs(frame_pcm16)
        peak_idx = int(np.argmax(abs_samples))
        peak_amplitude = float(abs_samples[peak_idx])
        peak_normalized = peak_amplitude / 32768.0

        # Clipping calculation
        clip_high = 32767 - self.CLIP_MARGIN
        clip_low = -32768 + self.CLIP_MARGIN
        clipped_count = np.sum((frame_pcm16 >= clip_high) | (frame_pcm16 <= clip_low))
        clip_fraction = float(clipped_count) / float(read_size)
        clipping_gate = clip_fraction >= self.CLIP_FRACTION_GATE
        clipping_severe = clip_fraction >= self.CLIP_FRACTION_SEVERE

        # 2. Append to history buffer
        self._append_history(frame_pcm16, current_time_ms)

        # 3. Timestamp impulse onset
        if peak_normalized >= self.IMPULSE_ONSET_PEAK_NORMALIZED or clipping_gate:
            frame_start_ms = current_time_ms - (read_size * 1000.0 / self.SAMPLE_RATE)
            self.last_impulse_at_ms = frame_start_ms + (peak_idx * 1000.0 / self.SAMPLE_RATE)

        # 4. Tier 1: EMA smoothing and Gate check
        self.smoothed_amplitude = (
            self.EMA_ALPHA * instant_rms + (1.0 - self.EMA_ALPHA) * self.smoothed_amplitude
        )

        has_enough_history = self.history_samples_written >= self.MODEL_INPUT_SAMPLES
        rms_gate_open = instant_rms >= self.detection_threshold
        peak_gate_open = peak_normalized >= self.PEAK_THRESHOLD_NORMALIZED
        impulsive_sound = (
            instant_rms > 50.0 and (peak_amplitude / max(1e-5, instant_rms)) > self.IMPULSIVE_PEAK_RMS_RATIO
        )

        gate_open = (
            rms_gate_open or peak_gate_open or impulsive_sound or clipping_gate
        ) and has_enough_history

        gate_reason = "NONE"
        if gate_open:
            if rms_gate_open:
                gate_reason = "RMS"
            elif peak_gate_open:
                gate_reason = "PEAK"
            elif impulsive_sound:
                gate_reason = "IMPULSIVE"
            elif clipping_gate:
                gate_reason = "CLIP"

        if not gate_open:
            self.consecutive_high_confidence = 0

        self.current_gate_open = gate_open

        # 5. Tier 2: Model inference rate check
        cooldown_remaining = max(0.0, self.TRIGGER_COOLDOWN_MS - (current_time_ms - self.last_trigger_time))
        should_run_model = (
            gate_open
            and (current_time_ms - self.last_model_inference >= self.MODEL_INFERENCE_INTERVAL_MS)
        )

        if not should_run_model:
            return None

        self.last_model_inference = current_time_ms

        # Select window to classify
        if self.window_strategy == "retrospective":
            now = current_time_ms
            impulse_at = self.last_impulse_at_ms
            heard_impulse = (impulse_at > 0.0) and (now - impulse_at <= self.IMPULSE_MEMORY_MS)
            window_end = (impulse_at + self.WINDOW_TAIL_MS) if heard_impulse else now
            model_input = self._snapshot_history_at(window_end)
        else:
            # Latest window ending now (Android sentinel default)
            model_input = self._snapshot_history_at(current_time_ms)

        if model_input is None:
            return None

        result = self.classifier.classify(model_input)
        if result is None:
            return None

        effective_threshold = (
            self.CLIPPED_CONFIDENCE_THRESHOLD
            if clipping_severe
            else self.GUNSHOT_CONFIDENCE_THRESHOLD
        )
        is_above_threshold = result.gunshot_confidence >= effective_threshold

        if is_above_threshold:
            self.consecutive_high_confidence += 1
        else:
            self.consecutive_high_confidence = 0

        # Trigger check
        if (
            self.consecutive_high_confidence >= self.MIN_CONSECUTIVE_DETECTIONS
            and cooldown_remaining == 0.0
        ):
            self.last_trigger_time = current_time_ms
            self.consecutive_high_confidence = 0

            impulse_onset = (
                self.last_impulse_at_ms
                if (self.last_impulse_at_ms > 0 and current_time_ms - self.last_impulse_at_ms <= self.IMPULSE_MEMORY_MS)
                else current_time_ms
            )
            trigger_latency = current_time_ms - impulse_onset

            alert = DetectionAlert(
                timestamp_ms=current_time_ms,
                trigger_latency_ms=trigger_latency,
                impulse_onset_ms=impulse_onset,
                confidence=result.gunshot_confidence,
                top_label=result.top_label,
                gunshot_label=result.best_gunshot_label,
                gate_reason=gate_reason,
                effective_threshold=effective_threshold,
                clip_fraction=clip_fraction,
            )
            self.alerts.append(alert)
            return alert

        return None

    def _append_history(self, source: np.ndarray, frame_end_time_ms: float):
        """Resample 44.1 kHz to 16.0 kHz and write into ring buffer."""
        size = len(source)
        if size == 0:
            return

        if self.resample_method == "polyphase":
            from scipy import signal
            norm_source = source.astype(np.float32) / 32768.0
            resampled = signal.resample_poly(norm_source, 160, 441)
        else:
            # Fast vectorized box-filter matching AudioSensorService.kt
            ratio = self.SAMPLE_RATE / self.MODEL_SAMPLE_RATE
            n_out = int(size / ratio)
            if n_out <= 0:
                return
            cum = np.cumsum(source.astype(np.float64) / 32768.0)
            starts = np.floor(np.arange(n_out) * ratio).astype(int)
            ends = np.clip(np.floor((np.arange(n_out) + 1) * ratio).astype(int), 1, size)
            prev_cum = np.where(starts > 0, cum[np.maximum(0, starts - 1)], 0.0)
            sums = cum[ends - 1] - prev_cum
            counts = np.maximum(1, ends - starts)
            resampled = (sums / counts).astype(np.float32)

        n_samples = len(resampled)
        if n_samples == 0:
            return

        # Vectorized ring buffer write
        idx = self.history_write_index
        if idx + n_samples <= self.HISTORY_SAMPLES:
            self.history_buffer[idx : idx + n_samples] = resampled
        else:
            first_chunk = self.HISTORY_SAMPLES - idx
            self.history_buffer[idx:] = resampled[:first_chunk]
            self.history_buffer[: n_samples - first_chunk] = resampled[first_chunk:]

        self.history_write_index = (idx + n_samples) % self.HISTORY_SAMPLES
        self.history_samples_written += n_samples
        self.history_end_time_ms = frame_end_time_ms

    def _snapshot_history_at(self, end_time_ms: float) -> Optional[np.ndarray]:
        """Extract 15,600 samples ending at end_time_ms from ring buffer."""
        if self.history_samples_written < self.MODEL_INPUT_SAMPLES:
            return None

        ms_behind = max(0.0, self.history_end_time_ms - end_time_ms)
        oldest_offset = min(
            self.history_samples_written - self.MODEL_INPUT_SAMPLES,
            self.HISTORY_SAMPLES - self.MODEL_INPUT_SAMPLES,
        )
        samples_behind = int(
            min(oldest_offset, max(0.0, ms_behind * self.MODEL_SAMPLE_RATE / 1000.0))
        )

        end = (self.history_write_index - samples_behind) % self.HISTORY_SAMPLES
        start = (end - self.MODEL_INPUT_SAMPLES) % self.HISTORY_SAMPLES

        output = np.zeros(self.MODEL_INPUT_SAMPLES, dtype=np.float32)
        if start + self.MODEL_INPUT_SAMPLES <= self.HISTORY_SAMPLES:
            output[:] = self.history_buffer[start : start + self.MODEL_INPUT_SAMPLES]
        else:
            first_chunk = self.HISTORY_SAMPLES - start
            output[:first_chunk] = self.history_buffer[start:]
            output[first_chunk:] = self.history_buffer[: self.MODEL_INPUT_SAMPLES - first_chunk]

        return output
