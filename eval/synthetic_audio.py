"""Calibrated synthetic audio generation for detector evaluation."""

import hashlib
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


def compute_audio_sha256(audio_pcm16: np.ndarray) -> str:
    """Compute SHA-256 digest of 16-bit PCM audio bytes."""
    return hashlib.sha256(np.ascontiguousarray(audio_pcm16).tobytes()).hexdigest()


@dataclass
class GroundTruthEvent:
    event_id: str
    label: str  # "gunshot", "door_slam", "hand_clap", "balloon_pop", "speech_shout", "dropped_book"
    start_sec: float
    end_sec: float
    is_threat: bool
    boundary_offset_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class SyntheticAudioGenerator:
    def __init__(self, sample_rate: int = 44100, seed: int = 42):
        self.sample_rate = sample_rate
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def reset_seed(self, seed: Optional[int] = None):
        """Reset RNG to fixed seed for repeatable corpus generation."""
        if seed is not None:
            self.seed = seed
        self.rng = np.random.RandomState(self.seed)

    def generate_ambient_background(
        self, duration_sec: float, rms_level: float = 30.0
    ) -> np.ndarray:
        """Generate low-level pinkish room noise."""
        num_samples = int(duration_sec * self.sample_rate)
        white = self.rng.normal(0, 1.0, num_samples)
        pink = np.zeros(num_samples, dtype=np.float64)
        for i in range(1, num_samples):
            pink[i] = 0.95 * pink[i - 1] + 0.05 * white[i]
        curr_rms = np.sqrt(np.mean(pink ** 2))
        if curr_rms > 0:
            pink = pink * (rms_level / curr_rms)
        return np.clip(pink, -32768, 32767).astype(np.int16)

    def generate_gunshot(
        self,
        duration_sec: float = 0.5,
        peak_level: float = 28000.0,
        reverb_decay_sec: float = 0.15,
        clip: bool = False,
    ) -> np.ndarray:
        """
        Synthesize an acoustic gunshot:
        1. Supersonic N-wave shockwave (rise < 0.2ms, duration ~1ms)
        2. Muzzle blast impulse (high broadband acoustic pulse 400-2500Hz)
        3. Indoor room reflections and exponential reverberation decay
        """
        n_samples = int(duration_sec * self.sample_rate)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        audio = np.zeros(n_samples, dtype=np.float64)

        shock_duration = int(0.0015 * self.sample_rate)
        if shock_duration > 0:
            shock = np.sin(2 * np.pi * (1.0 / 0.0015) * t[:shock_duration]) * np.exp(
                -t[:shock_duration] / 0.0005
            )
            audio[:shock_duration] += shock * 0.8

        noise = self.rng.normal(0, 1.0, n_samples)
        blast_freq = 1200.0
        osc = np.sin(2 * np.pi * blast_freq * t)
        blast = noise * osc * np.exp(-t / 0.015)
        audio += blast * 0.9

        for delay_ms, decay in [(8, 0.45), (15, 0.35), (28, 0.25)]:
            delay_samples = int(delay_ms * 0.001 * self.sample_rate)
            if delay_samples < n_samples:
                audio[delay_samples:] += audio[:-delay_samples] * decay

        reverb_env = np.exp(-t / max(0.01, reverb_decay_sec))
        reverb_noise = self.rng.normal(0, 0.2, n_samples) * reverb_env
        audio += reverb_noise

        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio * (peak_level / max_val)

        if clip:
            audio = np.clip(audio * 2.5, -32767, 32767)
        else:
            audio = np.clip(audio, -32768, 32767)

        return audio.astype(np.int16)

    def generate_door_slam(
        self, duration_sec: float = 0.6, peak_level: float = 24000.0
    ) -> np.ndarray:
        """Heavy bass impact resonance (60-120 Hz) with low high-frequency content."""
        n_samples = int(duration_sec * self.sample_rate)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        f0 = 85.0
        audio = (
            np.sin(2 * np.pi * f0 * t) * np.exp(-t / 0.12)
            + 0.5 * np.sin(2 * np.pi * 140 * t) * np.exp(-t / 0.08)
            + 0.2 * self.rng.normal(0, 1.0, n_samples) * np.exp(-t / 0.03)
        )
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio * (peak_level / max_val)
        return np.clip(audio, -32768, 32767).astype(np.int16)

    def generate_hand_clap(
        self, duration_sec: float = 0.3, peak_level: float = 22000.0
    ) -> np.ndarray:
        """Mid-frequency burst (2-4 kHz) with fast decay ~30ms."""
        n_samples = int(duration_sec * self.sample_rate)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        noise = self.rng.normal(0, 1.0, n_samples)
        carrier = np.sin(2 * np.pi * 2800 * t) + 0.5 * np.sin(2 * np.pi * 3600 * t)
        audio = noise * carrier * np.exp(-t / 0.025)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio * (peak_level / max_val)
        return np.clip(audio, -32768, 32767).astype(np.int16)

    def generate_balloon_pop(
        self, duration_sec: float = 0.4, peak_level: float = 26000.0
    ) -> np.ndarray:
        """Sudden acoustic pressure release: sharp crack followed by air resonance."""
        n_samples = int(duration_sec * self.sample_rate)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        audio = (
            self.rng.normal(0, 1.0, n_samples) * np.exp(-t / 0.008)
            + 0.6 * np.sin(2 * np.pi * 650 * t) * np.exp(-t / 0.05)
        )
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio * (peak_level / max_val)
        return np.clip(audio, -32768, 32767).astype(np.int16)

    def generate_dropped_book(
        self, duration_sec: float = 0.5, peak_level: float = 23000.0
    ) -> np.ndarray:
        """Two-phase impact: corner hit followed 20ms later by full slap."""
        n_samples = int(duration_sec * self.sample_rate)
        audio = np.zeros(n_samples, dtype=np.float64)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)

        audio += np.sin(2 * np.pi * 300 * t) * np.exp(-t / 0.02) * 0.4
        offset = int(0.025 * self.sample_rate)
        if offset < n_samples:
            t2 = t[: n_samples - offset]
            audio[offset:] += (
                (np.sin(2 * np.pi * 180 * t2) + 0.3 * self.rng.normal(0, 1.0, len(t2)))
                * np.exp(-t2 / 0.07)
                * 1.0
            )

        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio * (peak_level / max_val)
        return np.clip(audio, -32768, 32767).astype(np.int16)

    def generate_speech_shout(
        self, duration_sec: float = 1.0, peak_level: float = 18000.0
    ) -> np.ndarray:
        """Voiced shout with sustained harmonic formants (low peak-to-RMS ratio)."""
        n_samples = int(duration_sec * self.sample_rate)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        f0 = 220.0
        harmonics = [1, 2, 3, 4, 5, 6, 7, 8]
        audio = np.zeros(n_samples, dtype=np.float64)
        for h in harmonics:
            amp = 1.0 / (h ** 0.8)
            audio += amp * np.sin(2 * np.pi * f0 * h * t)

        attack = int(0.1 * self.sample_rate)
        decay = int(0.2 * self.sample_rate)
        env = np.ones(n_samples)
        env[:attack] = np.linspace(0, 1, attack)
        env[-decay:] = np.linspace(1, 0, decay)
        audio = audio * env

        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio * (peak_level / max_val)
        return np.clip(audio, -32768, 32767).astype(np.int16)

    def assemble_test_stream(
        self,
        duration_sec: float,
        events: List[Tuple[str, float, float, Dict[str, Any]]],
        ambient_rms: float = 35.0,
    ) -> Tuple[np.ndarray, List[GroundTruthEvent]]:
        """
        Assemble a continuous audio stream with embedded ground truth events.
        events: list of (label, start_time_sec, duration_sec, kwargs)
        """
        total_samples = int(duration_sec * self.sample_rate)
        stream = self.generate_ambient_background(duration_sec, rms_level=ambient_rms).astype(
            np.float64
        )
        gt_events = []

        generators = {
            "gunshot": self.generate_gunshot,
            "door_slam": self.generate_door_slam,
            "hand_clap": self.generate_hand_clap,
            "balloon_pop": self.generate_balloon_pop,
            "dropped_book": self.generate_dropped_book,
            "speech_shout": self.generate_speech_shout,
        }

        for idx, (label, start_sec, dur_sec, kwargs) in enumerate(events):
            gen = generators.get(label, self.generate_gunshot)
            signal = gen(duration_sec=dur_sec, **kwargs)
            start_samp = int(start_sec * self.sample_rate)
            end_samp = min(total_samples, start_samp + len(signal))
            inject_len = end_samp - start_samp
            if inject_len > 0:
                stream[start_samp:end_samp] += signal[:inject_len]

            is_threat = label == "gunshot"
            gt_events.append(
                GroundTruthEvent(
                    event_id=f"evt_{idx}_{label}",
                    label=label,
                    start_sec=start_sec,
                    end_sec=start_sec + dur_sec,
                    is_threat=is_threat,
                    metadata=kwargs,
                )
            )

        final_pcm = np.clip(stream, -32768, 32767).astype(np.int16)
        return final_pcm, gt_events

    def generate_boundary_evaluation_suite(
        self,
    ) -> Tuple[np.ndarray, List[GroundTruthEvent]]:
        """
        Generate a multi-shot stream specifically probing rolling-window boundary offsets:
        - 400ms inference hop interval
        - 975ms (15600 sample) window size
        - 5000ms cooldown between events
        """
        self.reset_seed()
        offsets_ms = [0.0, 50.0, 195.0, 390.0, 405.0, 950.0]
        events = []
        base_time = 2.0  # allow initial history buffer to fill (1s needed)

        for i, offset_ms in enumerate(offsets_ms):
            shot_time = base_time + (offset_ms / 1000.0)
            events.append(
                (
                    "gunshot",
                    shot_time,
                    0.5,
                    {"peak_level": 28000.0, "reverb_decay_sec": 0.15},
                )
            )
            base_time += 6.0  # > 5.0s cooldown so each test is independent

        total_duration = base_time + 2.0
        pcm, gt_events = self.assemble_test_stream(total_duration, events)
        for gt, offset in zip(gt_events, offsets_ms):
            gt.boundary_offset_ms = offset

        return pcm, gt_events

    def generate_hard_negatives_suite(
        self,
    ) -> Tuple[np.ndarray, List[GroundTruthEvent]]:
        """
        Generate audio stream containing challenging impulsive negatives:
        door slams, hand claps, balloon pops, dropped books, loud shouts.
        """
        self.reset_seed()
        negative_types = [
            ("door_slam", 0.6, {"peak_level": 25000.0}),
            ("hand_clap", 0.3, {"peak_level": 22000.0}),
            ("balloon_pop", 0.4, {"peak_level": 27000.0}),
            ("dropped_book", 0.5, {"peak_level": 24000.0}),
            ("speech_shout", 1.0, {"peak_level": 20000.0}),
            ("hand_clap", 0.3, {"peak_level": 26000.0}),
            ("door_slam", 0.6, {"peak_level": 28000.0}),
        ]

        events = []
        base_time = 2.0
        for label, dur, kwargs in negative_types:
            events.append((label, base_time, dur, kwargs))
            base_time += 6.0

        total_duration = base_time + 2.0
        return self.assemble_test_stream(total_duration, events)
