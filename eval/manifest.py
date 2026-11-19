"""Manifest schema and loader for labeled acoustic evaluation datasets."""

import json
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import numpy as np

from eval.synthetic_audio import GroundTruthEvent


@dataclass
class ManifestRecording:
    recording_id: str
    audio_path: Optional[str]
    duration_sec: float
    sample_rate: int
    environment: str
    events: List[GroundTruthEvent]
    is_synthetic: bool = False
    metadata: Optional[Dict[str, Any]] = None

    def load_audio(self) -> np.ndarray:
        """
        Load audio as int16 44.1kHz mono PCM array.
        Supports .wav, .npy, .raw or synthetic generator.
        """
        if self.is_synthetic or not self.audio_path or not os.path.exists(self.audio_path):
            raise FileNotFoundError(f"Audio file '{self.audio_path}' not found on disk.")

        ext = os.path.splitext(self.audio_path)[1].lower()
        if ext in [".wav", ".flac", ".ogg"]:
            import soundfile as sf
            data, sr = sf.read(self.audio_path)
            if data.ndim > 1:
                data = np.mean(data, axis=1)  # downmix mono
            if sr != self.sample_rate:
                from scipy import signal
                # Resample to target sample rate
                data = signal.resample_poly(data, self.sample_rate, sr)
            if data.dtype != np.int16:
                if np.max(np.abs(data)) <= 1.0:
                    data = (data * 32767.0).astype(np.int16)
                else:
                    data = np.clip(data, -32768, 32767).astype(np.int16)
            return data
        elif ext == ".npy":
            arr = np.load(self.audio_path)
            if arr.dtype != np.int16:
                arr = (arr * 32767.0).astype(np.int16)
            return arr.flatten()
        else:
            raise ValueError(f"Unsupported audio format: {ext}")


class EvaluationManifest:
    def __init__(self, recordings: Optional[List[ManifestRecording]] = None):
        self.recordings = recordings or []

    @classmethod
    def load_json(cls, manifest_path: str) -> "EvaluationManifest":
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        base_dir = os.path.dirname(os.path.abspath(manifest_path))
        recordings = []
        for rec_dict in data.get("recordings", []):
            audio_path = rec_dict.get("audio_path")
            if audio_path and not os.path.isabs(audio_path):
                audio_path = os.path.normpath(os.path.join(base_dir, audio_path))

            events = []
            for evt_dict in rec_dict.get("events", []):
                events.append(
                    GroundTruthEvent(
                        event_id=evt_dict["event_id"],
                        label=evt_dict["label"],
                        start_sec=float(evt_dict["start_sec"]),
                        end_sec=float(evt_dict["end_sec"]),
                        is_threat=bool(evt_dict.get("is_threat", evt_dict["label"] == "gunshot")),
                        boundary_offset_ms=evt_dict.get("boundary_offset_ms"),
                        metadata=evt_dict.get("metadata"),
                    )
                )

            recordings.append(
                ManifestRecording(
                    recording_id=rec_dict["recording_id"],
                    audio_path=audio_path,
                    duration_sec=float(rec_dict["duration_sec"]),
                    sample_rate=int(rec_dict.get("sample_rate", 44100)),
                    environment=rec_dict.get("environment", "unknown"),
                    events=events,
                    is_synthetic=bool(rec_dict.get("is_synthetic", False)),
                    metadata=rec_dict.get("metadata"),
                )
            )

        return cls(recordings)

    def save_json(self, output_path: str):
        data = {
            "version": "1.0",
            "description": "EchoShield acoustic detector evaluation manifest",
            "total_recordings": len(self.recordings),
            "recordings": [],
        }
        for rec in self.recordings:
            rec_dict = {
                "recording_id": rec.recording_id,
                "audio_path": rec.audio_path,
                "duration_sec": rec.duration_sec,
                "sample_rate": rec.sample_rate,
                "environment": rec.environment,
                "is_synthetic": rec.is_synthetic,
                "metadata": rec.metadata,
                "events": [
                    {
                        "event_id": e.event_id,
                        "label": e.label,
                        "start_sec": e.start_sec,
                        "end_sec": e.end_sec,
                        "is_threat": e.is_threat,
                        "boundary_offset_ms": e.boundary_offset_ms,
                        "metadata": e.metadata,
                    }
                    for e in rec.events
                ],
            }
            data["recordings"].append(rec_dict)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
