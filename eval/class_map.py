"""Class map loader and label selection for YAMNet acoustic classifier."""

import csv
import os
from typing import List, Tuple, Set

DEFAULT_CLASS_MAP_PATH = os.path.join(
    os.path.dirname(__file__), "..", "app", "src", "main", "assets", "yamnet_class_map.csv"
)

# Strict gun-related classes (TIGHTENED)
# Only direct firearms/gunshot events
TIGHTENED_CLASS_SUBSTRINGS = [
    "gunshot",
    "gunfire",
    "machine gun",
    "fusillade",
    "artillery fire",
]

# Legacy overly-broad keywords that catch bangs, door slams, pops, fireworks
LEGACY_CLASS_SUBSTRINGS = [
    "gunshot",
    "gunfire",
    "machine gun",
    "explosion",
    "artillery",
    "fusillade",
    "cap gun",
    "burst",
    "bang",
]


class YamnetClassMap:
    def __init__(self, class_map_path: str = DEFAULT_CLASS_MAP_PATH):
        self.class_map_path = os.path.abspath(class_map_path)
        self.labels: List[str] = []
        self.mids: List[str] = []
        self._load()

    def _load(self):
        if not os.path.exists(self.class_map_path):
            raise FileNotFoundError(f"Class map not found at {self.class_map_path}")

        labels = []
        mids = []
        with open(self.class_map_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)  # index,mid,display_name
            for row in reader:
                if len(row) >= 3:
                    # row[0] = index, row[1] = mid, row[2] = display_name
                    mids.append(row[1].strip())
                    labels.append(row[2].strip())
        self.labels = labels
        self.mids = mids

    def get_indices(self, mode: str = "tightened") -> List[int]:
        """
        Return target class indices based on selection mode:
        - 'tightened': Only direct gunshots/gunfire (421, 422, 423, 424)
        - 'legacy': Includes bang, explosion, cap gun, pop, burst
        """
        indices = []
        if mode == "tightened":
            for idx, label in enumerate(self.labels):
                low = label.lower()
                if any(sub in low for sub in TIGHTENED_CLASS_SUBSTRINGS):
                    indices.append(idx)
        elif mode == "legacy":
            for idx, label in enumerate(self.labels):
                low = label.lower()
                is_match = (
                    any(sub in low for sub in LEGACY_CLASS_SUBSTRINGS)
                    or ("pop" in low and "music" not in low)
                )
                if is_match:
                    indices.append(idx)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'tightened' or 'legacy'.")
        return indices

    def get_label(self, index: int) -> str:
        if 0 <= index < len(self.labels):
            return self.labels[index]
        return "Unknown"

    def __len__(self):
        return len(self.labels)
