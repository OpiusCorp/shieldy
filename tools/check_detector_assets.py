#!/usr/bin/env python3
"""Check the bundled YAMNet contract using only the Python standard library."""

import csv
from pathlib import Path
import struct


class FlatBuffer:
    """Read only the FlatBuffer fields needed for TFLite tensor metadata."""

    def __init__(self, data):
        self.data = data

    def uint(self, position):
        return struct.unpack_from("<I", self.data, position)[0]

    def field(self, table, index):
        vtable = table - struct.unpack_from("<i", self.data, table)[0]
        size = struct.unpack_from("<H", self.data, vtable)[0]
        slot = 4 + index * 2
        offset = struct.unpack_from("<H", self.data, vtable + slot)[0] if slot < size else 0
        return table + offset if offset else None

    def vector(self, table, index, tables=False):
        field = self.field(table, index)
        if field is None:
            raise ValueError(f"Missing vector field {index}")
        start = field + self.uint(field)
        positions = [start + 4 + i * 4 for i in range(self.uint(start))]
        return [p + self.uint(p) if tables else self.uint(p) for p in positions]

    def tensor(self, table):
        shape = self.vector(table, 0)
        dtype = self.field(table, 1)
        # TensorType.FLOAT32 is zero and can be omitted from a FlatBuffer table.
        dtype = self.data[dtype] if dtype is not None else 0
        return shape, dtype


def main():
    assets = Path(__file__).resolve().parents[1] / "app/src/main/assets"
    with (assets / "yamnet_class_map.csv").open(newline="") as source:
        labels = list(csv.DictReader(source))
    if [int(row["index"]) for row in labels] != list(range(521)):
        raise ValueError("Expected 521 contiguous YAMNet class indices")
    firearm_labels = {"Gunshot, gunfire", "Machine gun", "Fusillade", "Artillery fire"}
    actual = {row["display_name"] for row in labels}
    if not firearm_labels <= actual:
        raise ValueError("Missing firearm labels")

    data = (assets / "yamnet.tflite").read_bytes()
    if data[4:8] != b"TFL3":
        raise ValueError("Not a TFLite model")
    model = FlatBuffer(data)
    graphs = model.vector(model.uint(0), 2, tables=True)
    graph = graphs[0]
    tensors = model.vector(graph, 0, tables=True)
    inputs = [model.tensor(tensors[i]) for i in model.vector(graph, 1)]
    outputs = [model.tensor(tensors[i]) for i in model.vector(graph, 2)]
    if inputs not in [ [([15600], 0)], [([1, 15600], 0)] ]:
        raise ValueError(f"Unsupported input tensors: {inputs}")
    scores = [shape for shape, dtype in outputs if dtype == 0 and shape and shape[-1] == 521]
    if len(scores) != 1 or scores[0] not in ([521], [1, 521]):
        raise ValueError(f"Unsupported score tensors: {outputs}")
    print(f"PASS: inputs={inputs}, outputs={outputs}, labels={len(labels)}")
    print("Asset contract only; no inference or accuracy evaluation performed.")


if __name__ == "__main__":
    main()
