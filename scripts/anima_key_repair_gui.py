import argparse
import gc
import hashlib
import json
import os
import struct
import sys
import uuid
from pathlib import Path

import torch
from safetensors import safe_open
from PySide6 import QtCore, QtWidgets


ANIMA_DIT_KEY_PREFIXES = (
    "pipe.dit.",
    "dit.",
    "model.diffusion_model.",
    "diffusion_model.",
    "model.",
    "net.",
)

ANIMA_DIT_KEY_MARKERS = (
    "blocks.",
    "llm_adapter.",
    "final_layer.",
    "t_embedder.",
    "t_embedding_norm.",
    "x_embedder.",
)

SAFETENSORS_DTYPE_NAMES = {
    torch.float64: "F64",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}
if hasattr(torch, "float8_e4m3fn"):
    SAFETENSORS_DTYPE_NAMES[torch.float8_e4m3fn] = "F8_E4M3"
if hasattr(torch, "float8_e5m2"):
    SAFETENSORS_DTYPE_NAMES[torch.float8_e5m2] = "F8_E5M2"

SAFETENSORS_DTYPE_SIZES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}
KNOWN_ANIMA_DIT_HASH = "417673936471e79e31ed4d186d7a3f4a"

STYLE = """
QWidget { background: #15171b; color: #e7e9ee; font-family: Segoe UI, sans-serif; font-size: 13px; }
QGroupBox { border: 1px solid #353b45; border-radius: 6px; margin-top: 10px; padding: 10px; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 6px; color: #78c2ad; font-weight: 600; }
QPushButton { background: #242a31; border: 1px solid #3d4651; border-radius: 5px; padding: 7px 10px; }
QPushButton:hover { border-color: #78c2ad; color: #78c2ad; }
QPushButton:disabled { color: #69717d; border-color: #2b3037; }
QLineEdit { background: #101216; border: 1px solid #353b45; border-radius: 5px; padding: 6px; }
QPlainTextEdit { background: #101216; border: 1px solid #353b45; border-radius: 5px; font-family: Consolas, monospace; }
QProgressBar { border: 1px solid #353b45; border-radius: 5px; text-align: center; background: #101216; }
QProgressBar::chunk { background: #78c2ad; }
QCheckBox { spacing: 8px; }
"""


def strip_anima_dit_key_prefix(key: str) -> tuple[str, str]:
    for prefix in ANIMA_DIT_KEY_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):], prefix
    return key, ""


def dtype_name(dtype) -> str:
    if isinstance(dtype, str):
        if dtype not in SAFETENSORS_DTYPE_SIZES:
            raise TypeError(f"Unsupported safetensors dtype: {dtype}")
        return dtype
    name = SAFETENSORS_DTYPE_NAMES.get(dtype)
    if name is None:
        raise TypeError(f"Unsupported safetensors dtype: {dtype}")
    return name


def dtype_element_size(dtype) -> int:
    if isinstance(dtype, str):
        try:
            return SAFETENSORS_DTYPE_SIZES[dtype]
        except KeyError as exc:
            raise TypeError(f"Unsupported safetensors dtype: {dtype}") from exc
    return torch.empty((), dtype=dtype).element_size()


def stream_tensor_to_file(handle, tensor: torch.Tensor) -> None:
    tensor.reshape(-1).view(torch.uint8).numpy().tofile(handle)


def structural_hash_from_records(records: list[dict], key_name: str = "new_key") -> str:
    keys = []
    for record in records:
        key = record[key_name]
        shape = "_".join(map(str, record["shape"]))
        keys.append(f"{key}:{shape}")
        keys.append(key)
    keys.sort()
    return hashlib.md5(",".join(keys).encode("UTF-8")).hexdigest()


def analyze_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".safetensors":
        raise ValueError("Only .safetensors files are supported.")

    records = []
    prefix_counts = {prefix: 0 for prefix in ANIMA_DIT_KEY_PREFIXES}
    marker_count = 0
    total_bytes = 0
    seen = set()
    duplicates = []

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        for old_key in handle.keys():
            new_key, prefix = strip_anima_dit_key_prefix(old_key)
            if prefix:
                prefix_counts[prefix] += 1
            if new_key.startswith(ANIMA_DIT_KEY_MARKERS):
                marker_count += 1
            if new_key in seen:
                duplicates.append(new_key)
            seen.add(new_key)
            tensor_slice = handle.get_slice(old_key)
            shape = list(tensor_slice.get_shape())
            dtype = tensor_slice.get_dtype()
            total_bytes += dtype_element_size(dtype) * int(torch.tensor(shape).prod().item() if shape else 1)
            records.append({"old_key": old_key, "new_key": new_key, "shape": shape, "dtype": dtype})

    if not records:
        raise ValueError("No tensors were found in the checkpoint.")

    dominant_prefix, dominant_count = max(prefix_counts.items(), key=lambda item: item[1])
    dominant_prefix = dominant_prefix if dominant_count / len(records) >= 0.8 else ""
    looks_like_anima = marker_count / len(records) >= 0.8
    original_hash = structural_hash_from_records(records, key_name="old_key")
    repaired_hash = structural_hash_from_records(records, key_name="new_key")
    changed = sum(1 for record in records if record["old_key"] != record["new_key"])

    return {
        "path": path,
        "records": records,
        "metadata": metadata,
        "tensor_count": len(records),
        "changed_count": changed,
        "dominant_prefix": dominant_prefix,
        "looks_like_anima": looks_like_anima,
        "duplicates": duplicates,
        "original_hash": original_hash,
        "repaired_hash": repaired_hash,
        "total_bytes": total_bytes,
    }


def write_repaired_checkpoint(input_path: Path, output_path: Path, analysis: dict, progress=None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    records = analysis["records"]
    metadata = dict(analysis.get("metadata") or {})
    metadata.update(
        {
            "aozora_anima_key_repair": "1",
            "aozora_original_structural_hash": analysis["original_hash"],
            "aozora_repaired_structural_hash": analysis["repaired_hash"],
            "aozora_original_key_prefix": analysis["dominant_prefix"] or "none",
        }
    )

    header = {"__metadata__": {str(k): str(v) for k, v in metadata.items()}}
    offset = 0
    for record in records:
        dtype = record["dtype"]
        nbytes = dtype_element_size(dtype) * int(torch.tensor(record["shape"]).prod().item() if record["shape"] else 1)
        header[record["new_key"]] = {
            "dtype": dtype_name(dtype),
            "shape": record["shape"],
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((8 - (len(header_bytes) % 8)) % 8)

    try:
        with safe_open(str(input_path), framework="pt", device="cpu") as source, open(tmp_path, "wb") as out:
            out.write(struct.pack("<Q", len(header_bytes)))
            out.write(header_bytes)
            for index, record in enumerate(records, start=1):
                tensor = source.get_tensor(record["old_key"]).detach().cpu().contiguous()
                stream_tensor_to_file(out, tensor)
                del tensor
                if progress:
                    progress(index, len(records), record["new_key"])
                if index % 16 == 0:
                    gc.collect()
        os.replace(tmp_path, output_path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


class RepairWorker(QtCore.QObject):
    progress = QtCore.Signal(int, str)
    message = QtCore.Signal(str)
    finished = QtCore.Signal(bool, str)

    def __init__(self, input_path: Path, output_path: Path):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path

    @QtCore.Slot()
    def run(self):
        try:
            analysis = analyze_checkpoint(self.input_path)
            if analysis["duplicates"]:
                raise RuntimeError(f"Repair would create duplicate keys, first duplicate: {analysis['duplicates'][0]}")
            if not analysis["looks_like_anima"]:
                raise RuntimeError("Checkpoint keys do not look like an Anima DiT state dict.")
            self.message.emit(f"Original hash: {analysis['original_hash']}")
            self.message.emit(f"Repaired hash: {analysis['repaired_hash']}")
            self.message.emit(f"Dominant prefix: {analysis['dominant_prefix'] or 'none'}")

            def progress(index, total, key):
                pct = int(index * 100 / max(total, 1))
                self.progress.emit(pct, key)

            write_repaired_checkpoint(self.input_path, self.output_path, analysis, progress=progress)
            self.message.emit(f"Saved repaired checkpoint: {self.output_path}")
            self.finished.emit(True, "Repair complete.")
        except Exception as exc:
            self.finished.emit(False, str(exc))


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Anima DiT Key Repair")
        self.resize(820, 520)
        self.analysis = None
        self.thread = None
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)

        paths_group = QtWidgets.QGroupBox("Files")
        paths_layout = QtWidgets.QGridLayout(paths_group)
        self.input_edit = QtWidgets.QLineEdit()
        self.output_edit = QtWidgets.QLineEdit()
        input_btn = QtWidgets.QPushButton("Browse")
        output_btn = QtWidgets.QPushButton("Browse")
        input_btn.clicked.connect(self.browse_input)
        output_btn.clicked.connect(self.browse_output)
        paths_layout.addWidget(QtWidgets.QLabel("Input"), 0, 0)
        paths_layout.addWidget(self.input_edit, 0, 1)
        paths_layout.addWidget(input_btn, 0, 2)
        paths_layout.addWidget(QtWidgets.QLabel("Output"), 1, 0)
        paths_layout.addWidget(self.output_edit, 1, 1)
        paths_layout.addWidget(output_btn, 1, 2)
        root.addWidget(paths_group)

        actions = QtWidgets.QHBoxLayout()
        self.analyze_btn = QtWidgets.QPushButton("Analyze")
        self.repair_btn = QtWidgets.QPushButton("Repair")
        self.analyze_btn.clicked.connect(self.analyze)
        self.repair_btn.clicked.connect(self.repair)
        actions.addWidget(self.analyze_btn)
        actions.addWidget(self.repair_btn)
        actions.addStretch(1)
        root.addLayout(actions)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        root.addWidget(self.progress)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        root.addWidget(self.log, 1)

    def log_line(self, text: str):
        self.log.appendPlainText(text)

    def browse_input(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Anima DiT", "", "Safetensors (*.safetensors);;All files (*)")
        if path:
            self.input_edit.setText(path)
            input_path = Path(path)
            self.output_edit.setText(str(input_path.with_name(input_path.stem + "_aozora.safetensors")))
            self.analyze()

    def browse_output(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save repaired Anima DiT", self.output_edit.text(), "Safetensors (*.safetensors);;All files (*)")
        if path:
            if not path.lower().endswith(".safetensors"):
                path += ".safetensors"
            self.output_edit.setText(path)

    def analyze(self):
        try:
            input_path = Path(self.input_edit.text().strip())
            self.analysis = analyze_checkpoint(input_path)
            self.log.clear()
            self.log_line(f"Tensors: {self.analysis['tensor_count']}")
            self.log_line(f"Keys changed: {self.analysis['changed_count']}")
            self.log_line(f"Dominant prefix: {self.analysis['dominant_prefix'] or 'none'}")
            self.log_line(f"Looks like Anima DiT: {self.analysis['looks_like_anima']}")
            self.log_line(f"Original structural hash: {self.analysis['original_hash']}")
            self.log_line(f"Repaired structural hash: {self.analysis['repaired_hash']}")
            self.log_line(f"Bundled Anima hash match: {self.analysis['repaired_hash'] == KNOWN_ANIMA_DIT_HASH}")
            if self.analysis["duplicates"]:
                self.log_line(f"Duplicate repaired key: {self.analysis['duplicates'][0]}")
        except Exception as exc:
            self.analysis = None
            self.log_line(f"ERROR: {exc}")

    def repair(self):
        input_path = Path(self.input_edit.text().strip())
        output_path = Path(self.output_edit.text().strip())
        if not input_path.exists():
            self.log_line("ERROR: Input file does not exist.")
            return
        if not output_path:
            self.log_line("ERROR: Output file is required.")
            return
        self.progress.setValue(0)
        self.repair_btn.setEnabled(False)
        self.analyze_btn.setEnabled(False)
        self.thread = QtCore.QThread(self)
        self.worker = RepairWorker(input_path, output_path)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(lambda value, key: self.progress.setValue(value))
        self.worker.message.connect(self.log_line)
        self.worker.finished.connect(self.on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_finished(self, ok: bool, message: str):
        self.progress.setValue(100 if ok else self.progress.value())
        self.log_line(message if ok else f"ERROR: {message}")
        self.repair_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True)
        self.thread = None
        self.worker = None


def run_cli(args) -> int:
    input_path = Path(args.input)
    output_path = Path(args.output)
    analysis = analyze_checkpoint(input_path)
    print(f"Original hash: {analysis['original_hash']}")
    print(f"Repaired hash: {analysis['repaired_hash']}")
    print(f"Dominant prefix: {analysis['dominant_prefix'] or 'none'}")
    if analysis["duplicates"]:
        raise RuntimeError(f"Repair would create duplicate keys, first duplicate: {analysis['duplicates'][0]}")
    if not analysis["looks_like_anima"]:
        raise RuntimeError("Checkpoint keys do not look like an Anima DiT state dict.")
    if args.analyze_only:
        return 0
    write_repaired_checkpoint(input_path, output_path, analysis)
    print(f"Saved repaired checkpoint: {output_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair Anima DiT safetensors keys for Aozora loading.")
    parser.add_argument("--input", help="Input .safetensors checkpoint")
    parser.add_argument("--output", help="Output repaired .safetensors checkpoint")
    parser.add_argument("--analyze-only", action="store_true", help="Analyze input and print hashes without writing output")
    args = parser.parse_args()
    if args.input or args.output:
        if not args.input or not args.output:
            parser.error("--input and --output must be used together")
        return run_cli(args)

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
