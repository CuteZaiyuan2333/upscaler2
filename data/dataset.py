"""Dataset loader for folder-based semantic transfer training."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _list_image_files(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory not found: {folder}")

    files: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            files[path.stem] = path
    return files


def _load_rgb_image(path: Path, size: int) -> torch.Tensor:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        if rgb.size != (size, size):
            rgb = rgb.resize((size, size), Image.Resampling.LANCZOS)
    tensor = transforms.ToTensor()(rgb)
    return tensor * 2.0 - 1.0


class FolderSemanticTransferDataset(Dataset):
    """
    Reads triplets from ``dataset/input``, ``dataset/output``, ``dataset/reference``.

    Files with the same stem (filename without extension) form one training sample.
    Example: ``input/001.png``, ``output/001.png``, ``reference/001.png``.
    """

    def __init__(
        self,
        root: str | Path,
        input_size: int = 1024,
        ref_size: int = 256,
        samples: list[str] | None = None,
    ):
        self.root = Path(root)
        self.input_dir = self.root / "input"
        self.output_dir = self.root / "output"
        self.reference_dir = self.root / "reference"
        self.input_size = input_size
        self.ref_size = ref_size

        input_files = _list_image_files(self.input_dir)
        output_files = _list_image_files(self.output_dir)
        reference_files = _list_image_files(self.reference_dir)

        common = sorted(set(input_files) & set(output_files) & set(reference_files))
        if samples is not None:
            missing = sorted(set(samples) - set(common))
            if missing:
                raise ValueError(f"Requested samples not found in all folders: {missing}")
            common = sorted(samples)

        if not common:
            raise RuntimeError(
                f"No matching samples under {self.root}. "
                "Place same-named images in input/, output/, and reference/."
            )

        self.samples = common
        self.input_files = input_files
        self.output_files = output_files
        self.reference_files = reference_files

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        stem = self.samples[idx]
        return {
            "input": _load_rgb_image(self.input_files[stem], self.input_size),
            "target": _load_rgb_image(self.output_files[stem], self.input_size),
            "ref": _load_rgb_image(self.reference_files[stem], self.ref_size),
            "name": stem,
        }
