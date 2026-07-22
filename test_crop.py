#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

EXTRACTOR_ROOT = (
    PROJECT_ROOT
    / "external"
    / "manga-panel-extractor"
)

EXTRACTOR_SRC = EXTRACTOR_ROOT / "src"

sys.path.insert(0, str(EXTRACTOR_SRC))

from image_processing.panel import generate_panel_blocks_by_ai, MergeMode


INPUT_IMAGE = PROJECT_ROOT / "pages" / "test.webp"
OUTPUT_DIR = PROJECT_ROOT / "images" / "crop_test"

def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"No se pudo abrir la imagen: {path}")

    return image


def save_image(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", image)

    if not success:
        raise RuntimeError(f"No se pudo guardar la imagen: {path}")

    path.write_bytes(encoded.tobytes())


def main() -> None:
    if not INPUT_IMAGE.is_file():
        raise FileNotFoundError(f"No existe la imagen: {INPUT_IMAGE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image = read_image(INPUT_IMAGE)

    panels = generate_panel_blocks_by_ai(
        image=image,
        merge=MergeMode.NONE,
    )

    print(f"Paneles detectados: {len(panels)}")

    for i, panel in enumerate(panels):
        out_path = OUTPUT_DIR / f"{INPUT_IMAGE.stem}_panel_{i:03d}.png"
        save_image(out_path, panel)
        print(out_path)


if __name__ == "__main__":
    main()
