#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image
from tqdm import tqdm

ort.preload_dlls(directory="")


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera captions con WD ViT Tagger v3."
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Carpeta que contiene las imágenes.",
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/wd-vit-tagger-v3"),
        help="Carpeta que contiene model.onnx y selected_tags.csv.",
    )

    parser.add_argument(
        "--general-threshold",
        type=float,
        default=0.35,
        help="Umbral para etiquetas generales.",
    )

    parser.add_argument(
        "--character-threshold",
        type=float,
        default=0.85,
        help="Umbral para nombres de personajes.",
    )

    parser.add_argument(
        "--include-characters",
        action="store_true",
        help="Incluye nombres de personajes detectados.",
    )

    parser.add_argument(
        "--include-rating",
        action="store_true",
        help="Incluye las etiquetas de clasificación.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Busca imágenes en subcarpetas.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Carpeta donde guardar captions. "
            "Por defecto se guardan junto a las imágenes."
        ),
    )

    parser.add_argument(
        "--trigger",
        type=str,
        default="",
        help="Token que se agrega al comienzo de cada caption.",
    )

    return parser.parse_args()


def get_execution_providers() -> list[str]:
    available = ort.get_available_providers()

    if "CUDAExecutionProvider" in available:
        return [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    return ["CPUExecutionProvider"]


def load_image(image_path: Path, image_size: int) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGBA")

        # WD Tagger espera un fondo blanco para las transparencias.
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image).convert("RGB")

        width, height = image.size
        square_size = max(width, height)

        square = Image.new(
            "RGB",
            (square_size, square_size),
            (255, 255, 255),
        )

        offset = (
            (square_size - width) // 2,
            (square_size - height) // 2,
        )

        square.paste(image, offset)

        square = square.resize(
            (image_size, image_size),
            Image.Resampling.BICUBIC,
        )

        array = np.asarray(square, dtype=np.float32)

    # El modelo ONNX usa el orden BGR.
    array = array[:, :, ::-1]

    # Añadir dimensión de batch: HWC -> NHWC.
    return np.expand_dims(array, axis=0)


def load_tags(tags_path: Path) -> pd.DataFrame:
    tags = pd.read_csv(tags_path)

    required_columns = {"name", "category"}

    if not required_columns.issubset(tags.columns):
        raise ValueError(
            f"{tags_path} no contiene las columnas requeridas: "
            f"{required_columns}"
        )

    return tags


def select_tags(
    probabilities: np.ndarray,
    tags: pd.DataFrame,
    general_threshold: float,
    character_threshold: float,
    include_characters: bool,
    include_rating: bool,
) -> list[tuple[str, float]]:
    selected: list[tuple[str, float]] = []

    for probability, row in zip(probabilities, tags.itertuples()):
        name = str(row.name)
        category = int(row.category)
        probability = float(probability)

        # Categorías correctas:
        # 0: general
        # 4: character
        # 9: rating
        if category == 0:
            if probability >= general_threshold:
                selected.append((name, probability))

        elif category == 4:
            if include_characters and probability >= character_threshold:
                selected.append((name, probability))

        elif category == 9:
            if include_rating and probability >= general_threshold:
                selected.append((name, probability))

    selected.sort(key=lambda item: item[1], reverse=True)
    return selected


def normalize_tag(tag: str) -> str:
    # Los tags de Danbooru usan guion bajo.
    return tag.replace("_", " ")


def create_caption(
    selected_tags: list[tuple[str, float]],
    trigger: str,
) -> str:
    names = [normalize_tag(name) for name, _ in selected_tags]

    if trigger.strip():
        names.insert(0, trigger.strip())

    return ", ".join(names)


def find_images(
    input_dir: Path,
    recursive: bool,
) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")

    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_output_path(
    image_path: Path,
    input_dir: Path,
    output_dir: Path | None,
) -> Path:
    if output_dir is None:
        return image_path.with_suffix(".txt")

    relative_path = image_path.relative_to(input_dir)
    caption_path = output_dir / relative_path
    caption_path = caption_path.with_suffix(".txt")
    caption_path.parent.mkdir(parents=True, exist_ok=True)

    return caption_path


def main() -> None:
    args = parse_args()

    model_path = args.model_dir / "model.onnx"
    tags_path = args.model_dir / "selected_tags.csv"

    if not args.input_dir.is_dir():
        raise FileNotFoundError(
            f"No existe la carpeta de entrada: {args.input_dir}"
        )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"No se encontró el modelo: {model_path}"
        )

    if not tags_path.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo de tags: {tags_path}"
        )

    providers = get_execution_providers()
    print(f"Providers disponibles: {ort.get_available_providers()}")
    print(f"Providers utilizados: {providers}")

    start = perf_counter()

    session = ort.InferenceSession(
        str(model_path),
        providers=providers,
    )

    print(f"Creación de sesión: {perf_counter() - start:.3f} s")
    print(f"Providers activos: {session.get_providers()}")    
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape

    # Normalmente la entrada es [batch, height, width, channels].
    image_size = int(input_shape[1])

    tags = load_tags(tags_path)
    images = find_images(args.input_dir, args.recursive)

    if not images:
        raise RuntimeError(
            f"No se encontraron imágenes en {args.input_dir}"
        )

    print(f"Modelo: {model_path}")
    print(f"Resolución de entrada: {image_size}x{image_size}")
    print(f"Imágenes encontradas: {len(images)}")

    total_start = perf_counter()

    for image_path in tqdm(images, desc="Generando captions"):
        try:
            image_start = perf_counter()

            load_start = perf_counter()
            input_tensor = load_image(image_path, image_size)
            load_time = perf_counter() - load_start

            inference_start = perf_counter()
            outputs = session.run(
                None,
                {input_name: input_tensor},
            )
            inference_time = perf_counter() - inference_start

            postprocess_start = perf_counter()

            probabilities = outputs[0][0]

            selected = select_tags(
                probabilities=probabilities,
                tags=tags,
                general_threshold=args.general_threshold,
                character_threshold=args.character_threshold,
                include_characters=args.include_characters,
                include_rating=args.include_rating,
            )

            caption = create_caption(
                selected_tags=selected,
                trigger=args.trigger,
            )

            output_path = get_output_path(
                image_path=image_path,
                input_dir=args.input_dir,
                output_dir=args.output_dir,
            )

            output_path.write_text(
                caption + "\n",
                encoding="utf-8",
            )

            postprocess_time = perf_counter() - postprocess_start
            image_time = perf_counter() - image_start

            tqdm.write(
                f"{image_path.name}: "
                f"carga={load_time:.4f}s | "
                f"inferencia={inference_time:.4f}s | "
                f"postproceso={postprocess_time:.4f}s | "
                f"total={image_time:.4f}s"
            )

        except Exception as error:
            tqdm.write(f"Error procesando {image_path}: {error}")

    total_time = perf_counter() - total_start

    print(f"Proceso terminado en {total_time:.4f} segundos.")


if __name__ == "__main__":
    main()