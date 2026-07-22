# Estilos Manga

Proyecto experimental para preparar un dataset de manga destinado al entrenamiento de modelos de difusión, especialmente LoRA de estilo.

El pipeline previsto contempla:

- extracción de paneles desde páginas de manga;
- filtrado de crops defectuosos o poco útiles;
- generación automática de captions;
- entrenamiento y evaluación de modelos de estilo.

## Estado actual

Actualmente el proyecto utiliza **WD ViT Tagger v3** para generar tags automáticos a partir de imágenes manga.

El modelo se ejecuta localmente mediante ONNX Runtime y utiliza:

- `model.onnx`: pesos y grafo del modelo;
- `selected_tags.csv`: vocabulario que relaciona cada salida del modelo con su tag y categoría.

## Dependencias

```bash
pip install numpy pandas pillow tqdm huggingface-hub onnxruntime-gpu jupyter
```

Para ejecutar solo en CPU:

```bash
pip install onnxruntime
```

## Archivos principales

### `tag_images.py`

Procesa las imágenes de una carpeta y genera un archivo `.txt` con los tags detectados.

Ejemplo:

```bash
python tag_images.py images --trigger mangastyle123
```

### `analisis_tags.ipynb`

Notebook provisional para revisar los captions generados, analizar frecuencias, detectar tags redundantes y probar reglas de limpieza.

## Carpetas

### `images/`

Contiene imágenes de prueba y sus captions generados.

### `models/`

Contiene los modelos utilizados por el proyecto. Actualmente incluye `wd-vit-tagger-v3`.

Los archivos `.gitkeep` permiten conservar estas carpetas en Git aunque estén vacías.

## Estructura actual

```text
.
├── images/
│   └── .gitkeep
├── models/
│   ├── .gitkeep
│   └── wd-vit-tagger-v3/
│       ├── model.onnx
│       └── selected_tags.csv
├── analisis_tags.ipynb
├── tag_images.py
├── .gitignore
└── README.md
```