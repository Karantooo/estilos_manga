#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

SUBMODULE_DIR="$PROJECT_ROOT/external/manga-panel-extractor"
PATCH_FILE="$PROJECT_ROOT/patches/manga-panel-extractor-linux.patch"

echo "Raíz del proyecto: $PROJECT_ROOT"

git -C "$PROJECT_ROOT" submodule update --init --recursive

if [[ ! -d "$SUBMODULE_DIR" ]]; then
    echo "No existe el submódulo: $SUBMODULE_DIR" >&2
    exit 1
fi

if [[ ! -f "$PATCH_FILE" ]]; then
    echo "No existe el parche: $PATCH_FILE" >&2
    exit 1
fi

if git -C "$SUBMODULE_DIR" apply --check "$PATCH_FILE" 2>/dev/null; then
    git -C "$SUBMODULE_DIR" apply "$PATCH_FILE"
    echo "Parche aplicado correctamente."
elif git -C "$SUBMODULE_DIR" apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
    echo "El parche ya estaba aplicado."
else
    echo "El parche no puede aplicarse limpiamente." >&2
    echo "Revisa los cambios dentro del submódulo:" >&2
    echo "  git -C \"$SUBMODULE_DIR\" status" >&2
    exit 1
fi