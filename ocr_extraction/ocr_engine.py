from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
from paddleocr import PaddleOCR

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
_OCR_ENGINE: PaddleOCR | None = None


def get_ocr_engine() -> PaddleOCR:
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        _OCR_ENGINE = PaddleOCR(
            use_textline_orientation=True,
            lang='fr',
            enable_mkldnn=False,
        )
    return _OCR_ENGINE


def _to_python(value: Any) -> Any:
    return value.tolist() if hasattr(value, 'tolist') else value


def cluster_rows(blocks: list[dict[str, Any]], y_gap: float = 10.0) -> list[list[dict[str, Any]]]:
    if not blocks:
        return []

    blocks = sorted(blocks, key=lambda block: block['y'])
    rows: list[list[dict[str, Any]]] = []
    current = [blocks[0]]

    for block in blocks[1:]:
        current_y = sum(item['y'] for item in current) / len(current)
        if abs(block['y'] - current_y) <= y_gap:
            current.append(block)
        else:
            current.sort(key=lambda item: item['x'])
            rows.append(current)
            current = [block]

    current.sort(key=lambda item: item['x'])
    rows.append(current)
    rows.sort(key=lambda row: sum(item['y'] for item in row) / len(row))
    return rows


def _make_block(text: str, bbox: Any, score: float) -> dict[str, Any] | None:
    text = str(text).strip()
    bbox = _to_python(bbox)

    if not text or not bbox:
        return None

    xs = [float(point[0]) for point in bbox]
    ys = [float(point[1]) for point in bbox]

    return {
        'text': text,
        'bbox': bbox,
        'x': sum(xs) / len(xs),
        'y': sum(ys) / len(ys),
        'x0': min(xs),
        'y0': min(ys),
        'x1': max(xs),
        'y1': max(ys),
        'score': float(score),
    }


def _parse_predict(result: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []

    for page in result:
        texts = page.get('rec_texts', [])
        polygons = page.get('rec_polys', [])
        scores = page.get('rec_scores', [])

        for text, polygon, score in zip(texts, polygons, scores):
            block = _make_block(text, polygon, score)
            if block:
                blocks.append(block)

    return blocks


def _parse_legacy(result: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []

    for page in result or []:
        for item in page or []:
            if not item or len(item) < 2:
                continue
            bbox = item[0]
            text, score = item[1]
            block = _make_block(text, bbox, score)
            if block:
                blocks.append(block)

    return blocks


def run_ocr(image_path: str | Path, y_gap: float = 10.0) -> dict[str, Any]:
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f'Image introuvable : {path}')

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f'Image illisible : {path}')

    height, width = image.shape[:2]
    engine = get_ocr_engine()

    try:
        raw_result = engine.predict(str(path))
        blocks = _parse_predict(raw_result)
        api_used = 'predict'
    except (AttributeError, TypeError):
        raw_result = engine.ocr(str(path), cls=True)
        blocks = _parse_legacy(raw_result)
        api_used = 'ocr'

    rows = cluster_rows(blocks, y_gap=y_gap)
    ordered = [block for row in rows for block in row]

    return {
        'file': path.name,
        'original_path': str(path.resolve()),
        'image_width': int(width),
        'image_height': int(height),
        'ocr_engine': 'PaddleOCR',
        'ocr_api': api_used,
        'blocks': ordered,
        'rows': [[block['text'] for block in row] for row in rows],
        'full_text': ' '.join(block['text'] for block in ordered),
        '_raw_text': [block['text'] for block in ordered],
    }


def save_json(data: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    return output


def process_folder(input_dir: str | Path, output_dir: str | Path, overwrite: bool = False) -> dict[str, int]:
    source = Path(input_dir)
    destination = Path(output_dir)

    if not source.exists():
        raise FileNotFoundError(f'Dossier introuvable : {source}')

    destination.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        path for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    processed = skipped = errors = 0

    for image_path in image_paths:
        output_path = destination / f'{image_path.stem}.json'

        if output_path.exists() and not overwrite:
            print(f'[IGNORÉ] {output_path.name}')
            skipped += 1
            continue

        print(f'[OCR] {image_path.name}')
        try:
            result = run_ocr(image_path)
            save_json(result, output_path)
            processed += 1
        except Exception as error:
            save_json({
                'file': image_path.name,
                'original_path': str(image_path.resolve()),
                'error': str(error),
            }, output_path)
            print(f'[ERREUR] {image_path.name} : {error}')
            errors += 1

    summary = {
        'images_found': len(image_paths),
        'processed': processed,
        'skipped': skipped,
        'errors': errors,
    }

    print('\n===== RÉSUMÉ OCR =====')
    for key, value in summary.items():
        print(f'{key}: {value}')

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OCR global d’un dossier.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    process_folder(args.input, args.output, args.overwrite)
