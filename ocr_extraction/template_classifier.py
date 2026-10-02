from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any

import cv2
import numpy as np

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
TARGET_WIDTH = 900
TARGET_HEIGHT = 1250
MIN_DOC_TYPE_SCORE = 2.0
MIN_DOC_TYPE_MARGIN = 0.75
MIN_TEMPLATE_SCORE = 0.34
MIN_TEMPLATE_MARGIN = 0.045
ECC_CRITERIA = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
    120,
    1e-6,
)

DOC_TYPE_RULES: dict[str, list[tuple[str, float]]] = {
    'CIN': [
        ('carte nationale d identite', 4.0),
        ('carte d identite', 3.0),
        ('numero d identifiant', 3.0),
        ('date de naissance', 0.75),
        ('lieu de naissance', 0.75),
    ],
    'RELEVE_BANCAIRE': [
        ('releve de compte bancaire', 5.0),
        ('releve de compte', 4.0),
        ('solde depart', 3.0),
        ('nouveau solde', 3.0),
        ('solde final', 2.5),
        ('total des mouvements', 3.0),
        ('total mouvements', 2.5),
    ],
    'RIB': [
        ('releve d identite bancaire', 5.0),
        ('identite bancaire', 4.0),
        ('references bancaires', 3.0),
        ('intitule du compte', 2.5),
        ('titulaire du compte', 2.0),
        ('code banque', 1.5),
        ('code swift', 2.0),
        ('bic', 1.0),
        ('iban', 1.0),
    ],
    'BULLETIN_SALAIRE': [
        ('bulletin de paie', 5.0),
        ('bulletin de salaire', 5.0),
        ('salaire brut', 2.5),
        ('salaire net imposable', 2.5),
        ('net a payer', 1.5),
        ('cotisation cnss', 1.5),
        ('retenues', 1.0),
        ('gains', 1.0),
    ],
    'ATTESTATION_TRAVAIL': [
        ('attestation de travail', 5.0),
        ('nous soussignes', 2.5),
        ('nous soussigne', 2.5),
        ('certifions que', 2.0),
        ('date d embauche', 1.5),
        ('poste occupe', 1.5),
    ],
    'QUITTANCE': [
        ('avis de facturation', 4.0),
        ('periode de consommation', 2.5),
        ('date limite de paiement', 1.5),
        ('net a payer', 1.0),
        ('lydec', 3.0),
        ('amendis', 3.0),
        ('maroc telecom', 3.0),
        ('electricite', 0.75),
        ('consommation', 0.75),
    ],
}

BANK_KEYWORDS = {
    'CIH': ['cih bank', 'cihmamc', 'cih'],
    'ATTIJARI': ['attijariwafa', 'attijari', 'bcmamamc'],
}


def normalize_text(text: Any) -> str:
    if text is None:
        return ''
    value = unicodedata.normalize('NFD', str(text))
    value = ''.join(
        character for character in value
        if unicodedata.category(character) != 'Mn'
    )
    value = re.sub(r'[^a-z0-9]+', ' ', value.lower())
    return ' '.join(value.split())


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as file:
        return json.load(file)


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    return output


def detect_doc_type(full_text: str) -> dict[str, Any]:
    normalized = normalize_text(full_text)
    scores: dict[str, float] = {}

    for doc_type, rules in DOC_TYPE_RULES.items():
        scores[doc_type] = round(sum(
            weight for keyword, weight in rules
            if normalize_text(keyword) in normalized
        ), 4)

    ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_type, best_score = ranking[0]
    second_score = ranking[1][1] if len(ranking) > 1 else 0.0
    margin = best_score - second_score

    if best_score < MIN_DOC_TYPE_SCORE or margin < MIN_DOC_TYPE_MARGIN:
        return {
            'doc_type': 'UNKNOWN',
            'status': 'TO_REVIEW',
            'confidence': 0.0,
            'score': best_score,
            'margin': margin,
            'scores': scores,
        }

    total = sum(score for _, score in ranking)
    confidence = best_score / max(total, best_score, 1.0)
    return {
        'doc_type': best_type,
        'status': 'CLASSIFIED',
        'confidence': round(float(confidence), 4),
        'score': round(float(best_score), 4),
        'margin': round(float(margin), 4),
        'scores': scores,
    }


def detect_bank_from_ocr(full_text: str) -> dict[str, Any]:
    normalized = normalize_text(full_text)
    scores = {
        bank: sum(
            1 for keyword in keywords
            if normalize_text(keyword) in normalized
        )
        for bank, keywords in BANK_KEYWORDS.items()
    }
    ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_bank, best_score = ranking[0]
    second_score = ranking[1][1] if len(ranking) > 1 else 0

    if best_score == 0 or best_score == second_score:
        return {'bank': 'UNKNOWN', 'confidence': 0.0, 'scores': scores}

    confidence = best_score / max(sum(scores.values()), best_score, 1)
    return {
        'bank': best_bank,
        'confidence': round(float(confidence), 4),
        'scores': scores,
    }


def letterbox_document(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(TARGET_WIDTH / max(width, 1), TARGET_HEIGHT / max(height, 1))
    new_width = max(int(round(width * scale)), 1)
    new_height = max(int(round(height * scale)), 1)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((TARGET_HEIGHT, TARGET_WIDTH, 3), 255, dtype=np.uint8)
    x_offset = (TARGET_WIDTH - new_width) // 2
    y_offset = (TARGET_HEIGHT - new_height) // 2
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
    return canvas


def estimate_small_skew(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=100,
        minLineLength=180,
        maxLineGap=20,
    )
    angles: list[float] = []
    if lines is not None:
        for line in lines[:, 0]:
            x1, y1, x2, y2 = line
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if abs(angle) <= 6:
                angles.append(angle)
    return float(np.median(angles)) if angles else 0.0


def rotate_image(image_bgr: np.ndarray, angle: float) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        image_bgr,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def normalize_scan(image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    normalized = letterbox_document(image_bgr)
    angle = estimate_small_skew(normalized)
    if abs(angle) > 0.15:
        normalized = rotate_image(normalized, -angle)
    return normalized, angle


def build_fixed_structure_mask() -> np.ndarray:
    mask = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8)
    mask[0:220, :] = 255
    mask[390:1080, :] = 255
    mask[1080:1250, :] = 255
    mask[220:390, :] = 0
    return mask


FIXED_MASK = build_fixed_structure_mask()


def preprocess_structure(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 45, 140)
    horizontal = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)),
    )
    vertical = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)),
    )
    structure = cv2.max(edges, cv2.max(horizontal, vertical))
    structure = cv2.dilate(structure, np.ones((3, 3), dtype=np.uint8), iterations=1)
    structure = cv2.GaussianBlur(structure, (5, 5), 0)
    structure[FIXED_MASK == 0] = 0
    return structure.astype(np.float32) / 255.0


def align_and_score_template(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        try:
            ecc_score, warp = cv2.findTransformECC(
                reference, candidate, warp, cv2.MOTION_AFFINE,
                ECC_CRITERIA, FIXED_MASK, 5,
            )
        except TypeError:
            ecc_score, warp = cv2.findTransformECC(
                reference, candidate, warp, cv2.MOTION_AFFINE,
                ECC_CRITERIA, FIXED_MASK,
            )

        aligned = cv2.warpAffine(
            candidate,
            warp,
            (TARGET_WIDTH, TARGET_HEIGHT),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid = FIXED_MASK > 0
        ref_values = reference[valid]
        aligned_values = aligned[valid]

        if np.std(ref_values) < 1e-8 or np.std(aligned_values) < 1e-8:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(ref_values, aligned_values)[0, 1])

        if not np.isfinite(correlation):
            correlation = 0.0

        combined = float(
            0.55 * max(float(ecc_score), 0.0)
            + 0.45 * max(correlation, 0.0)
        )
        return {
            'success': True,
            'ecc': float(ecc_score),
            'correlation': correlation,
            'combined': combined,
        }
    except cv2.error as error:
        return {
            'success': False,
            'ecc': 0.0,
            'correlation': 0.0,
            'combined': 0.0,
            'error': str(error),
        }


def load_reference_structure(reference_path: Path) -> np.ndarray:
    image = cv2.imread(str(reference_path))
    if image is None:
        raise FileNotFoundError(f'Référence illisible : {reference_path}')
    normalized, _ = normalize_scan(image)
    return preprocess_structure(normalized)


def classify_releve_template(image_path: Path, templates_dir: Path) -> dict[str, Any]:
    references = {
        'RELEVE_ATTIJARI_V1': templates_dir / 'releve_bancaire' / 'attijari_v1' / 'reference.jpg',
        'RELEVE_CIH_V1': templates_dir / 'releve_bancaire' / 'cih_v1' / 'reference.jpg',
    }

    missing = [str(path) for path in references.values() if not path.exists()]
    if missing:
        return {
            'template_id': 'TEMPLATE_UNKNOWN',
            'status': 'TO_REVIEW',
            'confidence': 0.0,
            'method': 'visual_releve',
            'error': 'Références manquantes : ' + ' | '.join(missing),
        }

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f'Image illisible : {image_path}')

    normalized, skew_angle = normalize_scan(image)
    candidate = preprocess_structure(normalized)
    scores = {
        template_id: align_and_score_template(
            candidate,
            load_reference_structure(reference_path),
        )
        for template_id, reference_path in references.items()
    }

    ranking = sorted(scores.items(), key=lambda item: item[1]['combined'], reverse=True)
    best_template, best_details = ranking[0]
    best_score = float(best_details['combined'])
    second_score = float(ranking[1][1]['combined'])
    margin = best_score - second_score

    if best_score < MIN_TEMPLATE_SCORE or margin < MIN_TEMPLATE_MARGIN:
        return {
            'template_id': 'TEMPLATE_UNKNOWN',
            'status': 'TO_REVIEW',
            'confidence': round(best_score, 4),
            'method': 'visual_releve',
            'margin': round(margin, 4),
            'skew_angle': round(float(skew_angle), 4),
            'scores': scores,
        }

    return {
        'template_id': best_template,
        'status': 'CLASSIFIED',
        'confidence': round(best_score, 4),
        'method': 'visual_releve',
        'margin': round(margin, 4),
        'skew_angle': round(float(skew_angle), 4),
        'scores': scores,
    }


def classify_template(image_path: Path, doc_type: str, full_text: str, templates_dir: Path) -> dict[str, Any]:
    if doc_type == 'RELEVE_BANCAIRE':
        visual_result = classify_releve_template(image_path, templates_dir)
        if visual_result['status'] == 'CLASSIFIED':
            return visual_result

        bank_result = detect_bank_from_ocr(full_text)
        fallback = {
            'CIH': 'RELEVE_CIH_V1',
            'ATTIJARI': 'RELEVE_ATTIJARI_V1',
        }.get(bank_result['bank'])

        if fallback:
            return {
                'template_id': fallback,
                'status': 'CLASSIFIED',
                'confidence': bank_result['confidence'],
                'method': 'ocr_bank_fallback',
                'visual_result': visual_result,
                'bank_scores': bank_result['scores'],
            }
        return visual_result

    if doc_type == 'RIB':
        bank_result = detect_bank_from_ocr(full_text)
        template_id = {
            'CIH': 'RIB_CIH_V1',
            'ATTIJARI': 'RIB_ATTIJARI_V1',
        }.get(bank_result['bank'], 'TEMPLATE_UNKNOWN')
        return {
            'template_id': template_id,
            'status': 'CLASSIFIED' if template_id != 'TEMPLATE_UNKNOWN' else 'TO_REVIEW',
            'confidence': bank_result['confidence'],
            'method': 'ocr_bank',
            'bank_scores': bank_result['scores'],
        }

    if doc_type == 'QUITTANCE':
        normalized = normalize_text(full_text)
        for keyword, template_id in [
            ('lydec', 'QUITTANCE_LYDEC_V1'),
            ('amendis', 'QUITTANCE_AMENDIS_V1'),
            ('maroc telecom', 'QUITTANCE_MAROC_TELECOM_V1'),
        ]:
            if normalize_text(keyword) in normalized:
                return {
                    'template_id': template_id,
                    'status': 'CLASSIFIED',
                    'confidence': 1.0,
                    'method': 'ocr_keyword',
                }
        return {
            'template_id': 'TEMPLATE_UNKNOWN',
            'status': 'TO_REVIEW',
            'confidence': 0.0,
            'method': 'ocr_keyword',
        }

    template_id = {
        'CIN': 'CIN_V1',
        'BULLETIN_SALAIRE': 'BULLETIN_SALAIRE_V1',
        'ATTESTATION_TRAVAIL': 'ATTESTATION_TRAVAIL_V1',
    }.get(doc_type, 'TEMPLATE_UNKNOWN')

    return {
        'template_id': template_id,
        'status': 'CLASSIFIED' if template_id != 'TEMPLATE_UNKNOWN' else 'TO_REVIEW',
        'confidence': 1.0 if template_id != 'TEMPLATE_UNKNOWN' else 0.0,
        'method': 'single_known_template',
    }


def find_original_image(image_dir: Path, ocr_json: dict[str, Any], ocr_path: Path) -> Path | None:
    original_path = ocr_json.get('original_path')
    if original_path and Path(original_path).exists():
        return Path(original_path)

    filename = ocr_json.get('file')
    if filename and (image_dir / filename).exists():
        return image_dir / filename

    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f'{ocr_path.stem}{extension}'
        if candidate.exists():
            return candidate
    return None


def classify_document(image_path: Path, ocr_path: Path, templates_dir: Path) -> dict[str, Any]:
    ocr_json = load_json(ocr_path)

    if 'error' in ocr_json:
        return {
            'file': image_path.name,
            'original_path': str(image_path.resolve()),
            'ocr_path': str(ocr_path.resolve()),
            'doc_type': 'ERROR',
            'template_id': 'TEMPLATE_UNKNOWN',
            'status': 'ERROR',
            'error': ocr_json['error'],
        }

    full_text = ocr_json.get('full_text', '')
    doc_result = detect_doc_type(full_text)

    if doc_result['doc_type'] == 'UNKNOWN':
        return {
            'file': image_path.name,
            'original_path': str(image_path.resolve()),
            'ocr_path': str(ocr_path.resolve()),
            'doc_type': 'UNKNOWN',
            'template_id': 'TEMPLATE_UNKNOWN',
            'status': 'TO_REVIEW',
            'doc_type_result': doc_result,
        }

    template_result = classify_template(
        image_path,
        doc_result['doc_type'],
        full_text,
        templates_dir,
    )
    status = (
        'CLASSIFIED'
        if doc_result['status'] == 'CLASSIFIED' and template_result['status'] == 'CLASSIFIED'
        else 'TO_REVIEW'
    )

    return {
        'file': image_path.name,
        'original_path': str(image_path.resolve()),
        'ocr_path': str(ocr_path.resolve()),
        'doc_type': doc_result['doc_type'],
        'template_id': template_result['template_id'],
        'status': status,
        'doc_type_confidence': doc_result['confidence'],
        'template_confidence': template_result['confidence'],
        'doc_type_result': doc_result,
        'template_result': template_result,
    }


def route_document(classification: dict[str, Any], classified_dir: Path) -> str | None:
    original_path = Path(classification['original_path'])
    if not original_path.exists():
        return None

    folder_name = (
        classification['template_id']
        if classification['status'] == 'CLASSIFIED'
        else 'A_VERIFIER'
    )
    destination_dir = classified_dir / folder_name
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / original_path.name
    shutil.copy2(original_path, destination)
    return str(destination.resolve())


def classify_folder(
    image_dir: str | Path,
    ocr_dir: str | Path,
    output_manifest: str | Path,
    classified_dir: str | Path,
    templates_dir: str | Path,
) -> dict[str, Any]:
    images = Path(image_dir)
    ocr_cache = Path(ocr_dir)
    classified = Path(classified_dir)
    templates = Path(templates_dir)

    if not images.exists():
        raise FileNotFoundError(f'Dossier d’images introuvable : {images}')
    if not ocr_cache.exists():
        raise FileNotFoundError(f'Cache OCR introuvable : {ocr_cache}')

    classified.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for ocr_path in sorted(ocr_cache.glob('*.json')):
        ocr_json = load_json(ocr_path)
        image_path = find_original_image(images, ocr_json, ocr_path)

        if image_path is None:
            result = {
                'file': ocr_json.get('file', ocr_path.stem),
                'ocr_path': str(ocr_path.resolve()),
                'doc_type': 'ERROR',
                'template_id': 'TEMPLATE_UNKNOWN',
                'status': 'ERROR',
                'error': 'Image originale introuvable.',
            }
        else:
            print(f'[CLASSIFICATION] {image_path.name}')
            result = classify_document(image_path, ocr_path, templates)
            result['classified_path'] = route_document(result, classified)
        results.append(result)

    manifest = {
        'dossier_id': images.name,
        'image_dir': str(images.resolve()),
        'ocr_dir': str(ocr_cache.resolve()),
        'classified_dir': str(classified.resolve()),
        'documents': results,
        'summary': {
            'total': len(results),
            'classified': sum(item['status'] == 'CLASSIFIED' for item in results),
            'to_review': sum(item['status'] == 'TO_REVIEW' for item in results),
            'errors': sum(item['status'] == 'ERROR' for item in results),
        },
    }
    save_json(manifest, output_manifest)

    print('\n===== RÉSUMÉ CLASSIFICATION =====')
    for key, value in manifest['summary'].items():
        print(f'{key}: {value}')
    print(f'Manifeste : {Path(output_manifest)}')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification type + template.')
    parser.add_argument('--images', required=True)
    parser.add_argument('--ocr', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--classified-dir', required=True)
    parser.add_argument('--templates-dir', default='templates')
    args = parser.parse_args()

    classify_folder(
        image_dir=args.images,
        ocr_dir=args.ocr,
        output_manifest=args.output,
        classified_dir=args.classified_dir,
        templates_dir=args.templates_dir,
    )
