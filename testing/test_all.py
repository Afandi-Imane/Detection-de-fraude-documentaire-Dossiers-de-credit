from __future__ import annotations

import copy

from ocr_extraction.template_classifier import detect_doc_type
from ocr_extraction.extractors.cin_extractor import extract_cin
from ocr_extraction.extractors.attestation_extractor import extract_attestation_travail
from ocr_extraction.normalizer import normalize_document
from ocr_extraction.validator import validate_document


def block(text: str, x: float, y: float, score: float = 0.97) -> dict:
    return {"text": text, "x": x, "y": y, "x0": x - 50, "y0": y - 10, "x1": x + 50, "y1": y + 10, "score": score}


def field(document: dict, name: str) -> dict:
    return next(item for item in document["fields"] if item["field_name"] == name)


# 1. Classification textuelle.
assert detect_doc_type("CARTE D IDENTITE NUMERO D IDENTIFIANT DATE DE NAISSANCE")["doc_type"] == "CIN"
assert detect_doc_type("ATTESTATION DE TRAVAIL Nous soussignés Date d'embauche")["doc_type"] == "ATTESTATION_TRAVAIL"

# 2. CIN complète.
cin_ocr = {"file": "cin.jpg", "blocks": [
    block("Numéro d'identifiant", 120, 100, .99), block("J049505", 300, 100, .98),
    block("Nom", 70, 150, .99), block("EL ALAMI", 250, 150, .97),
    block("Prénom", 80, 200, .99), block("NADIA", 230, 200, .97),
    block("Date de naissance", 110, 250, .98), block("12/03/1998", 280, 250, .96),
    block("Lieu de naissance", 110, 300, .98), block("CASABLANCA", 280, 300, .96),
    block("Adresse", 80, 350, .99), block("10 RUE EXEMPLE CASABLANCA", 300, 350, .95),
    block("Nationalité", 90, 400, .98), block("MAROCAINE", 250, 400, .96),
    block("Sexe", 70, 450, .99), block("F", 200, 450, .98),
]}
cin_class = {"file": "cin.jpg", "doc_type": "CIN", "template_id": "CIN_V1", "status": "CLASSIFIED", "ocr_path": "cin.json"}
cin = validate_document(normalize_document(extract_cin(cin_ocr, cin_class, "D_TEST")))
assert cin["validation_status"] == "VALID", cin["validation"]
assert field(cin, "cin_numero")["normalized_value"] == "J049505"

# 3. CIN avec adresse manquante.
cin_missing_ocr = copy.deepcopy(cin_ocr)
cin_missing_ocr["blocks"] = [b for b in cin_missing_ocr["blocks"] if b["text"] not in {"Adresse", "10 RUE EXEMPLE CASABLANCA"}]
cin_missing = validate_document(normalize_document(extract_cin(cin_missing_ocr, cin_class, "D_TEST")))
assert cin_missing["validation_status"] == "INCOMPLETE", cin_missing["validation"]
assert "adresse" in cin_missing["validation"]["missing_required_fields"]

# 4. Attestation complète.
att_ocr = {"file": "attestation.jpg", "blocks": [
    block("COOPER PHARMA", 160, 40, .99),
    block("ATTESTATION DE TRAVAIL", 250, 150, .99),
    block("Nous soussignés, COOPER PHARMA, représentée par M. Karim ZNIBER", 300, 180, .97),
    block("Nom et Prénom", 120, 230, .99), block("SEFRIOUI Sara", 300, 260, .98),
    block("CIN", 120, 300, .99), block(": K470089", 300, 300, .98),
    block("Poste", 120, 340, .99), block(": Médecin", 300, 340, .98),
    block("Date d'embauche", 120, 380, .99), block(": 07/12/2009", 300, 380, .98),
    block("Signature", 400, 500, .98), block("SOCIÉTÉ", 400, 540, .96),
    block("R.C.: 38214", 400, 600, .96), block("ICE:001348760000045", 400, 630, .96),
]}
att_class = {"file": "attestation.jpg", "doc_type": "ATTESTATION_TRAVAIL", "template_id": "ATTESTATION_TRAVAIL_V1", "status": "CLASSIFIED", "ocr_path": "attestation.json"}
att = validate_document(normalize_document(extract_attestation_travail(att_ocr, att_class, "D_TEST")))
assert att["validation_status"] == "VALID", att["validation"]
assert field(att, "nom")["normalized_value"] == "SEFRIOUI SARA"
assert field(att, "cin")["normalized_value"] == "K470089"
assert field(att, "employeur")["normalized_value"] == "COOPER PHARMA"
assert field(att, "poste")["normalized_value"] == "MEDECIN"
assert field(att, "date_embauche")["normalized_value"] == "2009-12-07"
assert field(att, "cachet_present")["normalized_value"] is True

print("Tous les tests ont réussi.")
print("- Classification CIN : OK")
print("- Classification attestation : OK")
print("- Extraction/normalisation/validation CIN : VALID")
print("- Détection champ CIN manquant : INCOMPLETE")
print("- Extraction/normalisation/validation attestation : VALID")
