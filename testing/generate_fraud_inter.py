import copy
import json
from pathlib import Path
from typing import Any


# ============================================================
# OUTILS
# ============================================================

def get_value(field: Any):
    if isinstance(field, dict):
        return field.get("normalized_value", field.get("value"))
    return field


def set_value(field: Any, value: Any):
    if isinstance(field, dict):
        if "normalized_value" in field:
            field["normalized_value"] = value
        if "value" in field:
            field["value"] = value
    return value


# ============================================================
# FRAUDES INTER-DOSSIERS
# ============================================================

def fraud_reuse_cin(
    dossier: dict,
    reference_dossier: dict
) -> bool:
    """
    Réutilise le CIN d'un autre dossier.
    """

    target_docs = dossier.get("document_fields", {})
    reference_docs = reference_dossier.get("document_fields", {})

    reference_cin = None

    # Chercher le CIN du dossier de référence
    for fields in reference_docs.values():

        if not isinstance(fields, dict):
            continue

        for field_name in ("cin", "cin_numero"):

            if field_name in fields:
                reference_cin = get_value(fields[field_name])

                if reference_cin:
                    break

        if reference_cin:
            break

    if not reference_cin:
        return False

    # Modifier le CIN du dossier cible
    for fields in target_docs.values():

        if not isinstance(fields, dict):
            continue

        for field_name in ("cin", "cin_numero"):

            if field_name in fields:
                set_value(fields[field_name], reference_cin)
                return True

    return False


def fraud_reuse_nom(
    dossier: dict,
    reference_dossier: dict
) -> bool:
    """
    Réutilise le nom d'une personne provenant d'un autre dossier.
    """

    target_docs = dossier.get("document_fields", {})
    reference_docs = reference_dossier.get("document_fields", {})

    reference_name = None

    for fields in reference_docs.values():

        if not isinstance(fields, dict):
            continue

        for field_name in (
            "nom",
            "nom_employe",
            "prenom",
            "prenom_employe"
        ):

            if field_name in fields:

                value = get_value(fields[field_name])

                if value:
                    reference_name = value
                    break

        if reference_name:
            break

    if not reference_name:
        return False

    # Modifier seulement un document.
    # Cela crée une incohérence entre dossiers/documents.
    for fields in target_docs.values():

        if not isinstance(fields, dict):
            continue

        if "nom" in fields:
            set_value(fields["nom"], reference_name)
            return True

        if "nom_employe" in fields:
            set_value(fields["nom_employe"], reference_name)
            return True

    return False


def fraud_reuse_adresse(
    dossier: dict,
    reference_dossier: dict
) -> bool:
    """
    Réutilise l'adresse d'un autre dossier.
    """

    target_docs = dossier.get("document_fields", {})
    reference_docs = reference_dossier.get("document_fields", {})

    reference_address = None

    for fields in reference_docs.values():

        if not isinstance(fields, dict):
            continue

        if "adresse" in fields:

            reference_address = get_value(
                fields["adresse"]
            )

            if reference_address:
                break

    if not reference_address:
        return False

    for fields in target_docs.values():

        if not isinstance(fields, dict):
            continue

        if "adresse" in fields:

            set_value(
                fields["adresse"],
                reference_address
            )

            return True

    return False


# ============================================================
# APPLICATION
# ============================================================

def apply_inter_fraud(
    dossier: dict,
    reference_dossier: dict,
    fraud_type: str
) -> dict:

    # IMPORTANT :
    # On travaille sur une copie.
    fraud_dossier = copy.deepcopy(dossier)

    success = False

    if fraud_type == "REUSE_CIN":

        success = fraud_reuse_cin(
            fraud_dossier,
            reference_dossier
        )

    elif fraud_type == "REUSE_NOM":

        success = fraud_reuse_nom(
            fraud_dossier,
            reference_dossier
        )

    elif fraud_type == "REUSE_ADRESSE":

        success = fraud_reuse_adresse(
            fraud_dossier,
            reference_dossier
        )

    else:
        raise ValueError(
            f"Type de fraude inter inconnu : {fraud_type}"
        )

    if not success:
        raise RuntimeError(
            f"Impossible d'injecter la fraude {fraud_type}"
        )

    # Métadonnées de traçabilité
    fraud_dossier["fraud"] = {
        "is_fraud": True,
        "fraud_family": "INTER",
        "fraud_type": fraud_type,
        "reference_dossier": reference_dossier.get(
            "dossier_id"
        ),
    }

    return fraud_dossier


# ============================================================
# GENERATION
# ============================================================

def generate_fraud_inter(
    input_path: str | Path,
    reference_path: str | Path,
    output_root: str | Path,
    fraud_type: str,
    output_id: str,
):

    input_path = Path(input_path)
    reference_path = Path(reference_path)
    output_root = Path(output_root)

    # ----------------------------
    # Dossier cible
    # ----------------------------

    with input_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        dossier = json.load(file)

    # ----------------------------
    # Dossier de référence
    # ----------------------------

    with reference_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        reference_dossier = json.load(file)

    # ----------------------------
    # Créer la fraude
    # ----------------------------

    fraud_dossier = apply_inter_fraud(
        dossier=dossier,
        reference_dossier=reference_dossier,
        fraud_type=fraud_type,
    )

    # Nouveau ID
    fraud_dossier["dossier_id"] = output_id

    # ----------------------------
    # Sortie
    # ----------------------------

    output_dir = output_root / output_id

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = output_dir / "dossier.json"

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            fraud_dossier,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"[OK] Fraude INTER créée : {output_path}"
    )

    return output_path


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Génère une fraude inter-dossiers."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Dossier cible légitime"
    )

    parser.add_argument(
        "--reference",
        required=True,
        help="Dossier utilisé comme référence"
    )

    parser.add_argument(
        "--output-root",
        default="dossiers_fraude"
    )

    parser.add_argument(
        "--fraud-type",
        required=True,
        choices=[
            "REUSE_CIN",
            "REUSE_NOM",
            "REUSE_ADRESSE",
        ]
    )

    parser.add_argument(
        "--output-id",
        required=True
    )

    args = parser.parse_args()

    generate_fraud_inter(
        input_path=args.input,
        reference_path=args.reference,
        output_root=args.output_root,
        fraud_type=args.fraud_type,
        output_id=args.output_id,
    )