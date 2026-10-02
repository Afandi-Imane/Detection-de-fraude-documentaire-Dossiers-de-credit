from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

SIGNAL_ID = "VISUAL_GANOMALY"
SIGNAL_VERSION = "1.0"

# Association template de classification -> sous-dossier de modele dans
# signals/visual_engine/models/. Plusieurs templates de quittance partagent
# le meme modele (un seul entraine, peu importe le fournisseur).
TEMPLATE_TO_MODEL_DIR: dict[str, str] = {
    "CIN_V1": "cin",
    "ATTESTATION_TRAVAIL_V1": "attestation",
    "QUITTANCE_LYDEC_V1": "quittance",
    "QUITTANCE_AMENDIS_V1": "quittance",
    "QUITTANCE_MAROC_TELECOM_V1": "quittance",
    "BULLETIN_SALAIRE_V1": "bulletin",
    "RIB_CIH_V1": "rib_cih",
    "RIB_ATTIJARI_V1": "rib_attijari",
    "RELEVE_CIH_V1": "releve_cih",
    "RELEVE_ATTIJARI_V1": "releve_attijari",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# ============================================================
# ARCHITECTURE GANOMALY (identique aux notebooks d'entrainement)
# ============================================================

def _conv_block(in_c: int, out_c: int, norm: bool = True, act: str = "leaky") -> nn.Sequential:
    layers: list[nn.Module] = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm2d(out_c))
    if act == "leaky":
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def _deconv_block(
    in_c: int, out_c: int, kernel: int = 4, stride: int = 2, padding: int = 1,
    norm: bool = True, act: str = "relu",
) -> nn.Sequential:
    layers: list[nn.Module] = [nn.ConvTranspose2d(in_c, out_c, kernel, stride, padding, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm2d(out_c))
    if act == "relu":
        layers.append(nn.ReLU(inplace=True))
    elif act == "tanh":
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class _Encoder(nn.Module):
    """Encodeur simple (sans skip) -- utilise pour encoder2."""

    def __init__(self, in_channels: int = 3, img_size: int = 256, latent_dim: int = 100, base_channels: int = 64) -> None:
        super().__init__()
        n_layers = int(torch.log2(torch.tensor(img_size / 4)).item())
        layers = [_conv_block(in_channels, base_channels, norm=False)]
        ch = base_channels
        for _ in range(n_layers - 1):
            layers.append(_conv_block(ch, ch * 2))
            ch *= 2
        self.features = nn.Sequential(*layers)
        self.to_latent = nn.Conv2d(ch, latent_dim, 4, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.features(x))


class _UNetEncoder(nn.Module):
    """Encodeur qui garde chaque carte de features intermediaire (pour les skips)."""

    def __init__(self, in_channels: int = 3, img_size: int = 256, latent_dim: int = 100, base_channels: int = 64) -> None:
        super().__init__()
        n_layers = int(torch.log2(torch.tensor(img_size / 4)).item())
        self.blocks = nn.ModuleList()
        ch = base_channels
        self.blocks.append(_conv_block(in_channels, ch, norm=False))
        for _ in range(n_layers - 1):
            self.blocks.append(_conv_block(ch, ch * 2))
            ch *= 2
        self.to_latent = nn.Conv2d(ch, latent_dim, 4, 1, 0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        feats: list[torch.Tensor] = []
        h = x
        for block in self.blocks:
            h = block(h)
            feats.append(h)
        return self.to_latent(h), feats


class _UNetDecoder(nn.Module):
    """Decodeur qui concatene, a chaque etape, la feature map correspondante de l'encodeur."""

    def __init__(self, out_channels: int = 3, img_size: int = 256, latent_dim: int = 100, base_channels: int = 64) -> None:
        super().__init__()
        n_layers = int(torch.log2(torch.tensor(img_size / 4)).item())
        chs = [base_channels * (2 ** i) for i in range(n_layers)]
        top_ch = chs[-1]
        reversed_chs = list(reversed(chs))
        self.from_latent = _deconv_block(latent_dim, top_ch, kernel=4, stride=1, padding=0)
        self.blocks = nn.ModuleList()
        in_ch = top_ch * 2
        for i in range(n_layers):
            is_last = i == n_layers - 1
            if is_last:
                self.blocks.append(_deconv_block(in_ch, out_channels, norm=False, act="tanh"))
            else:
                out_ch = reversed_chs[i + 1]
                self.blocks.append(_deconv_block(in_ch, out_ch))
                in_ch = out_ch * 2

    def forward(self, z: torch.Tensor, feats: list[torch.Tensor]) -> torch.Tensor:
        h = self.from_latent(z)
        feats_rev = list(reversed(feats))
        for i, block in enumerate(self.blocks):
            h = torch.cat([h, feats_rev[i]], dim=1)
            h = block(h)
        return h


class Generator(nn.Module):
    def __init__(self, in_channels: int = 3, img_size: int = 256, latent_dim: int = 100) -> None:
        super().__init__()
        self.encoder1 = _UNetEncoder(in_channels, img_size, latent_dim)
        self.decoder = _UNetDecoder(in_channels, img_size, latent_dim)
        self.encoder2 = _Encoder(in_channels, img_size, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, feats = self.encoder1(x)
        x_hat = self.decoder(z, feats)
        z_hat = self.encoder2(x_hat)
        return x_hat, z, z_hat


# ============================================================
# CHARGEMENT DES MODELES (un dossier par template, mis en cache)
# ============================================================

def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"JSON invalide : {path}")
    return data


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    return output


def _find_checkpoint(model_dir: Path) -> Path:
    # Nom exact attendu, sinon on prend le premier "checkpoint_best*.pth" trouve
    # (ex: "checkpoint_best-001.pth").
    exact = model_dir / "checkpoint_best.pth"
    if exact.exists():
        return exact
    candidates = sorted(model_dir.glob("checkpoint_best*.pth"))
    if not candidates:
        raise FileNotFoundError(f"Aucun checkpoint_best*.pth dans {model_dir}")
    return candidates[0]


@lru_cache(maxsize=None)
def _load_model_bundle(model_dir_str: str) -> dict[str, Any]:
    model_dir = Path(model_dir_str)

    config = load_json(model_dir / "config.json")

    calibration_path = model_dir / "calibration.json"
    calibration = load_json(calibration_path) if calibration_path.exists() else None

    checkpoint_path = _find_checkpoint(model_dir)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    generator = Generator(
        in_channels=3,
        img_size=int(config.get("img_size", IMG_SIZE)),
        latent_dim=int(config.get("latent_dim", 100)),
    ).to(DEVICE)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    profile_mean = np.load(model_dir / "profile_mean.npy")
    profile_std = np.load(model_dir / "profile_std.npy")

    return {
        "generator": generator,
        "profile_mean": profile_mean,
        "profile_std": profile_std,
        "config": config,
        "calibration": calibration,
        "checkpoint_path": str(checkpoint_path),
    }


def get_model_bundle(models_root: str | Path, template_key: str) -> dict[str, Any]:
    model_dir = Path(models_root) / template_key
    if not model_dir.exists():
        raise FileNotFoundError(f"Dossier de modele introuvable : {model_dir}")
    return _load_model_bundle(str(model_dir))


def clear_model_cache() -> None:
    """Libere de la RAM tous les modeles GANomaly gardes en cache.

    A appeler une fois que le signal visuel a fini de tourner pour ce
    dossier -- sans ca, lru_cache garde tous les modeles charges en
    memoire pendant tout le reste du pipeline (extraction, LLM, Neo4j...),
    ce qui peut saturer la RAM et perturber d'autres services locaux
    (ex: Neo4j)."""
    _load_model_bundle.cache_clear()


# ============================================================
# SCORE D'ANOMALIE (identique a compute_anomaly_score des notebooks)
# ============================================================

def compute_anomaly_score(bundle: dict[str, Any], image_path: str | Path) -> float:
    generator = bundle["generator"]
    config = bundle["config"]
    profile_mean = bundle["profile_mean"]
    profile_std = bundle["profile_std"]

    patch_size = int(config.get("patch_size", 4))
    top_k_patches = int(config.get("top_k_patches", 3))
    w_con = float(config.get("w_con", 0.9))
    img_size = int(config.get("img_size", IMG_SIZE))

    img = Image.open(image_path)
    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        fake, z, z_hat = generator(tensor)
        diff = torch.abs(fake - tensor).mean(dim=1, keepdim=True)
        patch_map = F.avg_pool2d(diff, kernel_size=patch_size, stride=patch_size)
        patch_map_np = patch_map.squeeze().cpu().numpy()

        z_map = (patch_map_np - profile_mean) / profile_std
        flat = z_map.flatten()
        k = min(top_k_patches, flat.size)
        local_score = float(np.sort(flat)[::-1][:k].mean())

        lat_err = float(torch.mean((z - z_hat) ** 2).item())

    return w_con * local_score + (1 - w_con) * lat_err


def raw_to_visual_score(raw_score: float, bundle: dict[str, Any]) -> float:
    """Convertit un score brut (echelle propre a chaque template) en score
    universel [0, 1], comparable entre tous les templates.

    Utilise ancre_normal/ancre_fraude (calibration.json) si disponibles ;
    sinon repli sur le seuil Youden seul (config.json)."""
    calibration = bundle.get("calibration")

    if calibration and "ancre_normal" in calibration and "ancre_fraude" in calibration:
        ancre_normal = float(calibration["ancre_normal"])
        ancre_fraude = float(calibration["ancre_fraude"])
        ecart = ancre_fraude - ancre_normal
        if ecart <= 0:
            # calibration degeneree -- repli sur le seuil
            threshold = float(bundle["config"]["threshold"])
            return max(0.0, min(1.0, raw_score / (raw_score + threshold))) if raw_score >= 0 else 0.0
        score = (raw_score - ancre_normal) / ecart
        return max(0.0, min(1.0, score))

    threshold = float(bundle["config"]["threshold"])
    if raw_score < 0:
        return 0.0
    return raw_score / (raw_score + threshold)


# ============================================================
# CALCUL DU SIGNAL POUR UN DOSSIER COMPLET
# ============================================================

def compute_visual_signal(
    classification_manifest: dict[str, Any],
    models_root: str | Path,
) -> dict[str, Any]:
    dossier_id = str(classification_manifest.get("dossier_id", "UNKNOWN_DOSSIER"))
    documents = classification_manifest.get("documents", [])
    if not isinstance(documents, list):
        raise ValueError("Le manifeste de classification doit contenir une liste 'documents'.")

    doc_results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for document in documents:
        if not isinstance(document, dict):
            continue

        template_id = document.get("template_id")
        status = document.get("status")
        image_path = document.get("classified_path") or document.get("original_path")
        doc_label = document.get("file") or image_path or "DOCUMENT_INCONNU"

        if status != "CLASSIFIED":
            skipped.append({"file": doc_label, "reason": f"status={status}"})
            continue

        model_key = TEMPLATE_TO_MODEL_DIR.get(str(template_id))
        if model_key is None:
            skipped.append({"file": doc_label, "reason": f"template_id non mappe : {template_id}"})
            continue

        if not image_path or not Path(image_path).exists():
            skipped.append({"file": doc_label, "reason": "image introuvable"})
            continue

        try:
            bundle = get_model_bundle(models_root, model_key)
        except FileNotFoundError as error:
            skipped.append({"file": doc_label, "reason": f"modele indisponible ({model_key}) : {error}"})
            continue

        try:
            raw_score = compute_anomaly_score(bundle, image_path)
            visual_score = raw_to_visual_score(raw_score, bundle)
        except Exception as error:
            skipped.append({"file": doc_label, "reason": f"erreur de calcul : {error}"})
            continue

        doc_results.append({
            "file": doc_label,
            "template_id": template_id,
            "model": model_key,
            "raw_score": round(raw_score, 5),
            "score": round(visual_score, 4),
            "status": "SUSPECT" if visual_score > 0.5 else "NORMAL",
        })

    available_scores = [item["score"] for item in doc_results]
    score = max(available_scores) if available_scores else None
    available = bool(available_scores)

    if score is None:
        status = "UNAVAILABLE"
    elif score > 0.5:
        status = "TRIGGERED"
    else:
        status = "NORMAL"

    most_suspect = max(doc_results, key=lambda item: item["score"], default=None)

    return {
        "signal_id": SIGNAL_ID,
        "signal_version": SIGNAL_VERSION,
        "dossier_id": dossier_id,
        "available": available,
        "score": score,
        "status": status,
        "aggregation": "MAX_DOCUMENT_SCORE",
        "most_suspect_document": most_suspect["file"] if most_suspect else None,
        "documents": doc_results,
        "skipped_documents": skipped,
        "summary": {
            "documents_scored": len(doc_results),
            "documents_skipped": len(skipped),
            "suspect_count": sum(item["status"] == "SUSPECT" for item in doc_results),
        },
    }


def run_visual_signal(
    classification_manifest_path: str | Path,
    models_root: str | Path = "signals/visual_engine/models",
    output_root: str | Path = "data/signals",
) -> dict[str, Any]:
    manifest = load_json(classification_manifest_path)
    result = compute_visual_signal(manifest, models_root)

    output_path = Path(output_root) / result["dossier_id"] / "visual_signal.json"
    result["output_path"] = str(output_path)
    save_json(result, output_path)
    return result


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calcule le signal visuel GANomaly depuis un manifeste de classification.",
    )
    parser.add_argument("--manifest", required=True, help="Chemin vers data/classification/Dxxx_manifest.json")
    parser.add_argument("--models-root", default="signals/visual_engine/models")
    parser.add_argument("--output-root", default="data/signals")
    args = parser.parse_args()

    result = run_visual_signal(args.manifest, args.models_root, args.output_root)

    print("\n===== SIGNAL VISUEL (GANomaly) =====")
    print(f"Dossier : {result['dossier_id']}")
    print(f"Disponible : {result['available']}")
    print(f"Score : {result['score']}")
    print(f"Statut : {result['status']}")
    print(f"Document le plus suspect : {result['most_suspect_document']}")
    print(f"Documents notes : {result['summary']['documents_scored']}")
    print(f"Documents ignores : {result['summary']['documents_skipped']}")
    print(f"Résultat : {result['output_path']}")


if __name__ == "__main__":
    main()
    