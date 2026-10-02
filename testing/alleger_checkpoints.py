"""
Allege les checkpoints GANomaly deja entraines : garde uniquement les poids
du generator (necessaires pour l'inference), jette le discriminator et les
optimiseurs (inutiles en production, mais presents car sauvegardes pendant
l'entrainement).

A executer UNE SEULE FOIS, directement sur le PC, dans le dossier racine du
projet (celui qui contient signals/visual_engine/models/).

Usage :
    python alleger_checkpoints.py
"""

from __future__ import annotations

from pathlib import Path

import torch

MODELS_ROOT = Path("signals/visual_engine/models")


def alleger_checkpoint(model_dir: Path) -> None:
    checkpoints = sorted(model_dir.glob("checkpoint_best*.pth"))
    if not checkpoints:
        print(f"[IGNORE] {model_dir.name} : aucun checkpoint_best*.pth trouve.")
        return

    source = checkpoints[0]
    original_size_mo = source.stat().st_size / 1e6
    print(f"[TRAITEMENT] {model_dir.name} -> {source.name} ({original_size_mo:.1f} Mo)")

    full_checkpoint = torch.load(source, map_location="cpu")

    if "generator" not in full_checkpoint:
        print(f"   [ERREUR] Cle 'generator' absente dans {source.name}, fichier ignore.")
        return

    light_checkpoint = {"generator": full_checkpoint["generator"]}

    destination = model_dir / "checkpoint_best.pth"

    # Si le fichier allege remplace l'original (meme nom), on ecrit d'abord
    # sous un nom temporaire pour eviter d'ecraser la source avant d'avoir
    # fini de lire ses poids.
    temp_destination = model_dir / "checkpoint_best_light_tmp.pth"
    torch.save(light_checkpoint, temp_destination)

    if source.resolve() != destination.resolve():
        source.unlink()

    temp_destination.replace(destination)

    new_size_mo = destination.stat().st_size / 1e6
    print(f"   -> allege : {new_size_mo:.1f} Mo (avant : {original_size_mo:.1f} Mo)")


def main() -> None:
    if not MODELS_ROOT.exists():
        print(f"[ERREUR] Dossier introuvable : {MODELS_ROOT.resolve()}")
        print("Verifie que tu executes bien ce script depuis la racine du projet.")
        return

    model_dirs = sorted(p for p in MODELS_ROOT.iterdir() if p.is_dir())
    if not model_dirs:
        print(f"[ERREUR] Aucun sous-dossier de modele trouve dans {MODELS_ROOT.resolve()}")
        return

    print(f"Dossiers de modeles trouves : {[d.name for d in model_dirs]}\n")

    for model_dir in model_dirs:
        alleger_checkpoint(model_dir)

    print("\nTermine. Relance le pipeline pour verifier que le chargement est bien plus rapide.")


if __name__ == "__main__":
    main()