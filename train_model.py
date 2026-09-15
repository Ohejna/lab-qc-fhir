"""
Jednorazowy skrypt treningowy autoenkodera.

Użycie:
    python train_model.py patients.jsonl observations.jsonl

Model zostanie zapisany do model/autoencoder.pt.
Następnie aplikacja Streamlit wczyta go automatycznie.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from modules.loader import load_patients, load_observations_joined
from modules.validator import categorize_observation
from modules.standardizer import standardize_observation
from modules.enricher import enrich_observation
from modules.ml_anomaly import train_and_save

import json

_CONFIG_PATH = Path(__file__).parent / "config" / "reference_ranges.json"
with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    RANGES_CONFIG = json.load(f)


def _progress(pct: int, msg: str) -> None:
    bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
    print(f"\r[{bar}] {pct:3d}%  {msg:<60}", end="", flush=True)


def main():
    if len(sys.argv) != 3:
        print("Użycie: python train_model.py patients.jsonl observations.jsonl")
        sys.exit(1)

    patients_path = Path(sys.argv[1])
    obs_path      = Path(sys.argv[2])

    print(f"Wczytywanie pacjentów z {patients_path}…")
    patients = load_patients(patients_path.read_bytes())
    print(f"  Wczytano {len(patients):,} pacjentów.")

    print(f"Wczytywanie obserwacji z {obs_path}…")
    raw_obs = load_observations_joined(obs_path.read_bytes(), patients)
    print(f"  Wczytano {len(raw_obs):,} obserwacji.")

    print("Przetwarzanie (walidacja + wzbogacanie)…")
    processed = []
    total = len(raw_obs)
    for i, obs in enumerate(raw_obs):
        if i % 10000 == 0:
            print(f"  {i:,}/{total:,}", end="\r", flush=True)
        cat, reason = categorize_observation(obs)
        obs["_category"] = cat
        if cat == "green":
            obs, _, _ = standardize_observation(obs, RANGES_CONFIG)
            obs, _    = enrich_observation(obs, RANGES_CONFIG)
        processed.append(obs)

    print(f"\n  Przetworzono {total:,} obserwacji.")

    print("\nTrening autoenkodera:")
    stats = train_and_save(processed, progress_callback=_progress)
    print()
    print("\n✓ Gotowe!")
    print(f"  Pacjentów w treningu : {stats['n_patients']:,}")
    print(f"  Cech (badań)         : {stats['n_features']}")
    print(f"  Próg anomalii        : {stats['threshold']:.6f}")
    print(f"  Strata końcowa (MSE) : {stats['train_loss']:.6f}")
    print(f"  Model zapisany w     : model/autoencoder.pt")


if __name__ == "__main__":
    main()
