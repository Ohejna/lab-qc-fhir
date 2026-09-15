from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# gdy pacjent nie ma danego badania wstawiamy 0.5 czyli środek zakresu normy
# dzięki temu brakujące badanie nie wpływa na wynik modelu
_FILL_VALUE = 0.5

# domyślna lokalizacja wytrenowanego modelu
_MODEL_PATH = Path(__file__).parent.parent / "model" / "autoencoder.pt"


# sieć neuronowa typu autoenkoder
# encoder kompresuje wektor pacjenta do małej reprezentacji (bottleneck)
# decoder próbuje odtworzyć oryginalny wektor z tej skompresowanej postaci
# im większy błąd odtworzenia, tym bardziej pacjent odbiega od wzorców w danych treningowych
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        # rozmiary warstw dobierane dynamicznie względem liczby cech (rodzajów badań)
        h1 = max(input_dim, 32)
        h2 = max(input_dim // 2, 16)
        bottleneck = max(input_dim // 4, 8)
        # encoder: ściska dane wejściowe do coraz mniejszej reprezentacji
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2),        nn.ReLU(),
            nn.Linear(h2, bottleneck),
        )
        # decoder: próbuje odbudować oryginalny wektor z bottlenecka
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, h2), nn.ReLU(),
            nn.Linear(h2, h1),         nn.ReLU(),
            nn.Linear(h1, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# zamienia listę obserwacji FHIR na macierz liczbową gotową do podania sieci
# każdy wiersz macierzy to jeden pacjent, każda kolumna to jeden rodzaj badania
# wartości są znormalizowane: score = (wynik - dolna_norma) / (górna_norma - dolna_norma)
# dzięki temu wynik 0.0-1.0 oznacza wartość w normie, poniżej 0 lub powyżej 1 to odchylenie
def build_feature_matrix(
    observations: List[dict],
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    patient_obs: Dict[str, Dict[str, list]] = {}

    for obs in observations:
        # bierzemy tylko obserwacje poprawne strukturalnie — reszta nie ma zakresu normy
        if obs.get("_category") not in (None, "green"):
            continue
        value = (obs.get("valueQuantity") or {}).get("value")
        if not isinstance(value, (int, float)):
            continue
        rr = (obs.get("referenceRange") or [{}])[0]
        low  = (rr.get("low")  or {}).get("value")
        high = (rr.get("high") or {}).get("value")
        # bez zakresu normy nie możemy znormalizować wartości
        if low is None or high is None or high == low:
            continue

        pid  = obs.get("_patient_id") or obs.get("id", "__unknown__")
        code = (obs.get("code") or {}).get("text", "")
        if not code:
            continue

        # normalizacja: 0.0 = dolna granica normy, 1.0 = górna granica normy
        norm = (float(value) - float(low)) / (float(high) - float(low))
        patient_obs.setdefault(pid, {}).setdefault(code, []).append(norm)

    if not patient_obs:
        return np.empty((0, 0), dtype=np.float32), [], []

    # przy treningu ustalamy listę badań z danych — przy inferencji używamy tej samej listy
    if feature_cols is None:
        all_codes: set = set()
        for d in patient_obs.values():
            all_codes.update(d.keys())
        feature_cols = sorted(all_codes)

    patient_ids = list(patient_obs.keys())
    # macierz wypełniona 0.5 — nadpisujemy tylko te komórki gdzie pacjent ma wynik
    X = np.full((len(patient_ids), len(feature_cols)), _FILL_VALUE, dtype=np.float32)

    for i, pid in enumerate(patient_ids):
        for j, code in enumerate(feature_cols):
            vals = patient_obs[pid].get(code)
            # jeśli pacjent miał to samo badanie kilka razy, bierzemy średnią
            if vals:
                X[i, j] = float(np.mean(vals))

    return X, feature_cols, patient_ids


# trenuje autoenkoder na podanych obserwacjach i zapisuje model do pliku
# funkcja wywoływana jednorazowo przez train_model.py — aplikacja tylko wczytuje gotowy model
def train_and_save(
    observations: List[dict],
    save_path: Path = _MODEL_PATH,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    progress_callback=None,
) -> dict:
    if progress_callback:
        progress_callback(2, "Budowanie macierzy cech…")

    X, feature_cols, _ = build_feature_matrix(observations)

    if X.shape[0] < 10:
        raise ValueError(f"Za mało próbek: {X.shape[0]}. Potrzeba min. 10 pacjentów.")

    if progress_callback:
        progress_callback(10, f"{X.shape[0]} pacjentów × {X.shape[1]} cech. Trening…")

    tensor_X = torch.tensor(X)
    loader   = DataLoader(TensorDataset(tensor_X), batch_size=batch_size, shuffle=True)
    model    = Autoencoder(X.shape[1])
    # optymalizator Adam dobrze sprawdza się przy takich rozmiarach danych
    optim    = torch.optim.Adam(model.parameters(), lr=lr)
    # MSE mierzy średni kwadrat różnicy między oryginałem a rekonstrukcją
    loss_fn  = nn.MSELoss()

    last_loss = 0.0
    model.train()
    for epoch in range(epochs):
        last_loss = 0.0
        for (batch,) in loader:
            optim.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optim.step()
            last_loss += loss.item() * len(batch)
        last_loss /= len(tensor_X)
        if progress_callback:
            pct = 10 + int(80 * (epoch + 1) / epochs)
            progress_callback(pct, f"Epoka {epoch + 1}/{epochs} — MSE: {last_loss:.5f}")

    if progress_callback:
        progress_callback(93, "Wyznaczanie progu anomalii…")

    # po treningu liczymy błąd rekonstrukcji dla wszystkich pacjentów treningowych
    # próg to 95. percentyl tych błędów — górne 5% uznajemy za anomalie
    model.eval()
    with torch.no_grad():
        recon  = model(tensor_X)
        errors = ((tensor_X - recon) ** 2).mean(dim=1).numpy()

    threshold = float(np.percentile(errors, 95))

    # zapisujemy model razem z listą badań i progiem — potrzebne przy inferencji
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state":      model.state_dict(),
        "feature_cols":     feature_cols,
        "input_dim":        X.shape[1],
        "threshold":        threshold,
        "n_train_patients": X.shape[0],
    }, save_path)

    if progress_callback:
        progress_callback(100, f"Model zapisany → {save_path}")

    return {
        "n_patients":   X.shape[0],
        "n_features":   X.shape[1],
        "threshold":    threshold,
        "train_loss":   last_loss,
        "feature_cols": feature_cols,
    }


# wczytuje wytrenowany model z pliku przy starcie aplikacji
# zwraca None jeśli plik nie istnieje — wtedy aplikacja działa bez kolumny ML
def load_model(path: Path = _MODEL_PATH) -> Optional[dict]:
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = Autoencoder(ckpt["input_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    ckpt["model"] = model
    return ckpt


# ocenia każdą obserwację przy użyciu wczytanego modelu
# wynik jest przypisany do pacjenta — wszystkie obserwacje tego samego pacjenta
# dostają ten sam _ml_score i _ml_anomaly, bo model ocenia profil pacjenta, nie pojedyncze badanie
def score_observations(observations: List[dict], ckpt: dict) -> List[dict]:
    model: Autoencoder = ckpt["model"]
    feature_cols       = ckpt["feature_cols"]
    threshold: float   = ckpt["threshold"]

    X, _, patient_ids = build_feature_matrix(observations, feature_cols=feature_cols)
    if X.shape[0] == 0:
        return observations

    # przepuszczamy wszystkich pacjentów przez model i liczymy błąd rekonstrukcji
    tensor_X = torch.tensor(X)
    with torch.no_grad():
        recon  = model(tensor_X)
        errors = ((tensor_X - recon) ** 2).mean(dim=1).numpy()

    # mapujemy ID pacjenta na jego wynik i flagę anomalii
    pid_score = {
        pid: (float(errors[i]), bool(errors[i] > threshold))
        for i, pid in enumerate(patient_ids)
    }

    # dodajemy wyniki do każdej obserwacji — aplikacja wyświetla ikonkę przy nagłówku pacjenta
    result = []
    for obs in observations:
        pid = obs.get("_patient_id") or obs.get("id", "__unknown__")
        if pid in pid_score:
            score, is_anom = pid_score[pid]
            obs = {**obs, "_ml_score": round(score, 6), "_ml_anomaly": is_anom}
        result.append(obs)

    return result
