import json
from typing import Dict, List, Generator


# wczytuje plik JSONL linia po linii i zwraca kolejne rekordy jako słowniki
# używamy generatora zamiast listy, żeby nie ładować wszystkiego do pamięci naraz
# errors="replace" sprawi że zamiast wysypać się na złym kodowaniu, wstawi znak zastępczy
def _iter_jsonl(file_content: bytes) -> Generator[dict, None, None]:
    text = file_content.decode("utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        # puste linie pomijamy
        if line:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # jeśli linia nie jest poprawnym JSON-em, po prostu ją ignorujemy
                # nie chcemy żeby jeden uszkodzony rekord zatrzymał całe wczytywanie
                continue


# wczytuje pacjentów i zwraca słownik {id_pacjenta: cały_rekord}
# dzięki słownikowi możemy potem szybko znaleźć pacjenta po ID (zamiast przeglądać całą listę)
def load_patients(file_content: bytes) -> Dict[str, dict]:
    patients: Dict[str, dict] = {}
    for rec in _iter_jsonl(file_content):
        # bierzemy tylko rekordy które faktycznie są pacjentami i mają ID
        if rec.get("resourceType") == "Patient" and "id" in rec:
            # konwertujemy ID na string żeby uniknąć problemów gdy ID jest liczbą
            patients[str(rec["id"])] = rec
    return patients


# wczytuje obserwacje i od razu łączy każdą z odpowiadającym jej pacjentem
def load_observations_joined(file_content: bytes, patients: Dict[str, dict]) -> List[dict]:
    observations: List[dict] = []
    for idx, rec in enumerate(_iter_jsonl(file_content)):

        # każda obserwacja musi mieć ID żebyśmy mogli ją potem identyfikować
        # jeśli ID brakuje, generujemy syntetyczne na podstawie numeru wiersza
        if "id" in rec:
            rec["id"] = str(rec["id"])
        else:
            rec["id"] = f"_obs_{idx}"

        # wyciągamy ID pacjenta z pola subject.reference które wygląda tak: "Patient/100003784"
        # bierzemy część po ostatnim "/" czyli samo ID
        patient_id: str | None = None
        subject = rec.get("subject")
        if subject:
            ref = subject.get("reference", "")
            if ref and "/" in ref:
                patient_id = ref.rsplit("/", 1)[1]

        # dołączamy dane pacjenta bezpośrednio do rekordu obserwacji
        # dzięki temu nie musimy za każdym razem szukać pacjenta w słowniku
        # jeśli pacjent nie istnieje, _patient będzie None — walidator to wyłapie
        rec["_patient_id"] = patient_id
        rec["_patient"] = patients.get(patient_id) if patient_id else None
        observations.append(rec)

    return observations
