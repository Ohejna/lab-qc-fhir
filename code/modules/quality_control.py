# sprawdza czy wartość pomiaru nie jest przypadkiem skrajnie nieprawdopodobna
# np. hemoglobina 200 g/dl albo leukocyty 0.001 — takie wartości to prawie na pewno błąd
# jeśli znajdzie outliera, dodaje do rekordu flagę _outlier i opis co jest nie tak
def check_outliers(obs: dict) -> dict:

    # żeby ocenić czy wartość jest "za skrajna" potrzebujemy zakresu normy
    # bez niego nie mamy punktu odniesienia
    ref_range = obs.get("referenceRange")
    if not ref_range:
        return obs

    rr = ref_range[0]
    low  = (rr.get("low")  or {}).get("value")
    high = (rr.get("high") or {}).get("value")
    value = (obs.get("valueQuantity") or {}).get("value")

    # nie da się porównywać tekstów z liczbami — tylko wartości numeryczne sprawdzamy
    if not isinstance(value, (int, float)):
        return obs

    reason = ""

    # próg dolny: wartość poniżej 20% dolnej granicy normy
    # np. WBC norma od 4.0 — próg outliera to 4.0 * 0.20 = 0.8
    # low > 0 żeby nie dzielić przez zero i nie generować fałszywych alarmów gdy norma zaczyna od 0
    if low is not None and low > 0 and value < low * 0.20:
        pct = value / low * 100
        reason = (
            f"Wartość krytycznie niska: {value} — poniżej 20% dolnej granicy normy "
            f"({low}), wynosi {pct:.1f}% normy"
        )

    # próg górny: wartość powyżej 500% górnej granicy normy (czyli ponad 5-krotność)
    # np. WBC norma do 10.0 — próg outliera to 10.0 * 5.0 = 50.0
    elif high is not None and high > 0 and value > high * 5.0:
        mult = value / high
        reason = (
            f"Wartość krytycznie wysoka: {value} — powyżej 500% górnej granicy normy "
            f"({high}), {mult:.1f}x powyżej normy"
        )

    # jeśli któryś próg został przekroczony, oznaczamy rekord jako outlier
    # używamy {**obs, ...} żeby nie modyfikować oryginału tylko tworzyć kopię z nowymi polami
    if reason:
        obs = {**obs, "_outlier": True, "_outlier_reason": reason}

    return obs
