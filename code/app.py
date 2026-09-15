import json
import sys
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from modules.loader import load_patients, load_observations_joined
from modules.validator import categorize_observation
from modules.standardizer import standardize_observation
from modules.enricher import enrich_observation
from modules.quality_control import check_outliers
from modules.exporter import export_jsonl, export_report_csv
from modules.ml_anomaly import load_model, score_observations

_ML_MODEL = load_model()

_CONFIG_PATH = Path(__file__).parent / "config" / "reference_ranges.json"
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    RANGES_CONFIG: dict = json.load(_f)

st.set_page_config(
    page_title="System kontroli jakości i standaryzacji danych laboratoryjnych w standardzie HL7 FHIR",
    page_icon="🔬",
    layout="wide",
)

_DEFAULTS = {
    "screen": "upload",
    "observations": [],
    "patients": {},
    "patient_obs_idx": {},   
    "patient_stats_cache": {},
    "deleted_ids": set(),
    "proc_stats": {},
    "patient_page": 0,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def _interp_code(obs: dict) -> str:
    interp = obs.get("interpretation")
    if not interp:
        return ""
    return (interp[0].get("coding") or [{}])[0].get("code", "")


def _patient_priority(pid: str) -> int:
    """0=red errors, 1=yellow/suspicious, 2=clean."""
    cache = st.session_state.patient_stats_cache.get(pid, {})
    if cache.get("has_red"):
        return 0
    if cache.get("has_suspicious"):
        return 1
    return 2


def _build_patient_stats_cache(observations: List[dict]) -> Dict[str, dict]:
    idx_map: Dict[str, List[int]] = {}
    for i, obs in enumerate(observations):
        pid = obs.get("_patient_id") or "__unknown__"
        idx_map.setdefault(pid, []).append(i)

    cache: Dict[str, dict] = {}
    for pid, indices in idx_map.items():
        obs_list = [observations[i] for i in indices]
        has_red = any(o.get("_category") in ("red", "yellow") for o in obs_list)
        has_suspicious = any(
            o.get("_outlier") or _interp_code(o) in ("L", "H")
            for o in obs_list
        )
        n_errors = sum(1 for o in obs_list if o.get("_category") in ("red", "yellow"))
        n_suspicious = sum(
            1 for o in obs_list
            if o.get("_outlier") or _interp_code(o) in ("L", "H")
        )
        cache[pid] = {
            "indices": indices,
            "n_obs": len(indices),
            "has_red": has_red,
            "has_suspicious": has_suspicious,
            "n_errors": n_errors,
            "n_suspicious": n_suspicious,
        }
    return cache


def process_data(patients_bytes: bytes, obs_bytes: bytes) -> None:
    progress = st.progress(0, text="Wczytywanie pacjentów…")

    patients = load_patients(patients_bytes)
    st.session_state.patients = patients
    progress.progress(5, text=f"Wczytano {len(patients):,} pacjentów. Wczytywanie obserwacji…")

    raw_obs = load_observations_joined(obs_bytes, patients)
    total = len(raw_obs)
    progress.progress(15, text=f"Wczytano {total:,} obserwacji. Walidacja i przetwarzanie…")

    stats = {
        "total": total,
        "red": 0,
        "yellow": 0,
        "green": 0,
        "in_norm": 0,
        "out_of_norm": 0,
        "no_range": 0,
        "outliers": 0,
        "unknown_test": 0,
        "converted": 0,
    }

    processed: List[dict] = []
    BATCH = 5000

    for i, obs in enumerate(raw_obs):
        if i % BATCH == 0 and i > 0:
            pct = 15 + int(75 * i / total)
            progress.progress(pct, text=f"Przetwarzanie {i:,} / {total:,}…")

        # Step 2 – categorize
        cat, reason = categorize_observation(obs)
        obs["_category"] = cat
        obs["_category_reason"] = reason

        if cat == "red":
            stats["red"] += 1
        elif cat == "yellow":
            stats["yellow"] += 1
        else:
            stats["green"] += 1

            obs, was_conv, conv_info = standardize_observation(obs, RANGES_CONFIG)
            if was_conv:
                stats["converted"] += 1
                obs["_converted"] = True
                obs["_conversion_info"] = conv_info

            obs, warning = enrich_observation(obs, RANGES_CONFIG)
            if warning:
                obs["_warning"] = warning
                stats["unknown_test"] += 1
                stats["no_range"] += 1

            obs = check_outliers(obs)
            if obs.get("_outlier"):
                stats["outliers"] += 1

            ic = _interp_code(obs)
            if ic == "N":
                stats["in_norm"] += 1
            elif ic in ("L", "H"):
                stats["out_of_norm"] += 1

        processed.append(obs)

    if _ML_MODEL is not None:
        progress.progress(92, text="Ocena anomalii przez model ML…")
        processed = score_observations(processed, _ML_MODEL)

    progress.progress(95, text="Budowanie indeksu pacjentów…")
    patient_stats_cache = _build_patient_stats_cache(processed)

    patient_obs_idx = {pid: v["indices"] for pid, v in patient_stats_cache.items()}

    st.session_state.observations = processed
    st.session_state.patient_obs_idx = patient_obs_idx
    st.session_state.patient_stats_cache = patient_stats_cache
    st.session_state.deleted_ids = set()
    st.session_state.proc_stats = stats
    st.session_state.patient_page = 0
    st.session_state.screen = "review"

    progress.progress(100, text="Gotowe!")
    st.rerun()


def _build_obs_df(obs_list: List[dict], deleted_ids: Set[str]) -> pd.DataFrame:
    rows = []
    for obs in obs_list:
        obs_id = obs.get("id", "")
        if obs_id in deleted_ids:
            continue

        code = (obs.get("code") or {}).get("text", "")
        vq = obs.get("valueQuantity") or {}
        value = vq.get("value", "")
        unit = vq.get("unit", "")

        rr = (obs.get("referenceRange") or [{}])[0]
        low_v = (rr.get("low") or {}).get("value", "")
        high_v = (rr.get("high") or {}).get("value", "")
        norma = f"{low_v} – {high_v}" if low_v != "" and high_v != "" else "—"

        ic = _interp_code(obs)
        interp_map = {"N": "Norma", "L": "Poniżej normy", "H": "Powyżej normy"}
        interp_display = interp_map.get(ic, "—")

        cat = obs.get("_category", "green")
        reason = obs.get("_category_reason", "")
        warning = obs.get("_warning", "")
        outlier = obs.get("_outlier", False)
        outlier_reason = obs.get("_outlier_reason", "")

        if cat == "red":
            status = f"🔴 {reason}"
        elif cat == "yellow":
            status = f"🟡 {reason}"
        elif outlier:
            status = f"🟡 {outlier_reason}"
        elif ic in ("L", "H"):
            status = f"🟡 {interp_display}"
        elif warning:
            status = f"⚠️ {warning}"
        else:
            status = "✅ W normie"

        rows.append({
            "ID":            obs_id,
            "Zaznacz":       False,
            "Badanie":       code,
            "Wartość":       value,
            "Jednostka":     unit,
            "Norma":         norma,
            "Interpretacja": interp_display,
            "Status":        status,
            "_cat":          cat,
            "_outlier":      outlier,
            "_ic":           ic,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["ID", "Zaznacz", "Badanie", "Wartość", "Jednostka",
                 "Norma", "Interpretacja", "Status", "_cat", "_outlier", "_ic"]
    )


def screen_upload() -> None:
    st.title("🔬 System kontroli jakości i standaryzacji danych laboratoryjnych w standardzie HL7 FHIR")
    st.markdown("### Import danych")

    col1, col2 = st.columns(2)
    with col1:
        patients_file = st.file_uploader(
            "Wgraj plik pacjentów (patients.jsonl)",
            type=["jsonl", "json"],
            key="uf_patients",
        )
    with col2:
        obs_file = st.file_uploader(
            "Wgraj plik obserwacji (observations.jsonl)",
            type=["jsonl", "json"],
            key="uf_obs",
        )

    ready = patients_file is not None and obs_file is not None
    if st.button("Zatwierdź i przetwórz", disabled=not ready, type="primary"):
        process_data(patients_file.read(), obs_file.read())


def screen_review() -> None:
    observations: List[dict] = st.session_state.observations
    deleted_ids: Set[str] = st.session_state.deleted_ids
    patients: Dict[str, dict] = st.session_state.patients
    cache: Dict[str, dict] = st.session_state.patient_stats_cache
    stats: dict = st.session_state.proc_stats

    st.title("🔬 System kontroli jakości i standaryzacji danych laboratoryjnych w standardzie HL7 FHIR")
    st.markdown("### Przegląd i zarządzanie danymi")

    # ── Top metrics
    n_total = stats.get("total", 0)
    n_red_auto = stats.get("red", 0)
    n_green = stats.get("green", 0)
    n_outliers = stats.get("outliers", 0)
    n_manually_deleted = len(deleted_ids)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Łącznie obserwacji", f"{n_total:,}")
    c2.metric("Poprawnych strukturalnie", f"{n_green:,}")
    c3.metric("Podejrzanych (outlier)", f"{n_outliers:,}")
    c4.metric("Usuniętych automatycznie 🔴", f"{n_red_auto:,}")

    st.markdown(f"**Usuniętych przez operatora:** {n_manually_deleted:,}")

    n_errors_total = sum(
        1 for obs in observations
        if obs.get("_category") in ("red", "yellow") and obs.get("id") not in deleted_ids
    )
    col_btn1, col_btn2, _ = st.columns([2, 2, 6])
    with col_btn1:
        if st.button(
            f"🔴 Usuń wszystkie błędy strukturalne ({n_errors_total} rekordów)",
            disabled=n_errors_total == 0,
        ):
            ids_to_del = {
                obs["id"]
                for obs in observations
                if obs.get("_category") in ("red", "yellow") and obs.get("id")
            }
            st.session_state.deleted_ids.update(ids_to_del)
            st.rerun()
    with col_btn2:
        if st.button("➡️ Przejdź do eksportu", type="primary"):
            st.session_state.screen = "export"
            st.rerun()

    st.divider()

    with st.expander("🔍 Filtry i wyszukiwanie", expanded=False):
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            show_problems_only = st.checkbox("Pokaż tylko pacjentów z problemami", value=False)
        with fcol2:
            search_pid = st.text_input("Szukaj po ID pacjenta", "")

    all_pids = list(cache.keys())
    all_pids.sort(key=_patient_priority)

    if show_problems_only:
        all_pids = [p for p in all_pids if _patient_priority(p) < 2]
    if search_pid:
        all_pids = [p for p in all_pids if search_pid in p]

    st.markdown(f"**Pacjentów do wyświetlenia:** {len(all_pids):,}")

    PER_PAGE = 50
    n_pages = max(1, (len(all_pids) + PER_PAGE - 1) // PER_PAGE)
    page = min(st.session_state.patient_page, n_pages - 1)

    if n_pages > 1:
        pg1, pg2, pg3 = st.columns([1, 3, 1])
        with pg1:
            if st.button("◀ Poprzednia") and page > 0:
                st.session_state.patient_page = page - 1
                st.rerun()
        with pg2:
            st.caption(f"Strona {page + 1} z {n_pages}")
        with pg3:
            if st.button("Następna ▶") and page < n_pages - 1:
                st.session_state.patient_page = page + 1
                st.rerun()

    page_pids = all_pids[page * PER_PAGE : (page + 1) * PER_PAGE]

    for pid in page_pids:
        pinfo = cache.get(pid, {})
        patient_rec = patients.get(pid, {})
        gender_raw = patient_rec.get("gender", "")
        gender = {"male": "mężczyzna", "female": "kobieta"}.get(gender_raw, "nieznana")
        birth_year = (patient_rec.get("birthDate") or "?")[:4]

        obs_indices = pinfo.get("indices", [])
        obs_list_full = [observations[i] for i in obs_indices]

        active_obs = [o for o in obs_list_full if o.get("id") not in deleted_ids]
        n_err_active = sum(1 for o in active_obs if o.get("_category") in ("red", "yellow"))
        n_susp_active = sum(
            1 for o in active_obs
            if o.get("_outlier") or _interp_code(o) in ("L", "H")
        )
        active_count = len(active_obs)

        if n_err_active > 0:
            icon = "🔴"
        elif n_susp_active > 0:
            icon = "🟡"
        else:
            icon = "✅"

        ml_icon = ""
        if _ML_MODEL is not None:
            obs_list_for_ml = [observations[i] for i in obs_indices]
            ml_anomaly = next(
                (o.get("_ml_anomaly") for o in obs_list_for_ml if "_ml_anomaly" in o),
                None,
            )
            if ml_anomaly is True:
                ml_icon = " | 🧠 Podejrzany (ML)"
            elif ml_anomaly is False:
                ml_icon = " | 🧠 OK (ML)"

        header = (
            f"{icon} Pacjent ID: {pid} | {gender} | ur. {birth_year} | "
            f"{active_count} aktywnych obs. ({n_susp_active} podejrzanych, {n_err_active} błędów)"
            f"{ml_icon}"
        )

        with st.expander(header, expanded=False):
            obs_list = obs_list_full

            local_errors = [
                o for o in obs_list
                if o.get("_category") in ("red", "yellow") and o.get("id") not in deleted_ids
            ]
            if local_errors:
                if st.button(
                    f"🗑️ Usuń błędy tego pacjenta ({len(local_errors)})",
                    key=f"del_err_{pid}",
                ):
                    for o in local_errors:
                        st.session_state.deleted_ids.add(o["id"])
                    st.rerun()

            df = _build_obs_df(obs_list, deleted_ids)

            if df.empty:
                st.info("Brak aktywnych obserwacji dla tego pacjenta.")
                continue

            display_cols = ["Badanie", "Wartość", "Jednostka", "Norma", "Interpretacja", "Status"]
            editor_key = f"editor_{pid}"

            editor_df = df[["Zaznacz"] + display_cols].reset_index(drop=True)
            id_series = df["ID"].reset_index(drop=True)

            edited = st.data_editor(
                editor_df,
                column_config={
                    "Zaznacz": st.column_config.CheckboxColumn(
                        "Usuń?", help="Zaznacz wiersze do usunięcia", default=False
                    ),
                },
                hide_index=True,
                use_container_width=True,
                key=editor_key,
            )

            if st.button("🗑️ Usuń zaznaczone", key=f"del_sel_{pid}"):
                checked_indices = edited.index[edited["Zaznacz"].fillna(False)].tolist()
                to_delete = set(id_series.iloc[checked_indices].tolist())
                if to_delete:
                    st.session_state.deleted_ids.update(to_delete)
                    if editor_key in st.session_state:
                        del st.session_state[editor_key]
                    st.rerun()


def screen_export() -> None:
    observations: List[dict] = st.session_state.observations
    deleted_ids: Set[str] = st.session_state.deleted_ids
    stats: dict = st.session_state.proc_stats

    st.title("🔬 System kontroli jakości i standaryzacji danych laboratoryjnych w standardzie HL7 FHIR")
    st.markdown("### Eksport danych")

    n_total = stats.get("total", 0)
    n_red_auto = stats.get("red", 0)
    n_manually = len(deleted_ids)
    n_exported = sum(
        1 for o in observations
        if o.get("_category") != "red" and o.get("id") not in deleted_ids
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Łącznie rekordów", f"{n_total:,}")
    c2.metric("Usuniętych automatycznie 🔴", f"{n_red_auto:,}")
    c3.metric("Usuniętych przez operatora", f"{n_manually:,}")
    c4.metric("Do eksportu", f"{n_exported:,}")

    summary_stats = {
        "Łącznie rekordów":                          n_total,
        "Poprawnych (w normie)":                     stats.get("in_norm", 0),
        "Z ostrzeżeniem (poza normą)":               stats.get("out_of_norm", 0),
        "Usuniętych automatycznie (błędy strukturalne)": n_red_auto,
        "Oznaczonych jako outlier":                  stats.get("outliers", 0),
        "Usuniętych przez operatora":                n_manually,
        "Niepokrytych przez reference_ranges.json":  stats.get("unknown_test", 0),
        "Przeliczonych jednostek":                   stats.get("converted", 0),
    }

    st.markdown("### Podsumowanie raportu jakości")
    st.dataframe(
        pd.DataFrame(list(summary_stats.items()), columns=["Metryka", "Wartość"]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    ec1, ec2 = st.columns(2)
    with ec1:
        jsonl_bytes = export_jsonl(observations, deleted_ids)
        st.download_button(
            label="⬇️ Pobierz oczyszczony JSONL",
            data=jsonl_bytes,
            file_name="observations_clean.jsonl",
            mime="application/json",
        )
    with ec2:
        csv_bytes = export_report_csv(observations, deleted_ids, summary_stats)
        st.download_button(
            label="⬇️ Pobierz raport jakości (CSV)",
            data=csv_bytes,
            file_name="quality_report.csv",
            mime="text/csv",
        )

    st.markdown("---")
    if st.button("◀ Powrót do przeglądu"):
        st.session_state.screen = "review"
        st.rerun()

_screen = st.session_state.screen
if _screen == "upload":
    screen_upload()
elif _screen == "review":
    screen_review()
elif _screen == "export":
    screen_export()
