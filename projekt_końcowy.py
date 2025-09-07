
# app.py
# ──────────────────────────────────────────────────────────────────────────────
# Prognoza pogody dla podróżników — wersja połączona:
# - Geokodowanie (Nominatim/OSM)
# - Trasa (OSRM)
# - Próbkowanie trasy co X km
# - Prognoza 7-dniowa (Open-Meteo)
# - Ocena ryzyka z presetami dla środka transportu + możliwość ręcznej zmiany wag
# - Filtry widoczności dni: maks. opad i maks. wiatr
# - Rekomendacje najlepszych dni
# - Mapa, tabela, wykresy, szczegóły punktów, eksport CSV
#
# Uruchom:
#   1) pip install -r requirements.txt
#   2) streamlit run app.py
#
# Uwaga: potrzebny internet do usług zewnętrznych.
# ──────────────────────────────────────────────────────────────────────────────

import math
import time
import requests
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium

# ──────────────────────────────────────────────────────────────────────────────
# KONFIG
# ──────────────────────────────────────────────────────────────────────────────
APP_NAME = "TravelWeatherPlanner/2.0 (contact: example@example.com)"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OSRM_ROUTE_URL = "https://router.project-osrm.org/route/v1/driving/{coords}"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

SAMPLE_EVERY_KM_DEFAULT = 25

# Presety wag ryzyka dla środków transportu
TRANSPORT_PRESETS = {
    "Samochód": {"precip": 0.8, "wind": 0.03, "code": 0.8, "wind_thr": 60.0, "precip_thr": 12.0},
    "Rower":    {"precip": 1.2, "wind": 0.08, "code": 1.2, "wind_thr": 35.0, "precip_thr": 8.0},
    "Pieszo":   {"precip": 1.5, "wind": 0.10, "code": 1.5, "wind_thr": 25.0, "precip_thr": 6.0},
}

# Zestaw kodów pogodowych WMO uznanych za niekorzystne
BAD_WEATHER_CODES = set([
    45, 48,
    51, 53, 55, 56, 57,
    61, 63, 65, 66, 67, 80, 81, 82,
    71, 73, 75, 77, 85, 86,
    95, 96, 99
])

# ──────────────────────────────────────────────────────────────────────────────
# FUNKCJE POMOCNICZE
# ──────────────────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R*c


@st.cache_data(show_spinner=False, ttl=60*60)
def geocode_place(q):
    params = {"q": q, "format": "json", "limit": 1}
    headers = {"User-Agent": APP_NAME}
    r = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=30)
    if r.status_code != 200:
        return None
    data = r.json()
    if not data:
        return None
    item = data[0]
    return float(item["lat"]), float(item["lon"]), item.get("display_name", q)


@st.cache_data(show_spinner=False, ttl=30*60)
def osrm_route(coords):
    """
    coords: lista [(lon, lat), ...]
    Zwraca: ([(lat, lon), ...], distance_km, duration_h) lub None
    """
    if len(coords) < 2:
        return None
    coord_str = ";".join([f"{lon:.6f},{lat:.6f}" for lon, lat in coords])
    url = OSRM_ROUTE_URL.format(coords=coord_str)
    params = {
        "overview": "full",
        "geometries": "geojson",
        "annotations": "false",
        "steps": "false",
        "alternatives": "false"
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return None
    data = r.json()
    if not data.get("routes"):
        return None
    route = data["routes"][0]
    geometry = route["geometry"]["coordinates"]  # [[lon, lat], ...]
    distance_km = route["distance"] / 1000.0
    duration_h = route["duration"] / 3600.0
    latlon = [(lat, lon) for lon, lat in geometry]
    return latlon, distance_km, duration_h


def sample_route(latlon_points, every_km):
    if not latlon_points:
        return []
    sampled = [latlon_points[0]]
    acc = 0.0
    for i in range(1, len(latlon_points)):
        lat1, lon1 = latlon_points[i-1]
        lat2, lon2 = latlon_points[i]
        seg = haversine_km(lat1, lon1, lat2, lon2)
        if acc + seg < every_km:
            acc += seg
        else:
            sampled.append((lat2, lon2))
            acc = 0.0
    if sampled[-1] != latlon_points[-1]:
        sampled.append(latlon_points[-1])
    return sampled


@st.cache_data(show_spinner=False, ttl=30*60)
def fetch_open_meteo_7day(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,weathercode",
        "timezone": "auto"
    }
    r = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    r.raise_for_status()
    d = r.json().get("daily", {})
    df = pd.DataFrame({
        "date": pd.to_datetime(d.get("time", [])),
        "tmax": d.get("temperature_2m_max", []),
        "tmin": d.get("temperature_2m_min", []),
        "precip_mm": d.get("precipitation_sum", []),
        "wind_max_kmh": d.get("wind_speed_10m_max", []),
        "wcode": d.get("weathercode", []),
    })
    return df


def aggregate_route_weather(dfs, w_precip, w_wind, w_code):
    if not dfs:
        return pd.DataFrame()
    base = dfs[0][["date"]].copy()
    merged = base
    precip_cols, wind_cols, wcode_cols, tmax_cols, tmin_cols = [], [], [], [], []
    for i, df in enumerate(dfs):
        dfi = df.add_suffix(f"_{i}")
        merged = merged.merge(dfi, left_on="date", right_on=f"date_{i}", how="left")
        precip_cols.append(f"precip_mm_{i}")
        wind_cols.append(f"wind_max_kmh_{i}")
        wcode_cols.append(f"wcode_{i}")
        tmax_cols.append(f"tmax_{i}")
        tmin_cols.append(f"tmin_{i}")

    out = pd.DataFrame()
    out["date"] = merged["date_0"]
    out["precip_mm_max"] = merged[precip_cols].max(axis=1)
    out["wind_max_kmh_max"] = merged[wind_cols].max(axis=1)
    out["tmax_avg"] = merged[tmax_cols].mean(axis=1)
    out["tmin_avg"] = merged[tmin_cols].mean(axis=1)

    def worst_code(row):
        codes = [int(row[c]) for c in wcode_cols]
        bad = [c for c in codes if c in BAD_WEATHER_CODES]
        return bad[0] if bad else codes[0]

    out["wcode_worst"] = merged.apply(worst_code, axis=1)
    out["risk"] = out.apply(
        lambda r: r["precip_mm_max"] * w_precip
                  + r["wind_max_kmh_max"] * w_wind
                  + (w_code if r["wcode_worst"] in BAD_WEATHER_CODES else 0.0),
        axis=1
    )
    return out.sort_values("date").reset_index(drop=True)


def weathercode_label(code):
    mapping = {
        0: "Czyste niebo",
        1: "Głównie pogodnie",
        2: "Częściowe zachmurzenie",
        3: "Zachmurzenie",
        45: "Mgła",
        48: "Szroniąca mgła",
        51: "Mżawka lekka",
        53: "Mżawka umiarkowana",
        55: "Mżawka intensywna",
        56: "Marznąca mżawka lekka",
        57: "Marznąca mżawka intensywna",
        61: "Deszcz lekki",
        63: "Deszcz umiarkowany",
        65: "Deszcz intensywny",
        66: "Marznący deszcz lekki",
        67: "Marznący deszcz intensywny",
        71: "Śnieg lekki",
        73: "Śnieg umiarkowany",
        75: "Śnieg intensywny",
        77: "Ziarnisty śnieg",
        80: "Przelotne opady lekkie",
        81: "Przelotne opady umiarkowane",
        82: "Przelotne opady intensywne",
        85: "Przelotny śnieg lekki",
        86: "Przelotny śnieg intensywny",
        95: "Burza",
        96: "Burza z gradem lekka/umiark.",
        99: "Burza z gradem silna",
    }
    return mapping.get(int(code), f"Kod {int(code)}")


def dataframe_download_bytes(df, filename):
    csv = df.to_csv(index=False).encode("utf-8")
    return csv, filename


def make_map(route_latlon, start_name, dest_name, sampled_pts):
    fmap = folium.Map(location=route_latlon[len(route_latlon)//2], zoom_start=6, control_scale=True)
    folium.PolyLine(route_latlon, weight=6, opacity=0.8, color="blue").add_to(fmap)
    folium.Marker(route_latlon[0], icon=folium.Icon(color="green"), tooltip=f"Start: {start_name}").add_to(fmap)
    folium.Marker(route_latlon[-1], icon=folium.Icon(color="red"), tooltip=f"Meta: {dest_name}").add_to(fmap)
    for i, (lat, lon) in enumerate(sampled_pts):
        folium.CircleMarker(location=(lat, lon), radius=3, weight=1, fill=True, fill_opacity=0.7, popup=f"Punkt {i+1}").add_to(fmap)
    return fmap

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Prognoza dla podróżników", layout="wide")
st.title("Prognoza pogody dla podróżników")

top1, top2, top3 = st.columns([1,1,1])
with top1:
    start_place = st.text_input("Punkt startowy", value="Kraków")
with top2:
    dest_place = st.text_input("Cel podróży", value="Gdańsk")
with top3:
    waypoint_str = st.text_input("Punkty pośrednie (opcjonalnie, przecinki)", value="")

mid1, mid2, mid3 = st.columns([1,1,1])
with mid1:
    transport = st.selectbox("Środek transportu", list(TRANSPORT_PRESETS.keys()), index=0)
with mid2:
    sample_km = st.slider("Próbkowanie trasy co (km)", 10, 100, SAMPLE_EVERY_KM_DEFAULT, 5)
with mid3:
    run = st.button("Zaplanuj trasę i pobierz prognozę")

preset = TRANSPORT_PRESETS[transport]
w_prec_default = preset["precip"]
w_wind_default = preset["wind"]
w_code_default = preset["code"]

filters = st.expander("Filtry i ustawienia zaawansowane")
with filters:
    st.markdown("Presety wag dopasowane do wybranego środka transportu można nadpisać poniżej.")
    c1, c2, c3 = st.columns(3)
    with c1:
        w_prec = st.number_input("Waga: opad [mm]", value=w_prec_default, min_value=0.0, step=0.1)
    with c2:
        w_wind = st.number_input("Waga: wiatr [km/h]", value=w_wind_default, min_value=0.0, step=0.01, format="%.2f")
    with c3:
        w_code = st.number_input("Kara: zły kod pogody", value=w_code_default, min_value=0.0, step=0.1)

    st.markdown("Progi do filtrowania wyświetlanych dni.")
    f1, f2, f3 = st.columns(3)
    with f1:
        precip_thr = st.number_input("Maksymalny opad [mm]", value=preset["precip_thr"], min_value=0.0, step=0.5)
    with f2:
        wind_thr = st.number_input("Maksymalny wiatr [km/h]", value=preset["wind_thr"], min_value=0.0, step=1.0)
    with f3:
        apply_filters = st.checkbox("Ukryj dni przekraczające progi", value=True)

if run:
    with st.spinner("Geokodowanie"):
        geos = []
        start_geo = geocode_place(start_place)
        if not start_geo:
            st.error("Nie udało się zgeokodować punktu startowego.")
            st.stop()
        dest_geo = geocode_place(dest_place)
        if not dest_geo:
            st.error("Nie udało się zgeokodować celu podróży.")
            st.stop()
        geos.append(("START", start_geo))

        waypoints = []
        if waypoint_str.strip():
            for w in [x.strip() for x in waypoint_str.split(",") if x.strip()]:
                g = geocode_place(w)
                if g:
                    waypoints.append((w, g))
                else:
                    st.warning(f"Nie udało się zgeokodować punktu pośredniego: {w}")
            geos.extend(waypoints)

        geos.append(("META", dest_geo))

    coords_lonlat = [(lon, lat) for _, (lat, lon, _) in geos]

    with st.spinner("Wyznaczanie trasy"):
        route_result = osrm_route(coords_lonlat)
        if not route_result:
            st.error("Nie udało się wyznaczyć trasy (OSRM).")
            st.stop()
        route_latlon, dist_km, dur_h = route_result

    st.success(f"Trasa: około {dist_km:.1f} km, około {dur_h:.1f} h")

    sampled = sample_route(route_latlon, every_km=sample_km)
    st.caption(f"Punkty próbkowania: {len(sampled)}")

    # Pobranie prognoz dla punktów
    all_point_dfs = []
    with st.spinner("Pobieranie prognoz pogodowych"):
        for i, (lat, lon) in enumerate(sampled):
            try:
                df = fetch_open_meteo_7day(lat, lon)
                df["point_idx"] = i
                all_point_dfs.append(df)
                time.sleep(0.05)  # łagodniejsze traktowanie API przy wielu punktach
            except Exception as e:
                st.warning(f"Błąd prognozy dla punktu {i+1}: {e}")

    if not all_point_dfs:
        st.error("Nie udało się pobrać prognoz.")
        st.stop()

    dfs_clean = [df.drop(columns=["point_idx"]) for df in all_point_dfs]
    agg = aggregate_route_weather(dfs_clean, w_prec, w_wind, w_code)

    # Filtry prezentacji
    if apply_filters:
        mask = (agg["precip_mm_max"] <= float(precip_thr)) & (agg["wind_max_kmh_max"] <= float(wind_thr))
        agg_filtered = agg[mask].reset_index(drop=True)
    else:
        agg_filtered = agg.copy()

    # Rekomendacje najlepszych dni
    st.subheader("Rekomendowane dni wyjazdu")
    if len(agg_filtered) == 0:
        st.info("Brak dni spełniających filtry. Wyłącz filtry lub zwiększ progi.")
    else:
        best = agg_filtered.nsmallest(3, "risk")[["date", "risk", "precip_mm_max", "wind_max_kmh_max", "tmin_avg", "tmax_avg", "wcode_worst"]].reset_index(drop=True)
        cols = st.columns(len(best))
        for i in range(len(best)):
            r = best.loc[i]
            cols[i].metric(
                label=r["date"].strftime("%A, %Y-%m-%d"),
                value=f"Ryzyko {r['risk']:.2f}",
                delta=f"Opad {r['precip_mm_max']:.1f} mm • Wiatr {r['wind_max_kmh_max']:.0f} km/h"
            )

    st.divider()

    # Mapa
    st.subheader("Mapa trasy")
    fmap = make_map(
        route_latlon=route_latlon,
        start_name=geos[0][1][2],
        dest_name=geos[-1][1][2],
        sampled_pts=sampled
    )
    st_folium(fmap, use_container_width=True, height=480)

    # Tabela zbiorcza
    st.subheader("Prognoza 7-dniowa dla całej trasy (wartości najgorsze po punktach)")
    tbl = agg_filtered.copy()
    tbl["Data"] = tbl["date"].dt.strftime("%Y-%m-%d")
    tbl["Opis pogody"] = tbl["wcode_worst"].apply(weathercode_label)
    tbl["Tmin [°C] (avg)"] = tbl["tmin_avg"].round(1)
    tbl["Tmax [°C] (avg)"] = tbl["tmax_avg"].round(1)
    tbl["Opad [mm] (max)"] = tbl["precip_mm_max"].round(1)
    tbl["Wiatr [km/h] (max)"] = tbl["wind_max_kmh_max"].round(0)
    tbl["Ryzyko"] = tbl["risk"].round(2)
    show_cols = ["Data", "Opis pogody", "Ryzyko", "Opad [mm] (max)", "Wiatr [km/h] (max)", "Tmin [°C] (avg)", "Tmax [°C] (avg)"]
    st.dataframe(tbl[show_cols], use_container_width=True, hide_index=True)

    # Wykresy
    st.subheader("Wykresy")
    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.line_chart(agg.set_index("date")[["precip_mm_max"]].rename(columns={"precip_mm_max": "Opad [mm] (max)"}))
    with chart_cols[1]:
        st.line_chart(agg.set_index("date")[["wind_max_kmh_max"]].rename(columns={"wind_max_kmh_max": "Wiatr [km/h] (max)"}))

    # Szczegóły dla wybranego dnia
    st.subheader("Szczegóły punktów trasy dla wybranego dnia")
    day_list = list(agg["date"].dt.date.unique())
    if len(day_list):
        picked = st.selectbox("Dzień", options=day_list, index=0)
        rows = []
        for i, dfp in enumerate(all_point_dfs):
            row = dfp[dfp["date"].dt.date == picked]
            if not row.empty:
                r = row.iloc[0]
                rows.append({
                    "Punkt": i+1,
                    "Opad [mm]": round(float(r["precip_mm"]), 1),
                    "Wiatr [km/h]": round(float(r["wind_max_kmh"]), 0),
                    "Tmin [°C]": round(float(r["tmin"]), 1),
                    "Tmax [°C]": round(float(r["tmax"]), 1),
                    "Kod pogody": int(r["wcode"]),
                    "Opis": weathercode_label(r["wcode"])
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("Brak danych dla wybranego dnia.")

    # Eksport
    st.subheader("Eksport danych")
    csv_bytes, fname = dataframe_download_bytes(agg, "prognoza_trasa_7dni.csv")
    st.download_button("Pobierz CSV (pełne zestawienie)", data=csv_bytes, file_name=fname, mime="text/csv")

else:
    st.info("Uzupełnij pola i uruchom planowanie trasy.")
