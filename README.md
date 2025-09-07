# Prognoza pogody dla podróżników 🌍

Projekt końcowy w Pythonie z wykorzystaniem **Streamlit**.

## Opis
Aplikacja wyznacza trasę podróży (Nominatim + OSRM), próbkowanie punktów co X km i pobiera prognozę pogody na 7 dni z API [Open-Meteo](https://open-meteo.com/).  
Uwzględnia opady, wiatr, temperaturę i kody pogodowe.  
Pozwala ocenić ryzyko podróży w zależności od wybranego środka transportu (samochód, rower, pieszo).  
Wyniki prezentowane są na mapie, w tabelach i wykresach. Możliwy eksport danych do CSV.

## Funkcje
- Geokodowanie lokalizacji (start, meta, punkty pośrednie)  
- Wyznaczanie trasy i próbkowanie co X km  
- Prognoza pogody na 7 dni dla trasy  
- Presety ryzyka dla transportu + możliwość ręcznej regulacji  
- Filtry maks. opadu i wiatru  
- Rekomendacja najlepszych dni  
- Mapa trasy, tabela, wykresy, szczegóły punktów  
- Eksport do pliku CSV  

## Wymagania
- Python 3.9+  
- Biblioteki z pliku `requirements.txt`

## Instalacja i uruchomienie
1. Sklonuj repozytorium:
   ```bash
   git clone https://github.com/nhypen/projekt_koncowy.git
   cd projekt_koncowy
