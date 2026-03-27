# YOLO People Counter + Webpanel

Skrypt zlicza osoby na nagraniu wideo przy pomocy YOLO + trackera. Domyślnie używa trybu strefowego lewo/prawo (`zone`), który jest stabilniejszy przy obróconym obrazie. Generuje też raport HTML i ma prosty webpanel FastAPI.

## 1) Wymagania

- Python 3.10+ (zalecane)
- NVIDIA GPU + CUDA (dla szybkiego przetwarzania 30h materiału)
- Linux/Windows/macOS

## 2) Instalacja

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Uwaga: dla GPU zainstaluj wersję PyTorch zgodną z Twoją wersją CUDA.
Instrukcja: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

## 3) Konfiguracja

Skopiuj i edytuj:

```bash
cp config.example.yaml config.yaml
```

Najważniejsze pola:

- `preset`: `speed` / `balanced` / `accuracy` (szybkość kontra precyzja).
- `counting.mode`: `zone` (zalecane) lub `line`.
- `counting.zone.split_x`: pionowy podział kadru (`null` = środek szerokości).
- `counting.zone.margin_px`: martwa strefa przy granicy, ogranicza fałszywe przejścia.
- `counting.line.{x1,y1,x2,y2}`: linia dla trybu `line` (opcjonalnie).
- `processing.frame_step`: co ile klatek wykonywać inferencję (`1` = każda klatka).
- `model.device`: `"0"` dla pierwszego GPU, `"cpu"` dla CPU.
- `processing.min_crossing_gap_frames`: minimalny odstęp między kolejnymi zliczeniami jednego `track_id`.

## 4) Uruchomienie

```bash
python main.py --video /sciezka/do/video.mp4 --config config.yaml --output-dir outputs/run1
```

Wymuszenie presetu:

```bash
python main.py --video /sciezka/do/video.mp4 --config config.yaml --output-dir outputs/run1 --preset accuracy
```

Wznowienie po checkpoint:

```bash
python main.py --video /sciezka/do/video.mp4 --config config.yaml --output-dir outputs/run1 --resume
```

Nadpisanie urządzenia z CLI:

```bash
python main.py --video /sciezka/do/video.mp4 --config config.yaml --device 0
```

## 5) Wyniki

W `output-dir` otrzymasz:

- `summary.json` z podsumowaniem (`total_crossings`, `from_left`, `from_right`, liczba klatek).
- `events.csv` z eventami przejść między strefami (`LEFT_TO_RIGHT` / `RIGHT_TO_LEFT`).
- `report.html` z czytelnym podsumowaniem i tabelą zdarzeń.
- `preview.mp4` (opcjonalnie), jeśli `output.save_preview_video: true`.
- `checkpoint.json` do wznowienia pracy od ostatniej zapisanej klatki.

## 6) Strojenie pod 30h materiału

- Zacznij od presetu `balanced`; dla par osób przejdź na `accuracy`.
- Jeśli pary nadal są zbijane, obniż `model.conf` i podnieś `model.iou` o 0.05-0.10.
- Dla przyspieszenia zwiększ `processing.frame_step` (np. 2-3), kosztem dokładności.
- W trybie `zone` ustaw `split_x` tak, by osoby naturalnie przechodziły między lewą i prawą stroną.
- Zwiększ `margin_px` jeśli tracker drga przy granicy i pojawiają się fałszywe przejścia.
- Przetestuj 3-5 krótkich fragmentów i porównaj z ręcznym liczeniem, potem skaluj na całość.

## 7) Webpanel

Uruchom:

```bash
uvicorn web.app:app --reload --port 8000
```

Następnie otwórz:

- [http://localhost:8000](http://localhost:8000)

Funkcje MVP:
- upload filmu,
- wybór presetu,
- wynik analizy z liczbą osób,
- oś czasu zdarzeń (przycisk przeskakuje player do momentu zdarzenia),
- link do `report.html`, `summary.json`, `events.csv`.

## 8) Windows `.exe` (onefile)

Docelowy wariant: pojedynczy plik `people-counter.exe`, który po dwukliku uruchamia lokalny serwer i otwiera przeglądarkę.

### Build (wykonaj na Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pyinstaller people_counter.spec --clean
```

Wynik:
- `dist\people-counter.exe`

### Uruchomienie

- dwuklik `people-counter.exe`
- aplikacja otworzy `http://127.0.0.1:8000`

### Uwagi

- Build `.exe` powinien być robiony na Windows (cross-build z Linuxa nie jest wspierany).
- Przy pierwszym uruchomieniu firewall Windows może zapytać o zgodę.
- Jeśli source video ma kodek HEVC, aplikacja przygotowuje `web_preview.mp4` (H.264) dla kompatybilności z przeglądarką.
