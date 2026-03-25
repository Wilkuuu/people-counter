# YOLO People Counter (IN/OUT)

Skrypt zlicza osoby na nagraniu wideo przy pomocy YOLO + trackera i liczy `IN`/`OUT` po przekroczeniu zdefiniowanej linii.

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

- `counting.line.{x1,y1,x2,y2}`: linia zliczania (kierunek zależy od orientacji linii).
- `processing.frame_step`: co ile klatek wykonywać inferencję (`1` = każda klatka).
- `model.device`: `"0"` dla pierwszego GPU, `"cpu"` dla CPU.
- `processing.min_crossing_gap_frames`: minimalny odstęp między kolejnymi zliczeniami jednego `track_id`.

## 4) Uruchomienie

```bash
python main.py --video /sciezka/do/video.mp4 --config config.yaml --output-dir outputs/run1
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

- `summary.json` z podsumowaniem (`total_in`, `total_out`, `net`, liczba klatek).
- `events.csv` z eventami przekroczeń (czas, `track_id`, kierunek).
- `preview.mp4` (opcjonalnie), jeśli `output.save_preview_video: true`.
- `checkpoint.json` do wznowienia pracy od ostatniej zapisanej klatki.

## 6) Strojenie pod 30h materiału

- Zacznij od modelu `yolov8n.pt`, potem podbij do większego, jeśli jakość jest za słaba.
- Dla przyspieszenia zwiększ `processing.frame_step` (np. 2-3), kosztem dokładności.
- Ustaw linię tam, gdzie ruch jest możliwie prostopadły do kierunku przejścia.
- Przetestuj 3-5 krótkich fragmentów i porównaj z ręcznym liczeniem, potem skaluj na całość.
