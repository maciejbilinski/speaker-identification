# Dane
Common Voice Corpus 13.0 z [https://commonvoice.mozilla.org/pl/datasets](https://commonvoice.mozilla.org/pl/datasets), wypakowane w głównym folderze, tak żeby pojawiły się pliki train.tsv, test.tsv i folder clips.

# Przygotowanie danych treningowych
Zrobiłem mały preprocessing, żeby dało się tworzyć batche

    python src\prepare.py

# Trening

    python src\main.py

# Ewaluacja


- zmiana MelSpectrogramu na inne hyperparams
- zmiana architektury
- zmiana datasetu (!) -> streaming angielskiego datasetu
batch=64 wydaje sie okej, ewentualnie 128