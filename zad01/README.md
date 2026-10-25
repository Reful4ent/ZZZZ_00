# P1 photometric — стенд глобального сравнения изображений (ТЗ 01, раздел 3.2)

Реализует MSE, NCC, ZNCC, пересечение совместных RGB-гистограмм 8×8×8, среднее CIEDE2000
и четыре перцептивных хеша (aHash64, dHash64, pHash64, wHash64) на одном списке пар
KADID-10k и синтетических воздействий. Выход: исходные оценки пар, пороги, решения,
кривые устойчивости и время каждого метода. Агрегирование между программами — P4.

## Установка и запуск

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e .[test]
# KADID-10k скачать вручную: https://database.mmsp-kn.de/kadid-10k-database.html
# и указать kadid.root (каталог с images/ и dmos.csv) в configs/p1_confirm.yaml
.venv\Scripts\python -m photometric check-data --config configs/p1_confirm.yaml
.venv\Scripts\python -m photometric run --config configs/p1_confirm.yaml
.venv\Scripts\python -m pytest
```

Для дымового прогона можно задать `kadid.originals: [I01, ..., I08]` (минимум 4 оригинала в каждой части).

## API (раздел 6)

```python
from photometric import prepare_image, represent, compare, score_pair
img = prepare_image("I01.png", "sRGB")            # RGB float64 [0,1]
r = represent(img, "phash64")                      # или {"method_id": "phash64", ...}
res = compare(r, represent(prepare_image("I02.png"), "phash64"))   # Result(value, status)
rows = score_pair({"query_path": ..., "reference_path": ...}, config)
```

## Выходные данные (`output_dir`)

| Файл | Содержание |
|---|---|
| `manifest.parquet` | все пары всех опытов (одни и те же для всех методов) |
| `predictions.parquet` | строка на (pair_id, method_id): raw_value, status, score, threshold, decision, тайминги |
| `thresholds.json` | порог на метод, Recall/FPR/Coverage на validation, флаг fallback |
| `representation_cache/` | кеш представлений `{method_id}-{config_hash}/{image}.npz` |
| `metrics/e1_*.csv` | P1-E1: по split/методу, по типу, по типу×уровню, макро по original_id |
| `metrics/e2_curves*.csv` | P1-E2: кривые от параметра, шум — среднее и std по seeds |
| `metrics/e3_*.csv` | P1-E3: перестановка пикселей, прямая проверка инвариантности гистограммы |
| `metrics/e4_timing*.csv/json` | P1-E4: время представления, сравнения, полное |
| `curves/{test,validation}/*.png` | диагностические кривые |
| `synthetic/` | PNG + params.json синтетических воздействий (`synthetic: true`) |
| `split.csv`, `images.parquet`, `run_meta.json` | разбиение, состав базы, версии, seeds, SHA-256 манифеста |

## Зафиксированные проектные решения (не заданные ТЗ явно)

- **Разбиение**: validation/test по `original_id`, PCG64(`split.seed`), доли из YAML (по умолчанию 40/41).
  Отрицательные пары — только из той же части. Порог один на метод: подбирается на validation
  P1-E1 и применяется к test P1-E1, P1-E2, P1-E3.
- **Метрики**: Recall_all = TP / все положительные, FPR_all = FP / все отрицательные; undefined — отказ
  (решение 0), но в знаменателях учитывается. Coverage = доля определённых оценок.
- **Undefined**: NCC — `ZERO_ENERGY`, если Σx² ≤ 1e-12; ZNCC — `ZERO_VARIANCE`, если (1/n)Σ(x−x̄)² ≤ 1e-12;
  попиксельные меры при разных размерах — `SIZE_MISMATCH` (resize запрещён).
- **Bilinear resize хешей**: `skimage.transform.resize(order=1, anti_aliasing=False, mode="edge")`.
  Вариант записан в параметры метода; другой вариант даёт другой `config_hash`
  (и для публикации должен регистрироваться под новым method_id).
- **JPEG (P1-E2)**: Pillow/libjpeg, chroma subsampling 4:4:4, `optimize=False`; версия libjpeg в `run_meta.json`.
- **Синтетика хранится в 8-битном PNG** и оценивается после повторного декодирования. Шум — clip в [0,1].
  Контроли (a=1, b=0, γ=1, σ=0) дают точную копию оригинала.
- **P1-E3**: `identity` (запрос = оригинал, 1+3 пары) и `permuted` (перестановка RGB-троек, пара с оригиналом
  помечена label=0: общего пространственного сюжета нет).
- **P1-E4**: правила ТЗ 04 недоступны; используется `perf_counter_ns`, прогрев и повторы, протокол
  записан в `e4_timing_meta.json`.
- **Кеш**: на диск пишутся компактные представления всех изображений и все представления оригиналов
  (≈1 ГБ для 81 оригинала 512×384); попиксельные представления запросов не сохраняются
  (`cache.persist_pixel_queries`).

## Соответствие приёмке

| Пункт | Тесты |
|---|---|
| ПРИЁМ-Ф1 | `tests/test_acceptance_f1.py` |
| ПРИЁМ-Ф2 | `tests/test_acceptance_f2.py` |
| ПРИЁМ-Ф3 | `tests/test_acceptance_f3.py`, эталон `tests/fixtures/ciede2000_sharma.csv` (34 пары Sharma) |
| ПРИЁМ-Ф4 | `tests/test_acceptance_f4.py`, `tests/test_pipeline.py::test_rerun_same_manifest` |
| Формулы и правила методов | `tests/test_methods.py`, `tests/test_thresholds.py`, `tests/test_synthetic.py` |
