# p2geom — программа P2 geometric (ТЗ 02)

Реализация ТЗ 02 «Программа геометрического сопоставления и сравнения областей»:
локальные признаки → предварительные соответствия → RANSAC → проверки модели →
совмещение → общая область → характеристики участков → решение с возможностью
отказа. Конвейер не получает ни истинную гомографию, ни маску дефекта: GT
используют только оценщик (`metrics.py`) и контрольная oracle-ветвь P2-E3.

## Установка

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt     # runtime + pytest
# SuperPoint (ГЕОМ-03) — PyTorch можно держать в отдельном окружении:
python3 -m venv .venv-sp && .venv-sp/bin/pip install -r requirements-superpoint.txt
```

Запуск: `.venv/bin/python -m p2geom <команда> …`, тесты: `.venv/bin/python -m pytest`.

## Быстрая проверка

```bash
.venv/bin/python -m pytest                      # ПРИЁМ-Г1…Г5 + модульные тесты
.venv/bin/python -m p2geom smoke --out runs/smoke
```

`smoke` строит миниатюрные наборы в форматах HPatches и MVTec AD из встроенных
изображений scikit-image и проходит весь протокол (manifest, P2-E1, калибровка
геометрии, P2-E2, synthetic/real, калибровка цвета, P2-E7, абляции). Числа
smoke-прогона не являются результатами.

## Полный протокол

Данные: HPatches (`hpatches-sequences-release`) и MVTec AD (категории carpet,
leather, tile, wood).

```bash
python3 scripts/download_datasets.py            # -> data/hpatches-sequences-release, data/mvtec_ad/<cat>
python3 scripts/download_datasets.py --only mvtec --categories carpet   # выборочно
```

Скрипт без зависимостей, проверяет SHA-256 каждого файла и докачивает при
повторном запуске. Исходные хосты недоступны, поэтому источники — зеркала
Hugging Face: `vbalnt/hpatches` (автор набора, MIT) и `foersben/mvtec-ad`
(исходная структура каталогов; CC BY-NC-SA 4.0 — только некоммерческое
использование). Объём: HPatches ≈ 1.2 ГиБ, MVTec ≈ 0.5–0.7 ГиБ на категорию.

```bash
P=".venv/bin/python -m p2geom"
# 1. HPatches: формат H и проверка направления проекцией патчей -> manifest
$P hpatches-manifest --root data/hpatches-sequences-release --out runs/hp_manifest.parquet
# Разбиение задаёт ТЗ 00 (JSON {"validation": [...], "test": [...]}). Временное:
$P hpatches-split --root data/hpatches-sequences-release --out runs/hp_split.json

# 2. Геометрическая калибровка на validation (τr × n_min × c_min) -> frozen_config.yaml
$P calibrate-geometry --root data/hpatches-sequences-release --split runs/hp_split.json \
   --manifest runs/hp_manifest.parquet --method sift --out runs/geom_calib

# 3. P2-E1 (SIFT, ORB, SuperPoint) и P2-E2 (воздействия) с замороженной конфигурацией
$P geometry --root data/hpatches-sequences-release --split runs/hp_split.json \
   --manifest runs/hp_manifest.parquet --config runs/geom_calib/frozen_config.yaml --out runs/P2-E1
$P perturb --root data/hpatches-sequences-release --split runs/hp_split.json \
   --config runs/geom_calib/frozen_config.yaml --out runs/P2-E2

# 4. MVTec: synthetic_validation / synthetic_test (+ generation_report.csv) и пары real/normal
$P mvtec-build --root data/mvtec_ad --out runs/synthetic
$P mvtec-pairs --root data/mvtec_ad --kind normal_validation --out runs/normal_val_pairs.parquet
$P mvtec-pairs --root data/mvtec_ad --kind real_test --out runs/real_test_pairs.parquet

# 5. Цветовая калибровка: synthetic validation + normal validation против dev-галереи
$P inspect --pairs runs/synthetic/synthetic_validation/*/manifest.parquet runs/normal_val_pairs.parquet \
   --config runs/geom_calib/frozen_config.yaml --out runs/inspect_validation
$P calibrate-color --run runs/inspect_validation --out runs/color_calib

# 6. P2-E7: synthetic и real_test раздельно, замороженный протокол
$P inspect --pairs runs/synthetic/synthetic_test/*/manifest.parquet \
   --config runs/color_calib/frozen_config.yaml --out runs/P2-E7_synthetic --diagnostics 20
$P inspect --pairs runs/real_test_pairs.parquet \
   --config runs/color_calib/frozen_config.yaml --out runs/P2-E7_real

# 7. Абляции P2-E3…P2-E6 (fixed и recalibrated протоколы)
$P ablations --val runs/synthetic/synthetic_validation/*/manifest.parquet \
   --test runs/synthetic/synthetic_test/*/manifest.parquet \
   --config runs/color_calib/frozen_config.yaml --out runs/ablations
```

Режимы known-mask и unknown-mask (`--mask-mode`) запускаются в разные каталоги;
сводка отказывается объединять их. Опыт скрытых дефектов: `mvtec-build --hidden`
(ограничение видимости ≥ 50% снимается, сводка `by_defect_visibility` выводит
полностью скрытые случаи отдельно).

## Выходные файлы (раздел 9)

Каждый прогон пишет `predictions.parquet`, `correspondences.parquet`,
`homographies.npz` (ключи `<pair_id>__H`, `__H0`, `__A_I`, `__A_E`, `__trials`),
`region_metrics.parquet`, `reject_counts.csv`, `config.yaml`, `run_meta.json`,
при `--diagnostics N` — PNG (совмещение, Ω̂_com, карта δ00). `q_geom`, `q_align`,
`q_cmp` и `y_pred` хранятся раздельно; `y_pred` пуст (NaN) при отказе ⊥.

## Соответствие требований и кода

| Требование | Модуль |
|---|---|
| Г-1, Г2-102 (π, A_I, A_E, crop) | `homography.py`, `features.py` |
| ГЕОМ-02 SIFT/ORB, 2000 точек, ≤1024 px | `features.py` |
| Г2-103 ratio-test в обоих направлениях → взаимность, tie-break по индексу | `matching.py` |
| ГЕОМ-03 SuperPoint, URL/SHA-256/лицензия/commit, NOT_RUN | `superpoint.py` |
| Г2-104…Г2-107 RANSAC, нормализованный DLT, lex-выбор, уточнение, REFINE_FAILED | `ransac.py` |
| OpenCV findHomography (отдельный baseline) | `ransac.py`, `estimator: opencv` |
| Г2-108…Г2-110, H_adm | `validate.py`, `homography.check_admissible` |
| Г2-99, Г2-100, Г2-111, ГЕОМ-04 | `warp.py` |
| Г2-112…Г2-118, Г2-120 | `regions.py`, `color.py` |
| Г2-121…Г2-123 | `decision.py` |
| Раздел 2 (сетка воздействий), ОБЛ-01, ОБЛ-02, Г-2 | `synth.py`, `datasets/mvtec.py` |
| ОБЛ-03 multi-reference_inspection | `datasets/mvtec.py` |
| HPatches, manifest, отрицательные пары seed=2026 | `datasets/hpatches.py` |
| Г-3, Г-4, Г-5, матрица 2×3 | `metrics.py` |
| Калибровки (раздел 2, раздел 7) | `calibration.py` |
| P2-E1…P2-E7 | `experiments.py` |
| ПРИЁМ-Г1…Г5 | `tests/test_acceptance.py` |

## Принятые решения там, где ТЗ оставляет выбор

* **ТЗ 00 и ТЗ 01 недоступны.** Разбиения HPatches и MVTec по умолчанию — временные
  (seed=2026, описаны в `datasets/*.py`); нормативное разбиение передаётся через
  `--split`. CIEDE2000 — `skimage.color.deltaE_ciede2000` (D65/2°, kL=kC=kH=1).
* **Координаты**: центр пикселя — целые координаты (как в OpenCV); матрица resize
  `x' = s(x + 0.5) − 0.5` проверена на `cv2.resize` (ПРИЁМ-Г2).
* **RANSAC** работает в исходных координатах (точки переводятся через A⁻¹), поэтому
  εg сразу в пикселях исходного E; `H_work = A_E Ĥ A_I⁻¹` сохраняется для проверки
  Г2-102. Вырожденные выборки: коллинеарные/совпадающие тройки и несогласованная
  ориентация троек в I и E. Адаптивная остановка включена и протоколируется
  (`stop_reason`, `required_trials`); `adaptive: false` — полный бюджет 5000.
* **H_adm**: знаменатели проверяются в углах ROI I для `N_E H N_I⁻¹`, нормированной
  по норме Фробениуса; якобиан `det(H)/w³` должен быть положителен в углах и центре
  (зеркальное отражение недопустимо); перенесённый контур — без самопересечения.
* **Gate «none» (P2-E4)** оставляет только допустимость H: без неё warp невозможен.
* **Компенсация (Г2-116)** решается ограниченным линейным МНК (`lsq_linear`) без
  учёта clip; clip применяется к результату. Пустая опорная полоса → `CAL_EMPTY`,
  компенсация не применяется.
* **Проверка направления HPatches**: средний NCC патчей сетки E против их образов в I
  при гипотезах «файл E→I» и «файл I→E»; вердикт в manifest; при `I_TO_E` загрузчик
  останавливается (`on_direction_mismatch="error"`).
* **Генератор**: окклюдер — серый прямоугольник целиком в кадре; царапина — изменение
  L* на ±25 вдоль отрезка длиной 0.5 стороны центральной области; вставка —
  квадратный фрагмент другого изображения того же split. Сетки T и O перебираются
  циклически по фиксированной перестановке (seed). Допускаемый диапазон отношения
  площадей для синтетики — `SYNTH_AREA_RATIO_RANGE` в `synth.py`.
* **Начальные цветовые пороги** (`tau_pix=5, tau_d=5, tau_a=0.1, tau_B=0.1, tau_D=5`)
  нужны только для smoke; рабочие выбираются `calibrate-color`. Если ограничения
  FPR ≤ 0.05 и Coverage ≥ 0.5 недостижимы, калибровка возвращает `infeasible` и
  пороги не замораживаются.
* **SuperPoint**: веса `superpoint_v1.pth` скачиваются вручную по
  `superpoint_weights_url`; путь, SHA-256 и commit задаются в конфиге
  (`features.superpoint_*`). NMS — через max-pooling.
