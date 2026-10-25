# План запуска и использования P1 photometric (ТЗ 01, раздел 3.2)

## Назначение программы

Программа измеряет, насколько надёжно 9 методов глобального сравнения узнают «тот же исходный
сюжет» при фотометрических искажениях: MSE, NCC, ZNCC, пересечение RGB-гистограмм 8×8×8, среднее
CIEDE2000, aHash64, dHash64, pHash64, wHash64. Цель — измерить поведение методов, а не обеспечить
заранее установленное преимущество.

Конвейер одного запуска:

1. Проверка состава KADID-10k: 81 оригинал, 10 125 производных, `dmos.csv`.
2. Разбиение 81 оригинала на validation (40) и test (41) по `original_id`, PCG64(`split.seed`).
3. Пары (ФОТО-02). Для каждого запроса строятся 1 положительная пара (со своим оригиналом) и 3
   отрицательные (с другими оригиналами той же части). Список пар один для всех методов.
4. Синтетика: P1-E2 — сетки b, a, γ, σ (×5 seeds) и JPEG quality; P1-E3 — перестановка пикселей.
5. Оценка каждой пары каждым методом: исходное значение, score (−D или S) и статус.
   Возможные статусы undefined: `ZERO_ENERGY`, `ZERO_VARIANCE`, `SIZE_MISMATCH`.
6. Один порог на метод (Г2-167), подбирается на validation P1-E1: max Recall_all при FPR_all ≤ 0.05.
7. Решения, метрики Recall_all, FPR_all и Coverage, кривые и время (P1-E4).

---

## Шаг 0. Окружение

Окружение `D:\zad01\.venv` уже создано, пакет установлен в режиме editable. Переустановить при необходимости:

```powershell
cd D:\zad01
python -m venv .venv
.venv\Scripts\python -m pip install -e .[test]
```

## Шаг 1. Данные KADID-10k

1. Скачать архив (около 2,5 ГБ) со страницы https://database.mmsp-kn.de/kadid-10k-database.html.
2. Распаковать, например, в `D:\datasets\kadid10k\` так, чтобы получилось:

```
D:\datasets\kadid10k\images\I01.png ... I81.png
D:\datasets\kadid10k\images\I01_01_01.png ... I81_25_05.png
D:\datasets\kadid10k\dmos.csv
```

## Шаг 2. Конфигурация

В `configs/p1_confirm.yaml` задать:

```yaml
kadid:
  root: D:/datasets/kadid10k
cache:
  memory_mb: 512      # на этой машине мало памяти (по умолчанию 1536)
```

Остальные параметры — проектный протокол ТЗ (seeds, сетки, α = 0.05). **До запуска** их не менять.

## Шаг 3. Проверки перед прогоном

```powershell
.venv\Scripts\python -m photometric check-data --config configs/p1_confirm.yaml
.venv\Scripts\python -m pytest -q
```

Критерии готовности:
- `check-data` выводит `KADID-10k OK: 81 originals, 10125 distorted`;
- все тесты проходят (ПРИЁМ-Ф1…Ф4, формулы методов, пороги, синтетика, конвейер).

## Шаг 4. Дымовой прогон на 8 оригиналах

Скопировать `configs/p1_confirm.yaml` в `configs/p1_smoke.yaml` и изменить:

```yaml
output_dir: outputs/p1_smoke
kadid:
  root: D:/datasets/kadid10k
  originals: [I01, I02, I03, I04, I05, I06, I07, I08]
```

```powershell
.venv\Scripts\python -m photometric run --config configs/p1_smoke.yaml
```

Что проверить:
- прогон завершился строкой `done in … s`, без `Traceback`;
- в `outputs\p1_smoke\` есть `predictions.parquet`, `thresholds.json`, `metrics\`, `curves\`;
- по полю `total_seconds` в `run_meta.json` оценить полный прогон: время растёт примерно
  пропорционально числу оригиналов, то есть полный прогон займёт около ×10. Больше всего времени
  уходит на CIEDE2000.

## Шаг 5. Полный (подтверждающий) прогон

```powershell
.venv\Scripts\python -m photometric run --config configs/p1_confirm.yaml
```

Результаты сохраняются в `D:\zad01\outputs\p1_confirm\`. Повторный запуск берёт представления из
`representation_cache\`, около 1 ГБ на диске. Переопределить каталог результатов можно ключом
`--output-dir`.

## Шаг 6. Контроль результата

- `run_meta.json`:
  - `manifest_sha256` — зафиксировать в тексте; повтор с теми же seeds должен дать то же значение (ПРИЁМ-Ф4);
  - `fallback_thresholds` — методы, для которых ни один порог не дал FPR ≤ 0.05 (Recall = 0). О них
    нужно явно сообщить;
  - `undefined_counts` — сколько и каких undefined-оценок получил каждый метод;
  - `libraries`, `libjpeg` — версии для раздела «Методика».
- `thresholds.json` — порог, Recall_all, FPR_all и Coverage на validation для каждого из 9 методов.
- Наличие всех файлов `metrics\e1_*.csv`, `e2_*.csv`, `e3_*.csv`, `e4_*.csv` и `curves\test\*.png`.

## Шаг 7. Использование результатов в разделе 3.2

Для выводов брать **split = test**: порог подобран на validation.

| Что нужно для текста | Файл |
|---|---|
| Пороги методов | `thresholds.json` |
| Основная таблица сравнения | `metrics\e1_overall.csv`, `metrics\e1_macro_original.csv` (среднее и std по оригиналам) |
| Какой тип и уровень искажения ломает какой метод | `metrics\e1_by_type_level.csv`, `curves\test\e1_recall_by_level.png` |
| Устойчивость к контролируемым факторам (b, a, γ, σ, JPEG) | `metrics\e2_curves.csv`, `curves\test\e2_<фактор>_decisions.png`, `..._scores.png` |
| Контрпример: гистограмма слепа к перестановке пикселей | `metrics\e3_summary.csv`, `metrics\e3_histogram_invariance.csv` |
| Вычислительное время | `metrics\e4_timing.csv`, протокол в `metrics\e4_timing_meta.json` |
| Все исходные оценки пар (для P4) | `predictions.parquet` |

Пример:

```python
import pandas as pd
d = r"D:\zad01\outputs\p1_confirm"
e1 = pd.read_csv(fr"{d}\metrics\e1_overall.csv")
print(e1[e1.split == "test"][["method_id", "recall_all", "fpr_all", "coverage"]])

e2 = pd.read_csv(fr"{d}\metrics\e2_curves.csv")
print(e2[(e2.split == "test") & (e2.factor == "gamma")][["method_id", "param", "recall_all_mean"]])

pred = pd.read_parquet(fr"{d}\predictions.parquet")   # строка = (pair_id, method_id)
```

Агрегирование между программами выполняет P4: ей передаётся `predictions.parquet` вместе с
`thresholds.json` и `run_meta.json`.

## Использование API на своих изображениях

```python
from photometric import prepare_image, represent, compare
a = prepare_image(r"C:\img\a.png")            # RGB float64 [0,1]
b = prepare_image(r"C:\img\b.png")
r = compare(represent(a, "zncc"), represent(b, "zncc"))
print(r.value, r.status)                       # например 0.93 OK
```

Правило решения: score = −value для расстояний (mse, ciede2000_mean, *hash64) и score = value для
сходств (ncc, zncc, hist_rgb8). Ответ «тот же сюжет», если score ≥ порог метода из `thresholds.json`.
Если `status != "OK"`, результат — отказ.

## Правила протокола

- Не менять seeds, доли разбиения, список типов и уровни после просмотра результатов. Конфиг без
  полного набора уровней 1–5 программа отклоняет.
- Порог один на метод для основного опыта; не подбирать порог отдельно под test-искажение.
- Другой вариант метода регистрировать под новым `method_id`, не подменяя базовый. Пример: хеш с
  антиалиасингом или вариант другой публикации.
- Синтетические данные (`synthetic = true`) не выдавать за официальные уровни KADID.

## Открытые вопросы к согласованию

Подробнее — в `README.md`.

1. Bilinear resize в хешах без антиалиасинга (`skimage`, `order=1`, `anti_aliasing=False`). Этот
   выбор заметно влияет на устойчивость хешей к шуму.
2. JPEG в P1-E2: Pillow/libjpeg, chroma subsampling 4:4:4.
3. P1-E3: пара «перестановка — оригинал» помечена `label = 0`.
4. Протокол замера времени временный, до получения правил ТЗ 04.
5. Полный прогон на изображениях реального размера ещё не выполнялся, время и пиковая память не
   измерены. Сначала выполнить шаг 4.
