# Спецификация: общий протокол экспериментального комплекса (ТЗ 00)

| | |
|---|---|
| **Статус** | Черновик на ревью (архитектурный путь: спека → план → реализация) |
| **Дата** | 2026-10-24 |
| **Основание** | `TZ_00_Common_Protocol.docx`, редакция 1.0 от 23 октября 2026 года |
| **Охват** | Всё содержимое ТЗ 00: состав комплекса, датасеты и разбиения, контракт данных, семантика решения и метрики, воспроизводимость, интерфейс, комплект поставки |
| **Не входит** | Алгоритмы P1–P3 и планы экспериментов из ТЗ 01–04 (см. раздел 1.3 и раздел 13) |

---

## 1. Цель, область и границы

### 1.1 Цель

Реализовать воспроизводимый исследовательский комплекс — один репозиторий с общим пакетом и четырьмя независимо запускаемыми CLI-модулями (P1 photometric, P2 geometric, P3 metric, P4 experiments), — который гарантирует:

1. единый, проверяемый протокол данных (манифесты, разбиения, отсутствие утечек);
2. единый контракт входных/выходных файлов;
3. единую семантику решения с отказом `⊥` и единые формулы метрик;
4. воспроизводимость запусков (seeds, окружение, resume без перезаписи);
5. запрет на «подгонку под test» (профили, блокировка test, пороги только на validation).

Исходного кода авторских статей нет. Код реализуется с нуля по главам 1–2 диссертации; совпадение с неизвестной реализацией статей не заявляется.

### 1.2 В объёме этой спеки

- общий пакет `expcore`: конфигурация, схемы, манифесты, разбиения, проверки целостности, загрузчики, метрики, пороги, журналирование, `run.json`/`environment.json`, resume, seeds, измерение времени, реестр библиотечных примитивов;
- CLI-каркасы четырёх модулей с командами из раздела 6 ТЗ;
- реализуемые уже сейчас команды: `experiments validate-data`, `experiments build-manifests`, `experiments plan`, `experiments aggregate`, `experiments report` (на общих метриках и контракте);
- контрольные примеры (fixtures), автоматические проверки математических свойств, README, схемы файлов, журнал решений, lock-файлы.

### 1.3 Вне объёма (сознательно)

- Алгоритмы глобальных представлений (P1), локальных признаков и гомографии (P2), обучения эмбеддингов и поиска (P3). Команды `photometric run`, `geometric run`, `metric train|embed|search` в этой спеке — каркасы: полная обвязка (конфиг, манифест, seeds, run-каталог, журнал), но вместо алгоритма — явная ошибка «алгоритм не реализован» (код завершения 6). Алгоритмы подключаются через реестр (раздел 3.4).
- Пары (`pairs.parquet`) для конкретных постановок P1–P3: правила их построения задаются в ТЗ 01–03. Здесь реализованы схема, валидация и запись.
- Планы запусков, статистические процедуры, анализ ошибок и сводка опытов 3.1, 3.8–3.9 — ТЗ 04. Здесь реализованы механизмы, на которых они строятся (план с замороженными хэшами, агрегация по общим метрикам, правила статистики из ВОСП-01).
- GUI (явно исключён в ОБЩ-01).
- Скачивание датасетов. Архивы кладёт пользователь; код только описывает и проверяет их.

### 1.4 Допущения

1. Данные доступны локально как распакованные каталоги (и, желательно, исходные архивы для расчёта SHA-256).
2. SOP читается из исходной раскладки поставки (файлы `Ebay_train.txt`, `Ebay_test.txt` и каталоги изображений). Формат TFDS в версии 1 не поддерживается (решение D-14).
3. Структура остальных наборов соответствует публичным поставкам: KADID-10k (оригиналы `I01…I81`, производные `I{оригинал}_{тип}_{уровень}`), HPatches (каталоги последовательностей с префиксами `i_`/`v_`, изображения `1…6`, гомографии `H_1_2…H_1_6`), MVTec AD (`<категория>/train/good`, `<категория>/test/<тип>`, `<категория>/ground_truth/<тип>`). Парсеры проверяют структуру и при несоответствии завершаются ошибкой, а не угадывают.
4. Основной язык проекта — Python; точные версии библиотек определяются проверочной установкой (раздел 12.2), в спеке номера версий не фиксируются.

### 1.5 Критерий успеха

Приёмка подтверждает **правильность реализации и воспроизводимость**, а не превосходство метода (ПОСТ-02). Отрицательный результат сохраняется и попадает в отчёт: `experiments report` не имеет права опускать завершённые запуски плана (раздел 7.7).

### 1.6 Обязательные оговорки в отчётах

Каждый отчёт, README и сводка содержат фиксированный текст (проверяется тестом `T-REP-01`):

> Численные настройки являются назначенным проектным протоколом (редакция 1.0 от 23.10.2026), а не опубликованными результатами или доказанно оптимальными параметрами. Реализация выполнена с нуля по главам 1 и 2; совпадение с неизвестной реализацией авторских статей не заявляется.

Если HPatches неполон — везде вместо «HPatches» указывается «HPatches, поднабор `<имя>`» (раздел 4.3).

---

## 2. Журнал проектных решений

Решения, принятые там, где ТЗ допускает несколько прочтений. Они же станут первыми записями `DECISIONS.md`.

| ID | Решение | Причина |
|---|---|---|
| D-01 | Один репозиторий, src-layout, 5 пакетов: общий `expcore` + `photometric`, `geometric`, `metric`, `experiments`. Общий пакет назван `expcore`, а не `common` | ОБЩ-01; общее имя `common` часто конфликтует с другими пакетами в окружении |
| D-02 | Все разбиения используют `np.random.Generator(np.random.PCG64(2026))`, **новый генератор на каждую независимую операцию** (KADID; HPatches-`i_`; HPatches-`v_`; каждая категория MVTec; SOP) | ТЗ называет PCG64 явно для MVTec/SOP, для KADID/HPatches — только `seed=2026`; единый генератор исключает неоднозначность, а свежий генератор на операцию делает результат независимым от порядка обработки наборов |
| D-03 | Сортировка ID: строки сравниваются по кодовым точкам Unicode; `class_id` SOP — как целые числа | Порядок сортировки определяет результат перестановки; его нужно зафиксировать |
| D-04 | Границы долей — только целочисленная арифметика: `ceil(20%·n) = (20·n + 99)//100`, `floor(0.25·N) = N//4`, `floor(0.8·M) = (4·M)//5` | Исключает ошибки округления float |
| D-05 | `image_id` — непрозрачный: `"{dataset_id}:{ordinal:06d}"`, порядковый номер по отсортированному `relative_path`. Естественный ключ остаётся в `relative_path` | Имена файлов и каталогов MVTec содержат метку (`good`/тип дефекта); алгоритм не должен видеть метку через идентификатор или путь |
| D-06 | `source_group_id`: KADID — ID оригинала; HPatches — имя последовательности; MVTec — `{категория}/{stem}` исходного изображения; SOP — `class_id` | Единица, внутри которой наблюдения не независимы (ДАН-02, ВОСП-01) |
| D-07 | `dataset_manifest_hash` — SHA-256 от канонической сериализации содержимого таблиц (строки по ключу, колонки в порядке схемы, значения в каноническом JSON), а не от байтов parquet | Байты parquet зависят от версии pyarrow и метаданных записи |
| D-08 | `run_id` детерминирован: `{module}-{profile}-{config_hash[:12]}-s{seed}` | Повторный запуск той же конфигурации с тем же seed — это тот же запуск, поэтому resume работает без дополнительного состояния |
| D-09 | Секция конфигурации `paths` исключена из `config_hash` | Перенос данных на другую машину не должен менять идентичность опыта; идентичность данных обеспечивает `dataset_manifest_hash` |
| D-10 | В `predictions.parquet` добавлена колонка `outcome ∈ {decided, reject, not_applicable}`. Технические ошибки в `predictions` не попадают | ДАН-05: `reject`, `not_applicable`, `null` и ошибка запуска — разные вещи |
| D-11 | Добавлена команда `experiments build-manifests`; `validate-data` только проверяет и не создаёт манифест | Построение манифеста изменяет состояние, проверка — нет; ТЗ перечисляет обязательные команды, но не запрещает дополнительные |
| D-12 | `aggregate` завершается ошибкой, если в `predictions` нет строки для какой-либо пары оцениваемого набора; отсутствие строки не трактуется как отказ | ДАН-05, формула Coverage |
| D-13 | Зависимости сверх списка ОБЩ-02: `pyarrow` (parquet), `PyYAML` (config.yaml), `pytest` (проверки); `psutil` — опционально (пиковая память) | Без них контракт данных не реализуется |
| D-14 | SOP читается из исходной раскладки поставки; TFDS не поддерживается | Исходная раскладка содержит `class_id` и официальное разбиение; поддержка TFDS — отдельное расширение |
| D-15 | Правило порога: `unified_score ≥ threshold` ⇒ `y=1` (совпадение на границе — соответствие). Правило задаётся в конфиге явно (`tie_rule: ">="`) | Граничный случай должен быть однозначным |
| D-16 | Блокировка test реализована через реестр версий протокола и файл `test_opened.json` (раздел 7.5) | ВОСП-04 |
| D-17 | В `pairs.parquet` добавлена необязательная колонка `input_mask_path` — маски, доступные алгоритму по постановке; `gt_mask_path` остаётся только оценочной | ДАН-04 требует различать входные маски и эталонные |
| D-18 | Номера версий библиотек в спеке не фиксируются | Требуется проверочная установка (ОБЩ-02) |
| D-19 | Код платформенно-нейтрален (Linux и Windows): `pathlib`, без жёстко заданных разделителей и абсолютных путей | Раздел 6 ТЗ: код не содержит абсолютных путей разработчика |
| D-20 | Видимость колонок пары для алгоритма задаётся белым списком в конфиге (`inputs.pair_columns`); `y_match`, `gt_H_path`, `gt_mask_path` в нём запрещены | `transform_id` и `severity` могут в некоторых постановках определять метку; решение остаётся за ТЗ программы, но запрет на метку и GT абсолютен |

---

## 3. Архитектура

### 3.1 Структура репозитория

```
repo/
  pyproject.toml              # extras: [metric] = TensorFlow/Keras; [dev] = pytest
  requirements.lock           # CPU-профиль (P1/P2/P4) — после проверочной установки
  requirements-metric.lock    # + TensorFlow/Keras для P3
  README.md  DECISIONS.md
  configs/
    protocol_registry.json    # версии протокола (раздел 7.5)
    templates/                # полные шаблоны config.yaml без скрытых defaults
    smoke/                    # конфиги smoke-профиля каждого модуля
  schemas/                    # JSON Schema файлов контракта (генерируются из кода)
  scripts/                    # служебные скрипты (генерация схем, проверка lock)
  src/
    expcore/
      errors.py  config.py  schemas.py  io.py  hashing.py
      manifests.py  integrity.py  loaders.py  decisions.py
      metrics.py  thresholds.py  seeding.py  runlog.py
      resume.py  timing.py  primitives.py  registry.py
      reporting.py  cli_common.py
      splits/  kadid.py  hpatches.py  mvtec.py  sop.py  rng.py
    photometric/  __main__.py  cli.py
    geometric/    __main__.py  cli.py
    metric/       __main__.py  cli.py      # TensorFlow импортируется лениво
    experiments/  __main__.py  cli.py  plan.py  aggregate.py  report.py
  tests/
    unit/  smoke/  fixtures_builder.py  fakes.py
```

Фиксированный набор тестовых изображений в репозитории **не хранится**: fixtures строятся кодом из seed (`tests/fixtures_builder.py`). Реальные изображения датасетов в поставку не входят (ДАН-01).

### 3.2 Границы модулей

- `expcore` ничего не знает об алгоритмах P1–P3: он отдаёт загрузчики, схемы, метрики, пороги, журналирование, раннер.
- `photometric`, `geometric`, `metric`, `experiments` зависят только от `expcore`; друг от друга — нет.
- `metric` — единственный, кто импортирует TensorFlow; импорт ленивый, поэтому CPU-профиль P1/P2/P4 работает без TensorFlow.
- Каждая единица отвечает на три вопроса: что делает, как её использовать, от чего зависит (разделы 3.3–3.4).

### 3.3 Единицы `expcore`

| Модуль | Назначение | Зависит от |
|---|---|---|
| `errors` | Иерархия исключений и коды завершения (раздел 8.3) | — |
| `io` | Строгое чтение/запись JSON (`allow_nan=False`), атомарная запись файлов, чтение/запись parquet со схемой | `errors` |
| `hashing` | SHA-256 файла (потоково, блоки 1 МиБ), каноническая сериализация, `table_content_hash` | `io` |
| `config` | Строгий загрузчик `config.yaml` в dataclass-схемы; `config_hash`; реестр схем секций модулей | `io`, `hashing` |
| `schemas` | Схемы `datasets.json`, `images`, `pairs`, `predictions`, `run.json`; проверка инвариантов INV-01…INV-11 | `io` |
| `splits/*` | Правила разбиения четырёх наборов; чистые функции от списка ключей | `rng` |
| `manifests` | Построение (`build`) и загрузка (`load`) манифестов; заморозка; проверка хэша | `splits`, `integrity`, `schemas` |
| `integrity` | Пересечения SHA-256 и `source_group_id` между частями; отчёт о дубликатах | `hashing` |
| `loaders` | `ImageStore`, представления пар `inputs`/`evaluation`, охранник доступа к split | `manifests` |
| `decisions` | `Outcome`, `Decision`, единое направление оценок, правило порога | — |
| `metrics` | Формулы (О-3, Г2-168), матрица 2×3, условные метрики, смена положительного класса | `decisions` |
| `thresholds` | Выбор порога только на validation; `thresholds.json` | `metrics`, `loaders` |
| `seeding` | Seeds, потоки `SeedSequence`, сохранение random-state | — |
| `runlog` | `run.json`, `environment.json`, журнал `run.log.jsonl` | `io`, `hashing` |
| `resume` | Единицы работы, shard-запись, `progress.json`, проверка целостности завершённого | `runlog`, `io` |
| `timing` | Измерение `elapsed_ms`, таймеры стадий, пиковая память | — |
| `primitives` | Реестр названных библиотечных примитивов и их контрольных проверок | — |
| `registry` | Реестр алгоритмов, кодов причин, отношений `relation_id` | `decisions` |
| `reporting` | Фиксированные оговорки, шаблоны таблиц отчёта | `metrics` |
| `cli_common` | Общие аргументы, перехват исключений → коды завершения | `errors`, `config` |

### 3.4 Интерфейс подключения алгоритмов

Алгоритм — объект, зарегистрированный в `registry` под парой `(module, method_id)`:

```python
class PairPredictor(Protocol):
    method_id: str
    def predict(self, pair: PairInput, images: ImageStore, ctx: RunContext) -> Decision: ...

class QueryPredictor(Protocol):      # P3: запросы к галерее
    method_id: str
    def predict(self, queries: Sequence[QueryInput], images: ImageStore,
                ctx: RunContext) -> Sequence[Decision]: ...

@dataclass(frozen=True)
class Decision:
    outcome: Outcome                    # decided | reject | not_applicable
    y_pred: int | None                  # 1/0 при decided, иначе None
    score_raw: float | None             # конечное число или None
    score_direction: ScoreDirection     # higher_is_match | lower_is_match
    reason_code: str | None             # из реестра; обязателен, если outcome != decided или score_raw is None
    components: Mapping[str, float]     # исходные компоненты решения (колонки comp_*)
```

Раннер (`resume.py` + `timing.py`) оборачивает вызов: измеряет `elapsed_ms`, пишет shard, ведёт `progress.json`. Алгоритм получает **только** `PairInput` и `ImageStore` — ни путей, ни меток, ни GT (раздел 5.3).

### 3.5 Поток данных

```
архив + лицензия ─► build-manifests ─► манифесты (заморожены, хэш)
                                             │
                    validate-data ◄──────────┤   (проверка целостности, дубликатов, состава)
                                             ▼
config.yaml ─► модуль run ─► раннер ─► predictions/*.parquet ─► thresholds (validation)
                                             │
                                             ▼
                          experiments aggregate ─► metrics.parquet ─► experiments report
```

---

## 4. Датасеты и разбиения

### 4.1 Общие правила

1. Единица разбиения — группа `source_group_id`. Все наблюдения группы попадают в одну часть.
2. Алгоритм разбиения — чистая функция `split_*(keys, seed) → {part: [keys]}`:
   ```python
   def shuffled(keys: Sequence[K], seed: int = 2026) -> list[K]:
       ordered = sorted(keys)                                   # D-03
       rng = np.random.Generator(np.random.PCG64(seed))         # D-02
       return [ordered[i] for i in rng.permutation(len(ordered))]
   ```
3. Границы долей считаются по D-04. Пустая часть — ошибка (`InsufficientDataError`); молчаливый перенос объектов между частями запрещён.
4. Манифест — первичный артефакт. Он сохраняется и замораживается **до** расчёта результатов; код разбиения повторно используется только для проверки (`validate-data` пересчитывает разбиение и сравнивает с сохранённым манифестом; расхождение — ошибка, а не повод молча пересохранить). Поскольку поток `Generator` не гарантируется стабильным между версиями NumPy, версия NumPy зафиксирована lock-файлом, а в тестах есть эталонные перестановки (`T-SPL-07`).
5. Состав набора сверяется с ожидаемым (`expected` в конфиге). Отклонение — `CountMismatchError`, кроме HPatches (раздел 4.3).

### 4.2 KADID-10k

| Параметр | Значение |
|---|---|
| Ожидаемый состав | 81 оригинал; 10 125 производных = 81 × 25 × 5 (на каждый оригинал ровно 125) |
| Группа | ID оригинала |
| Операция | `ids = sorted(оригиналы)`; `shuffled(ids)`; первые 20 → `validation`, остальные 61 → `test` |
| Производные | Следуют за оригиналом в ту же часть |
| `category_id` | Код типа искажения (`01…25`) у производных; `null` у оригиналов |
| Строк в `images` | 81 + 10 125 = 10 206 |

### 4.3 HPatches

| Параметр | Значение |
|---|---|
| Состав | Полные последовательности: эталон, пять наблюдений, пять гомографий `H_1_2…H_1_6` |
| Группа | Имя последовательности |
| Операция | Отдельно для префиксов `i_` и `v_`: `sorted(имена)`; `shuffled`; первые `ceil(20%·n)` → `validation`, остальные → `test` |
| `category_id` | `i` или `v` |
| Ошибка | Пустая часть (например, `n = 1`) — `InsufficientDataError` |

**Отчёт о полноте (ДАН-03).** `build-manifests` пишет `completeness_report.json`: список всех последовательностей, наличие всех 6 изображений и 5 гомографий в каждой, итоговый статус:

- `full` — все последовательности полны и число последовательностей равно `expected_sequences` из конфига;
- `subset` — поставка сокращённая; точный список сохраняется, имя поднабора (`subset_name`) обязательно задаётся в конфиге, и во всех отчётах набор называется «HPatches, поднабор `<имя>`»;
- `unverified` — `expected_sequences` в конфиге не задан; набор можно использовать, но отчёт обязан это указать.

Гомографии (`gt_H_path`) — только в оценочной части (раздел 5.3).

### 4.4 MVTec AD

| Параметр | Значение |
|---|---|
| Категории | `carpet`, `leather`, `tile`, `wood` (в конфиге; остальные 11 в манифест не попадают) |
| Группа | `{категория}/{stem}` исходного изображения |
| Операция (для каждой категории отдельно) | `N` — число нормальных изображений официального train; `shuffled`; первые `N//4` → `dev`; следующие `N//4` → `validation`; остаток `N − 2·(N//4)` → `synthetic_test` |
| Официальный test | Целиком → `real_test` (нормальные и дефектные, с масками) |
| Ошибка | `N < 4` (пустая часть) — `InsufficientDataError` |
| Лицензия | CC BY-NC-SA 4.0; изображения не включаются в поставку кода |

Метка дефекта и тип дефекта — оценочные колонки (`defect_type`, `gt_mask_path`); в представление `inputs` не попадают.

### 4.5 Stanford Online Products

| Параметр | Значение |
|---|---|
| Ожидаемый состав | 22 634 ID товаров; 59 551 train и 60 502 test изображений |
| Группа и класс | `class_id` (ID товара) |
| Операция | `M` — число `class_id` в официальном train; `sorted` как целые; `shuffled`; первые `(4·M)//5` → `train`, остаток → `validation` |
| Официальный test | Неизменен → `test` |
| Проверка | `class_id` официального train и test не пересекаются (иначе `LeakageError`); сумма классов равна 22 634 |
| `category_id` | `super_class_id` исходной поставки |

Разделение — по `class_id`, а не по отдельным изображениям.

### 4.6 Целостность и дубликаты (ДАН-02)

`integrity.check(manifest)` выполняет:

1. SHA-256 каждого изображения (потоково);
2. пересечения SHA-256 между **частями** одного набора;
3. пересечения `source_group_id` между частями;
4. совпадение разбиения групп: все изображения одной группы лежат в одной части.

Результат — `duplicates_report.json` с перечислением каждого найденного пересечения (хэш, `image_id`, части, группы) и полем `blocking`. Правила:

- межчастевое пересечение → `blocking: true`; `validate-data` завершается с кодом 3, а любой `run` отказывается стартовать на такой версии манифеста (код 3);
- исправление — **новая** версия манифеста: `build-manifests` с файлом `overrides.yaml` (операция `exclude_image: [{sha256, reason}]`, применяется до разбиения); старая версия остаётся нетронутой, перезапись версии запрещена;
- дубликаты внутри одной части не блокируют, но перечисляются (`blocking: false`).

### 4.7 Происхождение и лицензии (ДАН-01)

`datasets.json` хранит на каждый набор: `source_url`, `download_date`, `archive_name`, `archive_sha256`, `license` и `license_file` (путь + SHA-256), `discovered_counts`, `manifest_version`. Если исходный архив недоступен (есть только распакованный каталог), `archive_sha256` берётся из конфига (записан при скачивании), а поле `archive_verified` получает `false`. `discovered_counts` — фактические числа: изображения, группы, классы, число по частям и категориям, плюс результат сверки с `expected`.

### 4.8 `build-manifests` и `validate-data`

`experiments build-manifests --config …` (добавлена, D-11):

1. читает конфиг и каталоги данных;
2. строит `datasets.json`, `images.parquet`, затем `pairs.parquet` для программ, которые зарегистрировали построители пар;
3. выполняет разбиение (4.2–4.5) и проверки (4.6);
4. записывает `manifest.frozen.json` (версия, время, хэш) и **только после этого** считает манифест замороженным.

Существующую версию он не перезаписывает.

`experiments validate-data --config …`:

1. находит замороженный манифест; отсутствует → код 3 с подсказкой запустить `build-manifests`;
2. проверяет хэш содержимого таблиц против `manifest.frozen.json`;
3. проверяет наличие файлов и SHA-256 изображений, схемы и инварианты INV-01…INV-04;
4. пересчитывает разбиение и сравнивает с сохранённым;
5. пересчитывает межчастевые пересечения;
6. пишет `validation_report.json` и возвращает 0 только если всё чисто.

Раскладка каталога манифеста:

```
<manifest_dir>/<manifest_version>/
  datasets.json  images.parquet  manifest.frozen.json
  completeness_report.json  duplicates_report.json  exclusions.json
  p1/pairs.parquet  p2/pairs.parquet  p3/pairs.parquet
```

---

## 5. Контракт данных

Типы: `str` (utf-8), `int32`/`int64`, `float64`, `bool`, `ts` (ISO 8601, UTC, суффикс `Z`). «Null» — допускается ли отсутствие значения.

### 5.1 `datasets.json`

JSON-массив записей, по одной на набор.

| Поле | Тип | Null | Описание |
|---|---|---|---|
| `dataset_id` | str | нет | `kadid10k` \| `hpatches` \| `mvtec_ad` \| `sop` |
| `source_url` | str | нет | URL поставщика |
| `archive_sha256` | str | нет | SHA-256 основного архива (64 hex, нижний регистр) |
| `license` | str | нет | Название лицензии |
| `discovered_counts` | object | нет | Фактические количества (4.7) |
| `manifest_version` | str | нет | Версия манифеста |
| `download_date` | ts | нет | Расширение (ДАН-01) |
| `archive_name` | str | нет | Расширение (ДАН-01) |
| `archive_verified` | bool | нет | Расширение: хэш вычислен по файлу архива |
| `license_file` | object | нет | Расширение: `{path, sha256}` |
| `extra_archives` | array | нет | Расширение: `[{name, sha256}]`, возможно пустой |
| `completeness` | str | да | Только HPatches: `full` \| `subset` \| `unverified`; `subset_name` при `subset` |

### 5.2 `images.parquet`

| Поле | Тип | Null | Описание |
|---|---|---|---|
| `image_id` | str | нет | Непрозрачный ID (D-05), уникален |
| `dataset_id` | str | нет | Перечисление |
| `source_group_id` | str | нет | D-06 |
| `class_id` | str | да | SOP — ID товара; в остальных наборах `null` |
| `category_id` | str | да | 4.2–4.5 |
| `split` | str | нет | `train`\|`validation`\|`test`\|`dev`\|`synthetic_test`\|`real_test` (допустимое множество зависит от набора) |
| `relative_path` | str | нет | Путь относительно `data_root`; разделитель `/` |
| `sha256` | str | нет | 64 hex |
| `width`, `height`, `channels` | int32 | нет | > 0 |
| `defect_type` | str | да | Расширение, **оценочная** колонка (MVTec) |

### 5.3 `pairs.parquet` и видимость колонок

| Поле | Тип | Null | Описание |
|---|---|---|---|
| `pair_id` | str | нет | Уникален |
| `query_id`, `reference_id` | str | нет | Внешние ключи на `images.image_id` |
| `relation_id` | str | нет | `P1_SAME_SCENE` \| `P2_NO_CHANGE_IN_REGION` \| `P3_SAME_PRODUCT` |
| `y_match` | int8 | нет | 1 — соответствующая пара, 0 — несоответствующая (О-1); **оценочная** |
| `split` | str | нет | Часть пары |
| `source_group_id` | str | нет | Группа запроса |
| `transform_id` | str | да | Тип преобразования |
| `severity` | float64 | да | Степень |
| `seed` | int64 | да | Seed построения пары |
| `gt_H_path` | str | да | **Оценочная** колонка |
| `gt_mask_path` | str | да | **Оценочная** колонка |
| `input_mask_path` | str | да | Расширение (D-17): маска, доступная по постановке |

Смысл `y_match` зависит от `relation_id`: P1 — один исходный сюжет в допустимом режиме; P2 — отсутствие обнаруживаемого изменения в заданной области; P3 — один ID товара. Метки разных `relation_id` в одну accuracy не объединяются: `metrics` бросает `MixedRelationError` при смешении.

**Видимость (ДАН-04).** `loaders` отдаёт два представления:

- `inputs` — только колонки белого списка `inputs.pair_columns` из конфига (всегда есть `pair_id`, `query_id`, `reference_id`, `relation_id`; опционально `transform_id`, `severity`, `seed`, `input_mask_path`). Попытка включить `y_match`, `gt_H_path`, `gt_mask_path` — `ConfigError`.
- `evaluation` — `y_match`, `gt_H_path`, `gt_mask_path`, `defect_type`; доступно только `thresholds`, `metrics`, `aggregate` и режиму `oracle`.

Алгоритм получает изображения через `ImageStore.load(image_id, color="bgr"|"rgb"|"gray")` (цветовой порядок задаётся явно, без default); путей и имён файлов он не видит.

**Контрольный тест ДАН-04** (`T-DAT-05`): запуск с физически удалёнными колонками `gt_H_path`, `gt_mask_path` (и без файлов GT) даёт побайтно те же предсказания. Режим `oracle` — только явный: `oracle.enabled: true` + `oracle.reason` в конфиге; он помечается в `run.json` (`oracle_mode: true`) и в `method_id` (суффикс `+oracle`), а `aggregate` выносит такие запуски в отдельную секцию.

### 5.4 `predictions.parquet`

| Поле | Тип | Null | Описание |
|---|---|---|---|
| `run_id` | str | нет | |
| `pair_id` / `query_id` | str | да | Ровно одно из двух непусто |
| `method_id` | str | нет | |
| `score_raw` | float64 | да | Конечное число или `null` (никаких NaN/Inf) |
| `score_direction` | str | нет | `higher_is_match` \| `lower_is_match` — направление **исходного** значения |
| `y_pred` | int8 | да | 1/0 при `decided`, иначе `null` |
| `q_valid` | bool | нет | `true` ⇔ `outcome == decided` (в формуле — `q_cmp`) |
| `reason_code` | str | да | Из реестра; обязателен при `reject`, `not_applicable`, `score_raw is null` |
| `elapsed_ms` | float64 | да | `null` только вместе с кодом причины |
| `outcome` | str | нет | Расширение (D-10): `decided` \| `reject` \| `not_applicable` |
| `timing_mode` | str | нет | Расширение: `per_item` \| `amortized_batch` |
| `comp_*` | float64 | да | Исходные компоненты решения (P2 с несколькими порогами) |

Единое направление: `score_match = score_raw` при `higher_is_match`, `−score_raw` при `lower_is_match`. Правило преобразования (`identity`/`negate`) определяется колонкой `score_direction` и записывается в `metrics`. Вероятность из компонентов не выдумывается.

### 5.5 `run.json`

| Поле | Тип | Null | Описание |
|---|---|---|---|
| `schema_version` | str | нет | |
| `run_id` | str | нет | D-08 |
| `config_hash` | str | нет | SHA-256 канонического JSON конфигурации без секции `paths` |
| `git_commit` | str | да | `null` допустим только в smoke; confirmatory требует коммит |
| `dirty_diff_hash` | str | да | SHA-256 от `git diff HEAD` + содержимого неотслеживаемых файлов; `null`, если дерево чистое |
| `dataset_manifest_hash` | str | нет | D-07 |
| `seed` | int | нет | |
| `environment` | object | нет | `{path, sha256, python, os, cpu, gpu}` — ссылка на `environment.json` и краткая сводка |
| `status` | str | нет | `running` \| `completed` \| `failed` |
| `started_at`, `finished_at` | ts | `finished_at`: да | `null`, пока запуск не завершён |
| `profile`, `module`, `command`, `protocol_version` | str | нет | Расширения |
| `oracle_mode` | bool | нет | Расширение |
| `post_test_study` | bool | нет | Расширение (раздел 7.5) |
| `library_primitives` | array | нет | Расширение: список `{name, library, version, check_id}` |
| `attempts` | array | нет | Расширение: история попыток (старт, финиш, статус, причина отказа) |
| `failure` | object | да | `{type, message, unit_id}` при `failed` |

### 5.6 `config.yaml`

Строгий загрузчик: **все** ключи обязательны, неизвестные ключи запрещены, скрытых defaults нет. Секции:

| Секция | Содержимое |
|---|---|
| `schema_version`, `protocol_version`, `profile` | `smoke`\|`pilot`\|`confirmatory` |
| `seeds` | `seed` (для run) / `seeds` (для plan) из допустимого множества профиля |
| `datasets` | `dataset_id`, `manifest_version`, `categories` (MVTec), `expected`, `provenance` |
| `inputs` | `pair_columns` (белый список), `color_order` |
| `data_modes` | `oracle: {enabled, reason}` и прочие режимы данных |
| `algorithms` | `{method_id: версия алгоритма}` |
| `params` | Параметры модуля; схема регистрируется модулем (`register_params_schema`); незарегистрированная схема — `ConfigError` |
| `decision` | `tie_rule`, правило порога и критерий выбора (из ТЗ программы) |
| `determinism` | Флаги детерминизма, число потоков OpenCV/NumPy |
| `runner` | `chunk_size` — число элементов в единице работы (7.6) |
| `logging` | Уровень, каталог |
| `paths` | `data_root`, `manifest_dir`, `out_root` — **исключена** из `config_hash` (D-09); переопределяется аргументами CLI |

`config_hash = SHA-256(canonical_json(config без paths))`, где `canonical_json` — ключи по алфавиту, без пробелов, UTF-8. Итоговый разрешённый конфиг сохраняется в каталоге запуска как `config.resolved.yaml`.

### 5.7 Различение `null`, `not_applicable`, `reject` (ДАН-05)

| Понятие | Представление | Учёт |
|---|---|---|
| Алгоритмический отказ `⊥` | `outcome=reject`, `y_pred=null`, `q_valid=false`, `reason_code=REJECT_*` | Входит в `Q` и в `N±`; снижает покрытие |
| Вопрос не ставится | `outcome=not_applicable`, `reason_code=NOT_APPLICABLE_*` | Исключается из `Q` и из метрик; число таких пар выводится отдельно |
| Значение отсутствует | `null` + `reason_code` (никогда `0`) | Нулевая оценка различия — не отказ |
| Техническая ошибка | Не пишется в `predictions`; фиксируется в `run.json` (`failed`, `failure`) и `errors.jsonl` | Запуск не считается завершённым; `aggregate` не принимает его |

JSON не содержит NaN/Infinity: запись идёт через `json.dumps(..., allow_nan=False)`; parquet-колонки с плавающей точкой проверяются на NaN/Inf до записи.

### 5.8 Инварианты контракта

| ID | Инвариант |
|---|---|
| INV-01 | `image_id` уникален; `sha256` — 64 hex нижнего регистра; `width`, `height`, `channels` > 0; `split` допустим для набора |
| INV-02 | Для каждой пары `query_id` и `reference_id` существуют; `split(query) = split(reference) = pair.split`; межчастевые пары запрещены |
| INV-03 | `pair_id` уникален; `relation_id` зарегистрирован; `y_match ∈ {0,1}` |
| INV-04 | Все изображения одного `source_group_id` лежат в одной части |
| INV-05 | `outcome=decided` ⇒ `y_pred ∈ {0,1}`, `q_valid=true`; `reject` ⇒ `y_pred=null`, `q_valid=false`, код `REJECT_*`; `not_applicable` ⇒ `y_pred=null`, `q_valid=false`, код `NOT_APPLICABLE_*` |
| INV-06 | `score_raw` конечен или `null`; при `null` задан `reason_code` |
| INV-07 | Для каждой пары `(run_id, method_id)` множество ключей `predictions` равно множеству ключей оцениваемого набора (без дублей, без пропусков) |
| INV-08 | Ни в JSON, ни в parquet нет NaN/Inf |
| INV-09 | Технические ошибки отсутствуют в `predictions` |
| INV-10 | Порог получен только на `split=validation` |
| INV-11 | Колонки `y_match`, `gt_H_path`, `gt_mask_path`, `defect_type` не доступны через `inputs` |

---

## 6. Семантика решения и метрики

### 6.1 Решение

`y ∈ {0, 1, ⊥}`, `q = 1[y ≠ ⊥]` (О-2). Выход `⊥` — отказ; нулевая оценка различия не является кодом отказа. Метки разных постановок (`relation_id`) не объединяются. Допускается отдельная метка `y_defect = 1 − y_match`; тогда `positive_class="defect"`, оценки инвертируются (`score_defect = −score_match`), а смысл положительного класса и инверсия указываются в **каждой** таблице и **каждом** графике (подпись/метаданные; проверяется `T-MET-09`).

### 6.2 Формулы

Пусть `Q` — все пары оцениваемого набора и отношения (без `not_applicable`), `N+` и `N−` — число пар с `y_match=1` и `y_match=0`. Счётчики `TP, FP, TN, FN` считаются **только по принятым решениям**; пары-отказы входят в `N±` своего истинного класса.

```
Coverage   = (1/|Q|) · Σ_{q∈Q} 1[q_cmp,q = 1]        # Г2-168
rho_rej    = 1 − Coverage
Coverage+  = (принятые среди y_match=1) / N+
Coverage−  = (принятые среди y_match=0) / N−

Recall_all  = TP / N+                                  # О-3
FPR_all     = FP / N−
Correct_all = (TP + TN) / (N+ + N−)
```

Условные метрики (только по принятым; подписываются как «conditional» и **никогда** не заменяют показатели `*_all`):

```
Recall_cond = TP/(TP+FN)    FPR_cond = FP/(FP+TN)    Correct_cond = (TP+TN)/(TP+TN+FP+FN)
```

Матрица 2×3 — строки: истинный класс `{y_match=1, y_match=0}`; столбцы: `{y_pred=1, y_pred=0, reject}`; выводятся числа и доли по строкам. P4 обязан показать её в отчёте.

Если знаменатель равен нулю (`N+=0`, `N−=0`, `TP+FN=0` …), метрика равна `null` с причиной `undefined_empty_class` / `undefined_no_accepted` — не нулём.

Свойства, которые обязаны выполняться и проверяются (`T-MET-*`):

- `0 ≤ Coverage ≤ 1`; `rho_rej = 1 − Coverage`;
- `Recall_all = Recall_cond · Coverage+`; `FPR_all = FPR_cond · Coverage−`; `Correct_all = Correct_cond · Coverage`;
- строки матрицы 2×3 суммируются в `N+` и `N−`;
- результат не зависит от порядка строк;
- при смене положительного класса (`y_defect = 1 − y_match`, `y_pred_defect = 1 − y_pred`, отказы остаются отказами) `N+ ↔ N−`, `TP ↔ TN`, `FP ↔ FN`; `Correct_all` не меняется, `Recall_all(defect) = TN / N−`, `FPR_all(defect) = FN / N+` — проверяется на ручных примерах.

### 6.3 Выход `metrics.parquet`

`run_id, method_id, relation_id, split, positive_class, score_transform, n_pairs, n_source_groups, n_seeds, n_not_applicable, n_pos, n_neg, tp, fp, tn, fn, rej_pos, rej_neg, coverage, coverage_pos, coverage_neg, rho_rej, recall_all, fpr_all, correct_all, recall_cond, fpr_cond, correct_cond, null_reasons`.

### 6.4 Пороги

- Пороги выбираются **только** на `split=validation` (INV-10). `select_threshold(...)` принимает только строки validation; иначе `ThresholdLeakageError`. Критерий выбора — вызываемый объект, заданный ТЗ программы.
- Результат пишется в `thresholds.json`: `run_id`, `method_id`, `criterion_id`, значения порога, `split_used`, `n_validation_pairs`, `manifest_hash`, время.
- Оценка на test читает `thresholds.json` в режиме «только чтение»; без файла или с `split_used != validation` — `ProtocolViolationError`.
- Для MVTec `dev` используется для разработки и подбора в pilot, а пороги выбираются на `validation`.

---

## 7. Воспроизводимость и этапность

### 7.1 Seeds (ВОСП-01)

| Назначение | Seeds |
|---|---|
| Разбиение датасетов | `2026` (отдельно от seeds запусков) |
| Подбор параметров (pilot, подбор в confirmatory-подготовке) | `11, 23, 37` |
| Подтверждающие запуски | `11, 23, 37, 51, 71` |
| Smoke | `11` |

Один `run` выполняется с одним `seed`; `experiments plan` разворачивает матрицу запусков по seeds. Повторы на тех же изображениях оценивают изменчивость алгоритма/обучения, но **не** увеличивают независимую выборку в пять раз. Следствия, которые закрепляются в `aggregate`:

- каждая строка итоговой таблицы содержит `n_pairs`, `n_source_groups`, `n_seeds`;
- разброс по seeds отчитывается отдельно («изменчивость по seeds»); доверительные интервалы, если они считаются, строятся по `source_group_id` (кластерное, а не по парам), а seeds не объединяются в одну выборку. Конкретные процедуры — ТЗ 04.

### 7.2 Потоки случайности (ВОСП-02)

`seeding.init(seed)`:

1. `random.seed(seed)`; `np.random.seed(seed)` (для библиотек с глобальным состоянием); TensorFlow — `tf.keras.utils.set_random_seed(seed)` или эквивалент из залоченной версии; OpenCV — `cv2.setRNGSeed(seed)` и фиксированное число потоков;
2. именованные независимые потоки: `stream(seed, name, unit_id) = np.random.Generator(np.random.PCG64(SeedSequence(seed, spawn_key=(STREAM_INDEX[name], unit_id))))`, где `STREAM_INDEX = {sampler: 0, augment: 1, init: 2, bootstrap: 3}`, а `unit_id` — целочисленный номер единицы работы (раздел 7.6). Поток привязан к единице работы, поэтому результат не зависит от того, какие единицы выполнены раньше (корректный resume).

Сохраняется (`rng_state.json` в начале запуска и на границах единиц работы): состояние `random`, `numpy` (глобальное и состояния генераторов), TensorFlow (seed, состояние глобального генератора, если доступно), OpenCV (seed; состояние не читается — фиксируется ограничение). Рядом: `file_order.json` (порядок файлов и пар), порядок семплеров по эпохам и `augmentations.parquet` (`unit_id, sample_id, aug_name, params_json`).

### 7.3 Детерминизм

В P3 при запуске вызывается `tf.config.experimental.enable_op_determinism()`, если функция доступна; иначе в `environment.json` записывается `deterministic_ops: false` и причина. `environment.json` содержит: версию Python, ОС, процессор, число ядер, ОЗУ, GPU/драйвер, версии всех библиотек из lock-файла, переменные `PYTHONHASHSEED`, `TF_DETERMINISTIC_OPS`, `CUDA_VISIBLE_DEVICES`, список `nondeterminism_notes` (недетерминированные операции и ограничения платформы), реестр `library_primitives`. Производительность заранее не обещается. Повторяемость проверяется на той же зафиксированной среде: тест `T-REP-02` дважды запускает smoke в свежих процессах и сравнивает хэши содержимого `predictions`.

### 7.4 Три профиля (ВОСП-03)

| Профиль | Данные | Разрешённые split | Seeds | Выходной корень | Попадает в таблицы подтверждающего эксперимента |
|---|---|---|---|---|---|
| `smoke` | Только fixtures, помеченные `fixture: true` | Любые (это синтетика) | `11` | `<out>/smoke/` | Нет |
| `pilot` | Реальные данные | `train`, `dev`, `validation` | `11, 23, 37` | `<out>/pilot/` | Нет |
| `confirmatory` | Реальные данные, зафиксированные конфигурации | Все, включая `test` по правилам 7.5 | `11, 23, 37, 51, 71` | `<out>/confirmatory/` | Да |

`loaders.SplitGuard(profile)` бросает `TestAccessError`, если профиль запрашивает недопустимый split. `aggregate` для подтверждающих таблиц читает только `confirmatory/` и отвергает запуски других профилей и запуски с `oracle_mode=true` (они уходят в отдельную секцию).

### 7.4.1 Порядок работ (ВОСП-04)

Сначала — формулы и контрольные примеры (разделы 6, 10), затем pilot с оценкой времени/памяти, затем confirmatory. Этот порядок закрепляется порядком задач в плане реализации; технически же confirmatory защищён от преждевременного старта правилами раздела 7.5 (замороженный план, версия протокола).

### 7.5 Версии протокола и блокировка test

`configs/protocol_registry.json`: список `{protocol_version, created_at, parent_version, test_opened, plan_hash}`.

- Любое изменение сеток параметров = новая `protocol_version` + новый `config_hash`; старая версия не редактируется.
- `experiments plan` (confirmatory) пишет `plan.json` (модуль, команда, seed, путь конфига, `config_hash`, `run_id`) и `plan.lock.json` (`plan_hash`, версия).
- Confirmatory-`run` проверяет, что его `config_hash` входит в `plan.lock.json`; при первом обращении к `test` создаёт `test_opened.json` и ставит `test_opened=true` в реестре.
- Если у родительской версии `test_opened=true`, дочерняя версия получает `post_test_study=true`; такие результаты в отчёте помечаются «отдельное исследование», а `plan` печатает предупреждение. Изменение плана внутри версии с открытым test — код 5.

### 7.6 Resume без перезаписи (ПОСТ-01)

Единица работы — чанк пар/запросов с целочисленным `unit_id` (порядковый номер чанка в отсортированном по ключу списке; размер чанка задаётся в конфиге). Раннер:

1. пишет результаты в shard `predictions/part-{unit_id}.parquet` через временный файл и атомарное переименование;
2. ведёт `progress.json` (`unit_id`, число строк, SHA-256 содержимого shard);
3. при повторном запуске пропускает единицы, у которых shard существует и его хэш совпадает с записанным;
4. при несовпадении хэша завершённого результата — `ResultTamperedError` (код 5); завершённое **не** перезаписывается и флага `--force` нет — пересчёт требует ручного удаления каталога;
5. изменение конфигурации даёт новый `config_hash`, а значит новый `run_id` и новый каталог; коллизий нет;
6. статус `failed` продолжается тем же `run_id`: `attempts` пополняется, `status → running → completed`.

Завершение фиксируется маркером `_SUCCESS` и `status=completed` в `run.json`.

Раскладка каталога запуска:

```
<out_root>/<profile>/<module>/<run_id>/
  run.json  config.resolved.yaml  environment.json  rng_state.json  file_order.json
  predictions/part-*.parquet  progress.json  thresholds.json
  run.log.jsonl  errors.jsonl  timers.json  augmentations.parquet  _SUCCESS
```

### 7.7 `aggregate` и `report`

- `aggregate` читает `predictions` и оценочное представление пар, проверяет INV-07 (иначе код 3, D-12), считает метрики раздела 6 по `(method_id, relation_id, split, seed)` и пишет `metrics.parquet`.
- `report` строит отчёт (Markdown) со всеми завершёнными запусками плана: метрики, матрицы 2×3, покрытие по классам, оговорки раздела 1.6. Если какой-либо завершённый запуск плана отсутствует в отчёте — `report` завершается ошибкой; порядок и состав не зависят от того, лучше или хуже результат у предложенного метода (ПОСТ-02).

---

## 8. Интерфейс командной строки

### 8.1 Команды

Команды из раздела 6 ТЗ (обязательные) и одна добавленная (D-11):

| Команда | Состояние в этой спеке |
|---|---|
| `python -m experiments validate-data --config …` | Реализуется полностью (4.8) |
| `python -m experiments build-manifests --config …` | Реализуется полностью (добавленная) |
| `python -m experiments plan --config …` | Реализуется: матрица запусков, хэши, `plan.lock.json` |
| `python -m experiments run --config …` | Реализуется: выполняет план, вызывая `run`/`train|embed|search` модулей |
| `python -m experiments aggregate --config …` | Реализуется |
| `python -m experiments report --config …` | Реализуется |
| `python -m photometric run --config …` | Каркас: полная обвязка, алгоритм — через реестр; без алгоритма код 6 |
| `python -m geometric run --config …` | То же |
| `python -m metric train|embed|search --config …` | То же; TensorFlow загружается лениво |

### 8.2 Общие аргументы

`--config PATH` (обязателен) и необязательные переопределения путей `--data-root`, `--manifest-dir`, `--out-root`. Профиль, seed и параметры — только в конфиге (никаких скрытых аргументов). Фактические пути записываются в `run.json` (`paths_resolved`) и не входят в `config_hash`. Любая команда поддерживает `--help` с кодом 0.

### 8.3 Коды завершения и исключения

| Код | Исключение | Когда |
|---|---|---|
| 0 | — | Успех |
| 2 | `ConfigError` | Ошибка конфигурации/аргументов (включая запрещённую колонку в `inputs.pair_columns`, незарегистрированную схему параметров) |
| 3 | `DataValidationError` (`InsufficientDataError`, `CountMismatchError`, `LeakageError`) | Данные не прошли проверку; незамороженный манифест; блокирующие дубликаты; неполные `predictions` |
| 4 | `TechnicalError` | Техническая ошибка запуска |
| 5 | `ProtocolViolationError` (`TestAccessError`, `ThresholdLeakageError`, `ResultTamperedError`) | Нарушение протокола |
| 6 | `NotImplementedAlgorithm` | Алгоритм не зарегистрирован (каркас) |

`cli_common` перехватывает исключения, пишет их в `errors.jsonl` и `run.json` (`failure`), печатает краткое сообщение и возвращает код.

---

## 9. Журналирование и измерения

- Структурный журнал `run.log.jsonl`: `ts_utc, level, run_id, event, payload`. Вывод в stdout — только итог и ошибки; `print` в библиотечном коде запрещён.
- `elapsed_ms` измеряется раннером через `time.perf_counter_ns()` вокруг вызова алгоритма на каждый элемент (`timing_mode=per_item`); для пакетных вызовов — амортизированное время (`amortized_batch`).
- `timers.json` — время стадий (загрузка, вычисление, запись) и суммарное время.
- Пиковая память — через `psutil`, если установлен; иначе `null` с кодом `psutil_unavailable`.
- Версии, оборудование и ОС регистрируются (`environment.json`); измерения времени без записи окружения недопустимы.

---

## 10. Библиотечные примитивы и контрольные примеры (ОБЩ-03)

Готовые SIFT/ORB, цветовые преобразования, линейная алгебра, Adam и стандартный backbone разрешены, но **явно называются**: `primitives.py` содержит реестр `{name, library, version, check_id}`, который попадает в `run.json`. Авторские части (правила объединения, очереди, потери, протоколы, отчётность) реализуются явно. Библиотечная функция не заменяет проверку её поведения — базовые контрольные проверки входят в эту спеку:

| `check_id` | Примитив | Проверка |
|---|---|---|
| `PRIM-SIFT` | SIFT/ORB (OpenCV) | На синтетической шахматной доске находится не меньше заданного числа точек; на копии, повёрнутой на 90°, дескрипторы сопоставляются с ratio test; найденная `findHomography` совпадает с известной `H` в допуске |
| `PRIM-COLOR` | Цветовые преобразования | sRGB↔Lab: белая точка D65 даёт `L=100`; прямое и обратное преобразования дают исходный цвет в допуске; OpenCV и scikit-image согласуются на эталонных значениях |
| `PRIM-LINALG` | NumPy/SciPy | `lstsq`/SVD восстанавливают заранее известное решение на фиксированной системе; `findHomography` с точным соответствием даёт ошибку, близкую к нулю |
| `PRIM-ADAM` | Adam (Keras) | Первый шаг на скалярной задаче совпадает с аналитической формулой `θ₁ = θ₀ − lr·ĝ/(|ĝ| + ε)` (маркер `tf`; пропускается с явной причиной, если TensorFlow не установлен) |
| `PRIM-BACKBONE` | ResNet50 (Keras) | Форма выхода и число параметров совпадают с ожидаемыми для выбранной конфигурации (маркер `tf`) |

Проверки специфичных для программ параметров примитивов добавляются ТЗ 01–03; механизм их регистрации уже есть.

---

## 11. Тестирование и приёмка

### 11.1 Маркеры

`unit` (быстрые, CPU), `smoke` (сквозной прогон на fixtures), `tf` (требует TensorFlow; пропускаются с явной причиной), `slow`.

### 11.2 Набор проверок

| ID | Проверяет |
|---|---|
| T-SPL-01…06 | Разбиения: непересечение и полнота частей, границы (`ceil`/`floor`), производные следуют за оригиналом (KADID), разделение по `class_id` (SOP), независимость от порядка входных ключей, `InsufficientDataError` на пустых частях |
| T-SPL-07 | Эталонные перестановки `PCG64(2026)` для малых `n` (защита от смены потока NumPy) |
| T-DAT-01…04 | Схемы и инварианты INV-01…INV-09; JSON/parquet без NaN/Inf; `null` ≠ `0` ≠ `reject` ≠ ошибка |
| T-DAT-05 | ДАН-04: удаление GT-колонок и файлов не меняет предсказания; `inputs` не содержит запрещённых колонок |
| T-DAT-06 | ДАН-02: искусственные межчастевые дубликаты SHA-256 и `source_group_id` обнаруживаются, `blocking=true`, запуск не стартует |
| T-DAT-07 | ДАН-03: сокращённый HPatches помечается `subset`, имя поднабора попадает в отчёт |
| T-DAT-08 | ДАН-01: поля происхождения заполнены; в репозитории нет изображений датасетов |
| T-DAT-09 | Хэш манифеста устойчив к перезаписи parquet (D-07) |
| T-MET-01…09 | Формулы и свойства раздела 6, ручные примеры, `MixedRelationError`, `y_defect`, подпись положительного класса |
| T-THR-01…03 | Порог только на validation; `ThresholdLeakageError`; чтение `thresholds.json` без возможности менять |
| T-REP-01 | Оговорки раздела 1.6 в README и отчёте |
| T-REP-02 | Воспроизводимость: два запуска smoke дают одинаковый хэш `predictions` |
| T-REP-03 | Resume: прерванный запуск продолжается, завершённое не пересоздаётся; подмена shard — код 5 |
| T-REP-04 | Профили: pilot не читает `test`; confirmatory без `plan.lock` не стартует; `test_opened.json`; `post_test_study` |
| T-CLI-01 | Все команды раздела 8.1 существуют, `--help` возвращает 0; код 6 у каркасов |
| T-CLI-02 | Коды завершения 0/2/3/4/5/6 для соответствующих сценариев |
| T-PRIM-* | Проверки раздела 10 |
| T-AGG-01 | `aggregate` отвергает неполные `predictions` (D-12); `report` включает все запуски плана, в том числе хуже базовых |

Тестовые двойники (`tests/fakes.py`): `RejectAllPredictor`, `RandomScorePredictor` (по seed) и `OracleLabelPredictor` (помечен `oracle`). Они лежат только в тестах и не попадают в поставку как алгоритмы.

### 11.3 Критерии приёмки (ПОСТ-02)

Приёмка считается пройденной, если одновременно:

1. все тесты `unit` и `smoke` проходят на CPU; `tf`-тесты проходят там, где TensorFlow установлен, и пропускаются с причиной иначе;
2. `build-manifests` и `validate-data` на fixtures строят и подтверждают манифест; на специально испорченных fixtures — завершаются кодом 3;
3. два запуска smoke в свежих процессах дают совпадающие хэши `predictions` (на той же среде);
4. lock-файлы получены проверочной установкой в чистом окружении, после которой прошли тесты;
5. README содержит все команды раздела 8, и `T-CLI-01` подтверждает их существование;
6. отчёт на fixtures содержит матрицу 2×3, покрытие по классам, оговорки и результаты хуже базовых без фильтрации.

Превосходство какого-либо метода условием приёмки не является.

### 11.4 Трассировка требований

| Требование ТЗ | Раздел спеки | Проверка |
|---|---|---|
| ОБЩ-01 | 3.1–3.2 | T-CLI-01 |
| ОБЩ-02 | 1.4, 12.2 | Критерий 11.3 п.4 |
| ОБЩ-03 | 10 | T-PRIM-* |
| Разбиения (раздел 2 ТЗ) | 4.1–4.5 | T-SPL-* |
| ДАН-01 | 4.7, 5.1 | T-DAT-08 |
| ДАН-02 | 4.6 | T-DAT-06 |
| ДАН-03 | 4.3 | T-DAT-07 |
| Контракт данных (раздел 3 ТЗ) | 5 | T-DAT-01…04 |
| ДАН-04 | 5.3 | T-DAT-05 |
| ДАН-05 | 5.7 | T-DAT-04 |
| О-1, О-2, Г2-168, О-3 | 6 | T-MET-* |
| Пороги только на validation | 6.4 | T-THR-* |
| ВОСП-01 | 7.1 | T-AGG-01, T-REP-04 |
| ВОСП-02 | 7.2–7.3 | T-REP-02 |
| ВОСП-03 | 7.4 | T-REP-04 |
| ВОСП-04 | 7.4.1, 7.5 | T-REP-04 |
| Интерфейс (раздел 6 ТЗ) | 8 | T-CLI-01, T-CLI-02 |
| ПОСТ-01 | 7.6, 12 | T-REP-03, критерий 11.3 |
| ПОСТ-02 | 1.5, 7.7, 11.3 | T-AGG-01 |

---

## 12. Комплект поставки

### 12.1 Состав (ПОСТ-01)

1. Исходный код (раздел 3.1).
2. Lock-файлы: `requirements.lock` (CPU-профиль) и `requirements-metric.lock` (с TensorFlow/Keras).
3. `README.md`: установка, команды раздела 8, порядок работы (`build-manifests → validate-data → pilot → plan → confirmatory → aggregate → report`), оговорки раздела 1.6.
4. `schemas/`: JSON Schema входных/выходных файлов, **генерируемые из кода** (`scripts/gen_schemas.py`); тест проверяет, что закоммиченные схемы совпадают со сгенерированными.
5. `configs/`: шаблоны без скрытых defaults, smoke-конфиги каждого модуля, реестр версий протокола. Конфигурации опытов ТЗ 01–04 поставляются вместе с этими ТЗ.
6. Контрольные fixtures и их построитель.
7. Автоматические проверки математических свойств (раздел 11).
8. `DECISIONS.md`: журнал решений, начальное наполнение — раздел 2.

### 12.2 Lock-файлы и проверочная установка

1. Чистое виртуальное окружение; установка проекта с нужными extras.
2. Прогон `pytest -m "unit or smoke"` (и `tf`-тестов для `requirements-metric.lock`).
3. Только после успеха фиксируются точные версии (`pip freeze` или `uv pip compile`) вместе с версией Python.
4. Совместимость TensorFlow/Keras и `tf.keras.applications.ResNet50` проверяется этой установкой, а не предполагается.

### 12.3 Resume и повторная поставка

Поставка не содержит результатов и данных. Все каталоги данных и вывода передаются аргументами.

---

## 13. Зависимости от ТЗ 01–04 и риски

**Что приходит из ТЗ 01–04** (интерфейсы для этого уже есть):

- построители `pairs.parquet` для P1–P3 и допустимый режим P1;
- схемы параметров модулей (`register_params_schema`), реестр кодов причин (`register_reason_code`), критерий выбора порога;
- сами алгоритмы (`registry`), определение `comp_*`-компонентов P2;
- эксперименты, сетки, статистика, анализ ошибок и итоговые таблицы (P4).

**Риски и меры**

| Риск | Мера |
|---|---|
| Поток `Generator` изменится между версиями NumPy | Версия зафиксирована lock-файлом; эталонные перестановки `T-SPL-07`; манифест — первичный артефакт |
| Структура поставок датасетов отличается от допущений | Парсеры проверяют структуру и падают с понятной ошибкой; тесты на fixtures воспроизводят публичную раскладку |
| Сокращённая поставка HPatches | Отчёт о полноте, имя поднабора во всех отчётах |
| Недетерминированность TensorFlow на платформе | `enable_op_determinism` где доступно; ограничения в `environment.json`; проверка повторяемости на той же среде |
| Утечка метки через пути/идентификаторы (MVTec) | Непрозрачный `image_id`, `ImageStore` без путей, белый список колонок пар |
| «Подгонка под test» | Профили, `SplitGuard`, реестр версий протокола, `test_opened.json`, `post_test_study` |
| Незаметная потеря пар в предсказаниях | INV-07 и D-12: отсутствие строки — ошибка, а не отказ |
| Несовместимость TF/Keras с выбранной версией Python | Проверочная установка (12.2) до фиксации lock-файла |
