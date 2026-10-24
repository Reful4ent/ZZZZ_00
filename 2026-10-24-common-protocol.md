# Экспериментальный комплекс (ТЗ 00) — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Реализовать общий пакет `expcore` и четыре CLI-модуля (P1 photometric, P2 geometric, P3 metric, P4 experiments), обеспечивающие протокол данных, контракт файлов, метрики с отказом, воспроизводимость и защиту от подгонки под test.

**Architecture:** Один репозиторий, src-layout; `expcore` — чистые функции и небольшие классы без знания об алгоритмах; четыре модуля зависят только от `expcore`. Манифесты замораживаются до результатов, раннер пишет shard-файлы атомарно и умеет resume; алгоритмы подключаются через реестр. Каркасы P1–P3 возвращают код 6.

**Tech Stack:** Python, NumPy, SciPy, OpenCV, scikit-image, pandas, Matplotlib, pyarrow, PyYAML, pytest; TensorFlow/Keras — только extra `metric`; `psutil` — опционально.

**Spec:** `docs/superpowers/specs/2026-10-24-common-protocol-design.md` (ссылки «§» ниже — на её разделы; «D-nn» — решения из §2). Относительно §3.3 спеки план добавляет два модуля `expcore`: `scanners.py` (парсеры раскладок датасетов, Task 9) и `protocol.py` (реестр версий и блокировка test, Task 17).

## Global Constraints

- Пакеты: `src/expcore`, `src/photometric`, `src/geometric`, `src/metric`, `src/experiments`; запуск `python -m <пакет> <команда> --config …`.
- Seeds: разбиение датасетов `2026`; pilot `11, 23, 37`; confirmatory `11, 23, 37, 51, 71`; smoke `11`.
- Разбиения: `np.random.Generator(np.random.PCG64(2026))`, новый генератор на каждую операцию; сортировка по кодовым точкам Unicode (`class_id` SOP — как int); границы только целочисленно: `ceil(20%·n)=(20n+99)//100`, `floor(0.25N)=N//4`, `floor(0.8M)=(4M)//5`.
- KADID: 81 оригинал, 125 производных на оригинал, 10 125 всего; первые 20 перемешанных — validation, остальные — test.
- MVTec: категории `carpet, leather, tile, wood`; `dev = N//4`, `validation = N//4`, остаток — `synthetic_test`; официальный test — `real_test`.
- SOP: 22 634 ID; 59 551 train и 60 502 test изображений; train/validation по `class_id`.
- Коды завершения: 0 успех, 2 `ConfigError`, 3 `DataValidationError`, 4 `TechnicalError`, 5 `ProtocolViolationError`, 6 `NotImplementedAlgorithm`.
- Ни JSON, ни parquet не содержат NaN/Inf (`allow_nan=False`); отсутствие значения — `null` + `reason_code`, никогда `0`.
- `run_id = "{module}-{profile}-{config_hash[:12]}-s{seed}"`; `config_hash` — SHA-256 канонического JSON конфига **без** секции `paths`.
- `image_id = "{dataset_id}:{ordinal:06d}"` (порядок по отсортированному `relative_path`); `relative_path` хранится с разделителем `/`.
- Нет абсолютных путей разработчика в коде; пути только через конфиг/аргументы; код работает на Linux и Windows (`pathlib`).
- Изображения датасетов в репозиторий не попадают; fixtures строятся кодом из seed.
- Завершённые результаты не перезаписываются; флага `--force` нет.
- Оговорка из §1.6 присутствует в README и в каждом отчёте.
- Каждый коммит заканчивается строками:
  `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` и `Claude-Session: https://claude.ai/code/session_01XLs3po6yPMhP2tKEteU7SQ`.
- Версии библиотек в коде/плане не фиксируются: их определяет Task 25.

## Review Focus

Входы, которые спека подразумевает, но которые не очевидны для обычных тестов; каждый закреплён тестом в указанной задаче.

1. Запуск, где все предсказания — отказы: `coverage=0`, `recall_all=0.0`, `recall_cond=null` с причиной `undefined_no_accepted`, без исключения и без деления на ноль (Task 13).
2. `predictions` содержит дубликат ключа или ключ, которого нет в `pairs`, либо не хватает ключа: `aggregate` падает (код 3), пропуск не считается отказом (Task 5, Task 21).
3. Процесс убит посреди записи shard: остаточный временный файл игнорируется и не считается результатом; на resume единица пересчитывается (Task 16).
4. Пути с пробелами, кириллицей и обратными слэшами Windows: `relative_path` нормализуется в `/`, а изображение загружается (Task 9, Task 12).
5. Повреждённое или нечитаемое изображение, а также изображение с неожиданным числом каналов при сканировании: `DataValidationError` с именем файла, без падения `cv2` на `None` (Task 9).

---

# Фаза 1. Фундамент проекта

**Выход фазы:** устанавливаемый пакет, иерархия ошибок, строгий JSON/хэширование, строгий конфиг. `pytest -m unit` зелёный.

### Task 1: Каркас репозитория и ошибки

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/expcore/__init__.py`, `src/expcore/errors.py`, `src/{photometric,geometric,metric,experiments}/__init__.py`, `tests/conftest.py`, `tests/unit/test_errors.py`

**Interfaces:**
- Produces: `ProtocolError(Exception)` с атрибутом класса `exit_code: int`; `ConfigError=2`; `DataValidationError=3` и потомки `InsufficientDataError`, `CountMismatchError`, `LeakageError`; `TechnicalError=4`; `ProtocolViolationError=5` и потомки `TestAccessError`, `ThresholdLeakageError`, `ResultTamperedError`, `MixedRelationError`; `NotImplementedAlgorithm=6`.

- [x] **Step 1: Failing-тест** (`tests/unit/test_errors.py`)

```python
@pytest.mark.unit
def test_exit_codes():
    assert [e.exit_code for e in (ConfigError, DataValidationError, TechnicalError,
            ProtocolViolationError, NotImplementedAlgorithm)] == [2, 3, 4, 5, 6]
    for sub in (InsufficientDataError, CountMismatchError, LeakageError):
        assert issubclass(sub, DataValidationError)
    for sub in (TestAccessError, ThresholdLeakageError, ResultTamperedError, MixedRelationError):
        assert issubclass(sub, ProtocolViolationError)
```

- [x] **Step 2:** `git init`, создать окружение (`python -m venv .venv`), `pip install -e ".[dev]"`; `pytest tests/unit/test_errors.py -v` → FAIL (ImportError).
- [x] **Step 3:** Создать `pyproject.toml` (src-layout, `requires-python` без верхней границы; зависимости из Tech Stack; extras `metric` = `tensorflow`, `dev` = `pytest`, `mem` = `psutil`; маркеры pytest `unit, smoke, tf, slow`), `.gitignore` (`.venv/`, `data/`, `out/`, `__pycache__/`), пакеты-пустышки и `errors.py` по интерфейсу.
- [x] **Step 4:** `pytest tests/unit/test_errors.py -v` → PASS; commit `feat: project scaffold and error hierarchy`.

### Task 2: Строгий JSON, атомарная запись, хэширование

**Files:**
- Create: `src/expcore/io.py`, `src/expcore/hashing.py`, `tests/unit/test_io_hashing.py`

**Interfaces:**
- Produces:
  - `dump_json(obj: Any, path: Path) -> None` — `allow_nan=False`, атомарная запись; NaN/Inf → `DataValidationError`.
  - `load_json(path: Path) -> Any`
  - `atomic_write_bytes(path: Path, data: bytes) -> None` — временный файл `<name>.tmp-<pid>` + `os.replace`.
  - `write_parquet(df: pd.DataFrame, path: Path) -> None` — проверка float-колонок на NaN/Inf, атомарно; `read_parquet(path: Path) -> pd.DataFrame`.
  - `sha256_file(path: Path) -> str` (блоки 1 МиБ); `sha256_text(s: str) -> str`.
  - `canonical_json(obj: Any) -> str` — ключи по алфавиту, без пробелов, UTF-8.
  - `table_content_hash(df: pd.DataFrame, columns: Sequence[str], key: Sequence[str]) -> str` — D-07.

- [x] **Step 1: Failing-тесты**

```python
def test_sha256_known(tmp_path):
    p = tmp_path/"a"; p.write_bytes(b"abc")
    assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
def test_canonical_json():
    assert canonical_json({"b": 1, "a": [1, 2]}) == '{"a":[1,2],"b":1}'
def test_dump_json_rejects_nan(tmp_path):
    with pytest.raises(DataValidationError): dump_json({"x": float("nan")}, tmp_path/"x.json")
    assert not (tmp_path/"x.json").exists()
def test_parquet_rejects_inf(tmp_path):
    with pytest.raises(DataValidationError): write_parquet(pd.DataFrame({"s": [1.0, np.inf]}), tmp_path/"p.parquet")
def test_table_hash_order_and_roundtrip(tmp_path):   # T-DAT-09
    df = pd.DataFrame({"k": ["b", "a"], "v": [2, 1]})
    h = table_content_hash(df, ["k", "v"], ["k"])
    assert h == table_content_hash(df.iloc[::-1], ["k", "v"], ["k"])
    write_parquet(df, tmp_path/"t.parquet")
    assert h == table_content_hash(read_parquet(tmp_path/"t.parquet"), ["k", "v"], ["k"])
def test_atomic_write_failure_leaves_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError): atomic_write_bytes(tmp_path/"f", b"x")
    assert list(tmp_path.iterdir()) == []
```

- [x] **Step 2:** `pytest tests/unit/test_io_hashing.py -v` → FAIL.
- [x] **Step 3:** Реализовать по интерфейсу; `table_content_hash`: строки по ключу, колонки в заданном порядке, каждое значение → `canonical_json`, `null` для пропусков.
- [x] **Step 4:** тесты PASS; commit `feat: strict io and hashing`.

### Task 3: Строгая конфигурация

**Files:**
- Create: `src/expcore/config.py`, `tests/unit/test_config.py`, `configs/templates/full.yaml`

**Interfaces:**
- Consumes: `ConfigError`, `canonical_json`, `sha256_text`.
- Produces:
  - dataclass-схемы секций §5.6: `Config(schema_version, protocol_version, profile, seeds, datasets: Mapping[str, DatasetCfg], inputs, data_modes, algorithms, params, decision, determinism, runner, logging, paths)`.
  - `DatasetCfg(dataset_id, manifest_version, fixture: bool, categories: tuple[str,...]|None, n_validation: int|None, expected: Mapping[str, Any], provenance: Mapping[str, Any])`; `InputsCfg(pair_columns: tuple[str,...], color_order: str)`; `DecisionCfg(tie_rule: str, criterion_id: str|None)`; `RunnerCfg(chunk_size: int)`.
  - `ALLOWED_SEEDS = {"smoke": (11,), "pilot": (11, 23, 37), "confirmatory": (11, 23, 37, 51, 71)}`.
  - `load_config(path: Path, *, module: str, path_overrides: Mapping[str, Path] | None = None) -> Config`
  - `config_hash(cfg: Config) -> str`
  - `register_params_schema(module: str, schema: type) -> None`
  - `strict_from_dict(cls: type, data: Mapping[str, Any]) -> Any` — все ключи обязательны, лишние запрещены, вложенные dataclass.

- [x] **Step 1: Failing-тесты**

```python
def test_missing_key_raises(full_cfg_dict): del full_cfg_dict["decision"]["tie_rule"]; assert_config_error(full_cfg_dict)
def test_unknown_key_raises(full_cfg_dict): full_cfg_dict["extra"] = 1; assert_config_error(full_cfg_dict)
def test_paths_excluded_from_hash(cfg_a, cfg_b_other_paths): assert config_hash(cfg_a) == config_hash(cfg_b_other_paths)
def test_forbidden_pair_columns(full_cfg_dict):
    for col in ("y_match", "gt_H_path", "gt_mask_path"):
        full_cfg_dict["inputs"]["pair_columns"] = ["pair_id", col]; assert_config_error(full_cfg_dict)
def test_seed_must_match_profile(full_cfg_dict): full_cfg_dict["profile"] = "pilot"; full_cfg_dict["seeds"]["seed"] = 51; assert_config_error(full_cfg_dict)
def test_fixture_dataset_only_in_smoke(full_cfg_dict): full_cfg_dict["profile"] = "pilot"; ... fixture True → ConfigError
def test_unregistered_params_schema(full_cfg_dict): assert_config_error(full_cfg_dict, module="unknown_module")
def test_real_kadid_requires_n_validation_20(full_cfg_dict): n_validation=3, fixture=False → ConfigError
def test_path_override_does_not_change_hash(): ...
```

- [x] **Step 2:** FAIL.
- [x] **Step 3:** Реализовать: YAML → `strict_from_dict`; проверки профиль/seed/fixture/KADID `n_validation` (20, если не fixture)/запрещённые колонки; `config_hash = sha256_text(canonical_json(asdict(cfg) без "paths"))`. Записать `configs/templates/full.yaml` — полный шаблон без defaults (проверяется загрузкой в тесте).
- [x] **Step 4:** PASS; commit `feat: strict config loader and hash`.

---

# Фаза 2. Контракт данных

**Выход фазы:** реестры, модель решения и валидаторы схем с инвариантами INV-01…INV-09.

### Task 4: Реестры и модель решения

**Files:**
- Create: `src/expcore/registry.py`, `src/expcore/decisions.py`, `tests/unit/test_decisions.py`

**Interfaces:**
- Produces:
  - `class Outcome(str, Enum): DECIDED="decided"; REJECT="reject"; NOT_APPLICABLE="not_applicable"`
  - `class ScoreDirection(str, Enum): HIGHER_IS_MATCH="higher_is_match"; LOWER_IS_MATCH="lower_is_match"`
  - `@dataclass(frozen=True) class Decision(outcome, y_pred: int|None, score_raw: float|None, score_direction, reason_code: str|None, components: Mapping[str,float])` — `__post_init__` проверяет INV-05/06 по Outcome (нарушение → `DataValidationError`).
  - `unified_score(score_raw: float | None, direction: ScoreDirection) -> float | None` (`negate` при `LOWER_IS_MATCH`).
  - `apply_threshold(score_raw: float, direction: ScoreDirection, threshold: float, tie_rule: str) -> int` — при `">="` граница даёт 1.
  - `RELATIONS = ("P1_SAME_SCENE", "P2_NO_CHANGE_IN_REGION", "P3_SAME_PRODUCT")`.
  - `register_reason_code(code: str, outcome: Outcome) -> None`; `is_registered(code: str, outcome: Outcome) -> bool`; коды `REJECT_*` ↔ `reject`, `NOT_APPLICABLE_*` ↔ `not_applicable`; предустановлены `REJECT_GENERIC`, `NOT_APPLICABLE_NO_REGION`, `SCORE_UNDEFINED`.
  - `register_algorithm(module: str, method_id: str, predictor: object) -> None`; `get_algorithm(module: str, method_id: str) -> object` (нет → `NotImplementedAlgorithm`).

- [x] **Step 1: Failing-тесты**

```python
def test_unified_score(): assert unified_score(0.3, LOWER_IS_MATCH) == -0.3 and unified_score(0.3, HIGHER_IS_MATCH) == 0.3 and unified_score(None, LOWER_IS_MATCH) is None
def test_tie_rule(): assert apply_threshold(0.5, HIGHER_IS_MATCH, 0.5, ">=") == 1
def test_zero_score_is_not_reject(): d = Decision(DECIDED, 0, 0.0, HIGHER_IS_MATCH, None, {}); assert d.outcome is DECIDED   # ДАН-05
def test_reject_needs_registered_code(): with pytest.raises(DataValidationError): Decision(REJECT, None, None, HIGHER_IS_MATCH, "UNKNOWN", {})
def test_decided_requires_y_pred(): with pytest.raises(DataValidationError): Decision(DECIDED, None, 0.1, HIGHER_IS_MATCH, None, {})
def test_get_unregistered_algorithm(): with pytest.raises(NotImplementedAlgorithm): get_algorithm("photometric", "nope")
```

- [x] **Step 2:** FAIL. **Step 3:** реализовать. **Step 4:** PASS; commit `feat: decision model and registries`.

### Task 5: Схемы и инварианты

**Files:**
- Create: `src/expcore/schemas.py`, `tests/unit/test_schemas.py`

**Interfaces:**
- Consumes: `registry.RELATIONS`, `registry.is_registered`, `io`.
- Produces: колонки и dtype таблиц §5.2–5.4 как `IMAGES_COLUMNS`, `PAIRS_COLUMNS`, `PREDICTIONS_COLUMNS` (`dict[str, str]`); `ALLOWED_SPLITS: dict[str, frozenset[str]]` по `dataset_id`; `EVAL_ONLY_PAIR_COLUMNS = ("y_match", "gt_H_path", "gt_mask_path")`; `EVAL_ONLY_IMAGE_COLUMNS = ("defect_type",)`;
  - `validate_images(images: pd.DataFrame) -> None` (INV-01, INV-04)
  - `validate_pairs(pairs: pd.DataFrame, images: pd.DataFrame) -> None` (INV-02, INV-03)
  - `validate_predictions(pred: pd.DataFrame, pairs: pd.DataFrame, *, require_complete: bool) -> None` (INV-05…INV-09; при `require_complete` — INV-07, иначе `DataValidationError`)
  - `validate_datasets_json(obj: list[dict]) -> None`, `validate_run_json(obj: dict) -> None`.
  - Все нарушения → `DataValidationError` с номером инварианта в сообщении.

- [x] **Step 1: Failing-тесты** (по одному на инвариант; ключевые):

```python
def test_inv02_cross_split_pair(): pair.split=="test" but reference in validation → DataValidationError("INV-02")
def test_inv04_group_in_two_splits(): ...
def test_inv05_reject_row_shape(): reject with y_pred=1 → error; reject with q_valid=True → error
def test_inv06_nan_score(): score_raw=float("nan") → error
def test_inv07_missing_duplicate_extra_keys(preds_missing, preds_dup, preds_extra):   # Review Focus 2
    for p in (preds_missing, preds_dup, preds_extra):
        with pytest.raises(DataValidationError, match="INV-07"): validate_predictions(p, pairs, require_complete=True)
def test_inv09_no_technical_error_rows(): outcome="error" → error
def test_sha256_format(): uppercase or len!=64 → INV-01
def test_split_not_allowed_for_dataset(): kadid with split "train" → INV-01
def test_nullable_vs_reason(): score_raw None without reason_code → INV-06
```

- [x] **Step 2:** FAIL. **Step 3:** реализовать на pandas без построчных циклов, где возможно. **Step 4:** PASS; commit `feat: data contract schemas and invariants`.

---

# Фаза 3. Датасеты, разбиения, манифесты

**Выход фазы:** `build_manifests` на синтетических fixtures строит замороженные манифесты; разбиения, проверка целостности и отчёт о полноте работают.

### Task 6: Генератор и разбиения KADID, HPatches

**Files:**
- Create: `src/expcore/splits/__init__.py`, `rng.py`, `kadid.py`, `hpatches.py`, `tests/unit/test_splits_a.py`

**Interfaces:**
- Produces:
  - `rng.shuffled(keys: Sequence[K], seed: int = 2026) -> list[K]` (сортировка, затем `PCG64(seed).permutation`; `K` — str или int).
  - `rng.ceil_pct(n: int, pct: int) -> int` = `(pct*n + 99)//100`.
  - `kadid.split_kadid(original_ids: Sequence[str], n_validation: int) -> dict[str, list[str]]` с ключами `validation`, `test`; пустая часть → `InsufficientDataError`.
  - `hpatches.split_hpatches(sequence_names: Sequence[str]) -> dict[str, list[str]]` — раздельно для префиксов `i_` и `v_`, `validation` = `ceil_pct(n, 20)` на префикс; имя без префикса → `DataValidationError`.

- [x] **Step 1: Failing-тесты**

```python
def test_ceil_pct(): assert [ceil_pct(n, 20) for n in (1, 5, 57, 59)] == [1, 1, 12, 12]
def test_shuffled_independent_of_input_order(): assert shuffled(["c","a","b"]) == shuffled(["b","c","a"])
def test_shuffled_reference_permutation():   # T-SPL-07: эталон зафиксирован при реализации один раз из PCG64(2026)
    assert shuffled(list("abcdefgh")) == REFERENCE_PERM_8
def test_kadid_81_20_61(): parts = split_kadid([f"I{n:02d}" for n in range(1, 82)], 20); assert (len(parts["validation"]), len(parts["test"])) == (20, 61) and not set(parts["validation"]) & set(parts["test"])
def test_kadid_empty_part(): with pytest.raises(InsufficientDataError): split_kadid(["I01"], 1)
def test_hpatches_per_prefix_and_ceil(): names = [f"i_{k:02d}" for k in range(57)] + [f"v_{k:02d}" for k in range(59)]; parts = split_hpatches(names); assert sum(n.startswith("i_") for n in parts["validation"]) == 12 and sum(n.startswith("v_") for n in parts["validation"]) == 12
def test_hpatches_single_sequence_prefix_fails(): with pytest.raises(InsufficientDataError): split_hpatches(["i_a", "v_a", "v_b", "v_c", "v_d", "v_e"])
def test_split_deterministic_across_calls(): assert split_kadid(ids, 20) == split_kadid(list(reversed(ids)), 20)
```

- [x] **Step 2:** FAIL. **Step 3:** реализовать; константу `REFERENCE_PERM_8` получить однократным запуском реализации и вписать в тест как литерал (эталон на случай смены потока NumPy). **Step 4:** PASS; commit `feat: rng helpers, KADID and HPatches splits`.

### Task 7: Разбиения MVTec и SOP

**Files:**
- Create: `src/expcore/splits/mvtec.py`, `src/expcore/splits/sop.py`, `tests/unit/test_splits_b.py`

**Interfaces:**
- Consumes: `rng.shuffled`.
- Produces:
  - `split_mvtec_category(train_good_keys: Sequence[str]) -> dict[str, list[str]]` — ключи `dev`, `validation`, `synthetic_test`; `N<4` → `InsufficientDataError`.
  - `split_sop(class_ids: Sequence[int]) -> dict[str, list[int]]` — ключи `train`, `validation`; `train = (4*M)//5`.

- [x] **Step 1: Failing-тесты**

```python
def test_mvtec_boundaries(): parts = split_mvtec_category([f"carpet/{i:03d}" for i in range(280)]); assert [len(parts[k]) for k in ("dev","validation","synthetic_test")] == [70, 70, 140]
def test_mvtec_odd_n(): parts = split_mvtec_category([f"c/{i:03d}" for i in range(247)]); assert [len(parts[k]) for k in ("dev","validation","synthetic_test")] == [61, 61, 125]
def test_mvtec_too_small(): with pytest.raises(InsufficientDataError): split_mvtec_category(["a","b","c"])
def test_sop_80_20(): parts = split_sop(list(range(1, 11319))); assert len(parts["train"]) == (4*11318)//5 and len(parts["train"]) + len(parts["validation"]) == 11318
def test_sop_numeric_sort(): assert split_sop([2, 10, 1, 3, 4])  # порядок входа не влияет; 10 сортируется как число после 4
def test_parts_disjoint_and_complete(): for n in (4, 7, 100, 247): parts = split_mvtec_category([f"c/{i:03d}" for i in range(n)]); assert sorted(sum(parts.values(), [])) == sorted(f"c/{i:03d}" for i in range(n))
```

- [x] **Step 2:** FAIL. **Step 3:** реализовать. **Step 4:** PASS; commit `feat: MVTec and SOP splits`.

### Task 8: Построитель fixtures

**Files:**
- Create: `tests/fixtures_builder.py`, `tests/unit/test_fixtures_builder.py`, расширение `tests/conftest.py`

**Interfaces:**
- Produces (все строят синтетические PNG детерминированно из seed, в публичной раскладке поставки):
  - `make_kadid(root: Path, n_originals: int = 6, n_types: int = 2, n_levels: int = 2) -> dict[str, int]` — `I01.png`, `I01_01_01.png`…
  - `make_hpatches(root: Path, n_i: int = 5, n_v: int = 5, complete: bool = True) -> dict[str, int]` — `<seq>/1.ppm…6.ppm`, `H_1_2…H_1_6`
  - `make_mvtec(root: Path, categories: Sequence[str] = ("carpet","leather","tile","wood"), n_train: int = 8, n_test_good: int = 2, n_test_defect: int = 2) -> dict[str, int]` — `train/good`, `test/good`, `test/<defect>`, `ground_truth/<defect>/<stem>_mask.png`
  - `make_sop(root: Path, n_train_classes: int = 10, n_test_classes: int = 4, per_class: int = 3) -> dict[str, int]` — `Ebay_train.txt`, `Ebay_test.txt`
  - фикстуры pytest `kadid_root`, `hpatches_root`, `mvtec_root`, `sop_root`, `fixture_cfg` (конфиг с `fixture: true`, профиль smoke, `expected` по возвращённым счётчикам, `n_validation: 2`).

- [x] **Step 1: Failing-тест**

```python
def test_fixtures_deterministic_and_complete(tmp_path):
    a, b = tmp_path/"a", tmp_path/"b"; make_kadid(a); make_kadid(b)
    assert sha_tree(a) == sha_tree(b)
    counts = make_kadid(tmp_path/"c", 6, 2, 2); assert counts == {"originals": 6, "derivatives": 6*2*2}
    # то же — hpatches (6 изображений + 5 H на последовательность), mvtec (маски у дефектных), sop (строки в txt = изображения)
```

- [x] **Step 2:** FAIL. **Step 3:** реализовать на NumPy (`PCG64(7)`) + `cv2.imwrite`; сами файлы нигде не коммитятся. **Step 4:** PASS; commit `test: synthetic dataset fixtures builder`.

### Task 9: Сканеры датасетов и проверка состава

**Files:**
- Create: `src/expcore/scanners.py`, `tests/unit/test_scanners.py`

**Interfaces:**
- Consumes: `errors`, конфиг `DatasetCfg`.
- Produces:
  - `scan(dataset_cfg: DatasetCfg, data_root: Path) -> pd.DataFrame` — сырая таблица с колонками `natural_key, dataset_id, source_group_id, class_id, category_id, relative_path, defect_type, role, width, height, channels`, где `role ∈ {original, derivative, train_good, test, sequence_image}` (по набору); `relative_path` с разделителем `/`.
  - `check_counts(raw: pd.DataFrame, dataset_cfg: DatasetCfg) -> dict` — сверка с `expected`; расхождение → `CountMismatchError` (HPatches: не ошибка, а результат полноты).
  - `hpatches_completeness(raw: pd.DataFrame, dataset_cfg: DatasetCfg) -> dict` — `{"status": "full"|"subset"|"unverified", "subset_name": str|None, "sequences": [...], "incomplete": [...]}`.
  - `read_image_meta(path: Path) -> tuple[int, int, int]` — `(width, height, channels)`; нечитаемый файл → `DataValidationError` с путём.

- [x] **Step 1: Failing-тесты**

```python
def test_scan_kadid_counts(kadid_root, fixture_cfg): raw = scan(cfg("kadid10k"), kadid_root); assert (raw.role == "original").sum() == 6 and (raw.role == "derivative").sum() == 24
def test_kadid_count_mismatch(kadid_root, fixture_cfg_wrong_expected): with pytest.raises(CountMismatchError): check_counts(...)
def test_mvtec_scan_excludes_unused_categories(mvtec_root): ...
def test_hpatches_incomplete_is_subset(tmp_path):   # T-DAT-07
    make_hpatches(tmp_path, complete=False); rep = hpatches_completeness(...); assert rep["status"] == "subset" and rep["incomplete"]
def test_hpatches_without_expected_is_unverified(...): rep["status"] == "unverified"
def test_corrupt_image_names_file(tmp_path):      # Review Focus 5
    (tmp_path/"I01.png").write_bytes(b"not an image"); with pytest.raises(DataValidationError, match="I01.png"): read_image_meta(tmp_path/"I01.png")
def test_paths_with_spaces_unicode_normalized(tmp_path): path "папка с пробелом\\x.png" → relative_path == "папка с пробелом/x.png"   # Review Focus 4
def test_sop_class_disjoint(sop_root): ...  # пересечение train/test class_id → LeakageError
```

- [x] **Step 2:** FAIL. **Step 3:** реализовать парсеры по публичным раскладкам; при несоответствии структуры — `DataValidationError` с описанием. **Step 4:** PASS; commit `feat: dataset scanners and count checks`.

### Task 10: Целостность и дубликаты

**Files:**
- Create: `src/expcore/integrity.py`, `tests/unit/test_integrity.py`

**Interfaces:**
- Consumes: `hashing.sha256_file`.
- Produces: `check_integrity(images: pd.DataFrame) -> dict` — отчёт §4.6: `{"cross_split_sha256": [...], "cross_split_groups": [...], "within_split_duplicates": [...], "blocking": bool}`; `write_duplicates_report(report: dict, path: Path) -> None`; `assert_not_blocking(report: dict) -> None` → `LeakageError`.

- [x] **Step 1: Failing-тесты** (`T-DAT-06`)

```python
def test_cross_split_sha_detected(): two images same sha in validation/test → blocking True and listed
def test_cross_split_group_detected(): ...
def test_within_split_duplicate_not_blocking(): blocking False, listed
def test_clean_images(): blocking False and empty lists
def test_assert_not_blocking_raises(): with pytest.raises(LeakageError): ...
```

- [x] **Step 2:** FAIL. **Step 3:** реализовать группировкой pandas. **Step 4:** PASS; commit `feat: integrity and duplicate checks`.

### Task 11: Построение, заморозка и загрузка манифестов

**Files:**
- Create: `src/expcore/manifests.py`, `tests/unit/test_manifests.py`

**Interfaces:**
- Consumes: `scanners`, `splits.*`, `integrity`, `schemas`, `io`, `hashing`, `Config`.
- Produces:
  - `assign_splits(raw: pd.DataFrame, dataset_cfg: DatasetCfg) -> pd.DataFrame` (добавляет колонку `split`, применяя Task 6–7).
  - `build_manifests(cfg: Config, data_root: Path, manifest_dir: Path, overrides: Mapping | None = None) -> ManifestInfo` — порядок: scan → overrides (`exclude_image`) → check_counts → assign_splits → image_id (D-05) → sha256 → validate_images → integrity → пишет `datasets.json`, `images.parquet`, `completeness_report.json`, `duplicates_report.json`, `exclusions.json`; **последним** — `manifest.frozen.json`. Существующая версия → `ProtocolViolationError`; блокирующие дубликаты → `LeakageError` (манифест не замораживается).
  - `@dataclass(frozen=True) class ManifestInfo(version: str, manifest_hash: str, dir: Path)`.
  - `load_manifest(manifest_dir: Path, version: str, *, verify: bool = True) -> LoadedManifest` с `.datasets: list[dict]`, `.images: pd.DataFrame`; незамороженный → `DataValidationError`; расхождение хэша → `DataValidationError`.
  - `manifest_hash(images: pd.DataFrame, datasets: list[dict], pairs: pd.DataFrame | None) -> str`.
  - `write_pairs(pairs: pd.DataFrame, images: pd.DataFrame, manifest_dir: Path, version: str, program: str) -> None` — валидация INV-02/03, запись `p1|p2|p3/pairs.parquet`.

- [x] **Step 1: Failing-тесты**

```python
def test_build_all_four_datasets_smoke(fixture_cfg, all_roots, tmp_path):
    info = build_manifests(fixture_cfg, data_root, tmp_path); m = load_manifest(tmp_path, info.version)
    assert set(m.images.dataset_id) == {"kadid10k", "hpatches", "mvtec_ad", "sop"} and m.images.image_id.is_unique
def test_kadid_derivatives_follow_original(manifest): groups → one split each
def test_image_id_opaque(manifest): no row's image_id contains "good", "defect", or a filename stem
def test_frozen_marker_written_last(...): manifest.frozen.json exists and mtime >= every other file
def test_rebuild_same_version_refused(...): ProtocolViolationError
def test_tampered_images_detected(...): modify a cell in images.parquet → load_manifest raises DataValidationError
def test_unfrozen_manifest_refused(...)
def test_blocking_duplicate_not_frozen(...): copy a file across splits → LeakageError and no manifest.frozen.json
def test_overrides_exclude_image_new_version(...): exclusions.json lists reason; counts updated
def test_hash_is_stable_across_builds(...): two builds in different dirs → equal manifest_hash
def test_mvtec_real_test_and_masks_not_in_inputs(manifest): defect_type only in EVAL_ONLY_IMAGE_COLUMNS
```

- [x] **Step 2:** FAIL. **Step 3:** реализовать; `image_id` присваивается после сортировки по `relative_path`; `discovered_counts` включает изображения, группы, классы, части, категории и результат `check_counts`; поля происхождения §4.7 берутся из `provenance` конфига (SHA-256 архива и лицензии вычисляются, если файлы указаны; иначе `archive_verified=false`). **Step 4:** PASS; commit `feat: manifest build, freeze and load`.

---

# Фаза 4. Загрузчики, метрики, пороги

**Выход фазы:** алгоритму выдаются только входные представления; формулы §6 и защита порогов проверены.

### Task 12: Загрузчики и охранник доступа

**Files:**
- Create: `src/expcore/loaders.py`, `tests/unit/test_loaders.py`

**Interfaces:**
- Consumes: `LoadedManifest`, `schemas.EVAL_ONLY_*`.
- Produces:
  - `@dataclass(frozen=True) class PairInput(pair_id, query_id, reference_id, relation_id, extras: Mapping[str, Any])` (extras — только колонки белого списка); `QueryInput(query_id: str, extras: Mapping[str, Any])`.
  - `class SplitGuard: __init__(self, profile: str, *, fixture: bool)`; `check(self, split: str) -> None` → `TestAccessError` по таблице §7.4 (pilot: `train/dev/validation`; confirmatory: всё, но `test` — только через `protocol` Task 17; smoke+fixture: всё).
  - `class ImageStore: __init__(self, data_root: Path, images: pd.DataFrame, guard: SplitGuard)`; `load(self, image_id: str, color: Literal["bgr","rgb","gray"]) -> np.ndarray`.
  - `load_pair_inputs(pairs: pd.DataFrame, columns: Sequence[str], split: str, guard: SplitGuard) -> list[PairInput]`.
  - `load_pair_evaluation(pairs: pd.DataFrame, split: str) -> pd.DataFrame` (с `y_match`, `gt_*`).

- [x] **Step 1: Failing-тесты**

```python
def test_inputs_have_no_forbidden_columns(pairs):           # INV-11
    for p in load_pair_inputs(pairs, ["pair_id","query_id","reference_id","relation_id"], "validation", guard):
        assert not {"y_match","gt_H_path","gt_mask_path"} & set(p.extras) and not hasattr(p, "y_match")
def test_predictions_unchanged_without_gt(...):             # T-DAT-05
    run_fake_predictor(pairs) == run_fake_predictor(pairs.drop(columns=["gt_H_path","gt_mask_path"])) ; also with GT files deleted
def test_pilot_cannot_load_test(): with pytest.raises(TestAccessError): ImageStore(...).load(id_in_test, "gray") under pilot
def test_color_required(): load(id) without color → TypeError
def test_imagestore_loads_path_with_spaces_and_unicode(...): # Review Focus 4
def test_imagestore_never_exposes_paths(): not hasattr(store, "path_of")
```

- [x] **Step 2:** FAIL. **Step 3:** реализовать; `ImageStore` читает через `cv2.imdecode(np.fromfile(...))`, чтобы не зависеть от кодировки пути на Windows. **Step 4:** PASS; commit `feat: loaders with input/evaluation views`.

### Task 13: Метрики с отказом

**Files:**
- Create: `src/expcore/metrics.py`, `tests/unit/test_metrics.py`

**Interfaces:**
- Consumes: `Outcome`, `MixedRelationError`.
- Produces:
  - `@dataclass(frozen=True) class MetricsRow` с полями §6.3 (`tp, fp, tn, fn, rej_pos, rej_neg, n_pos, n_neg, n_not_applicable, coverage, coverage_pos, coverage_neg, rho_rej, recall_all, fpr_all, correct_all, recall_cond, fpr_cond, correct_cond, positive_class, null_reasons: dict[str, str]`).
  - `compute_metrics(df: pd.DataFrame, positive_class: Literal["match","defect"] = "match") -> MetricsRow` — `df`: колонки `y_match, outcome, y_pred, relation_id`; смешанные `relation_id` → `MixedRelationError`.
  - `confusion_2x3(df: pd.DataFrame, positive_class: str) -> np.ndarray` (форма `(2, 3)`: строки `[positive, negative]`, столбцы `[pred_pos, pred_neg, reject]`).

- [x] **Step 1: Failing-тесты** (ручной пример: 10 пар, `N+=6`, `N−=4`; принято 8: `TP=3, FN=1, FP=1, TN=3`, отказы: 2 у положительных)

```python
def test_hand_example():
    m = compute_metrics(HAND_DF)
    assert (m.tp, m.fn, m.fp, m.tn, m.rej_pos, m.rej_neg) == (3, 1, 1, 3, 2, 0)
    assert m.coverage == 0.8 and m.rho_rej == pytest.approx(0.2)
    assert m.recall_all == 3/6 and m.fpr_all == 1/4 and m.correct_all == 6/10
    assert m.recall_cond == 3/4 and m.fpr_cond == 1/4 and m.correct_cond == 6/8
    assert m.coverage_pos == 4/6 and m.coverage_neg == 1.0
def test_product_identities(): recall_all == recall_cond*coverage_pos; fpr_all == fpr_cond*coverage_neg; correct_all == correct_cond*coverage
def test_matrix_rows_sum(): confusion_2x3(HAND_DF,"match").tolist() == [[3,1,2],[1,3,0]]
def test_order_invariance(): shuffled df → same row
def test_all_reject_run():                                   # Review Focus 1
    m = compute_metrics(ALL_REJECT); assert m.coverage == 0.0 and m.recall_all == 0.0 and m.recall_cond is None and m.null_reasons["recall_cond"] == "undefined_no_accepted"
def test_empty_class_is_null(): no negatives → fpr_all None, reason "undefined_empty_class"
def test_not_applicable_excluded_from_Q(): n_not_applicable counted, not in coverage denominator
def test_mixed_relations_raise(): MixedRelationError
def test_defect_swap(): m2 = compute_metrics(HAND_DF, "defect"); assert m2.correct_all == m.correct_all and m2.recall_all == 3/4 and m2.fpr_all == 1/6 and m2.positive_class == "defect"   # TN/N−, FN/N+
```

- [x] **Step 2:** FAIL. **Step 3:** реализовать счётчиками numpy; смена класса — инверсия меток и предсказаний принятых строк, отказы остаются отказами. **Step 4:** PASS; commit `feat: metrics with reject semantics`.

### Task 14: Пороги только на validation

**Files:**
- Create: `src/expcore/thresholds.py`, `tests/unit/test_thresholds.py`

**Interfaces:**
- Consumes: `unified_score`, `ThresholdLeakageError`, `ProtocolViolationError`.
- Produces:
  - `@dataclass(frozen=True) class Threshold(run_id, method_id, criterion_id, value: float, split_used: str, n_validation_pairs: int, manifest_hash: str)`.
  - `select_threshold(val: pd.DataFrame, criterion: Callable[[np.ndarray, np.ndarray], float], *, run_id: str, method_id: str, criterion_id: str, manifest_hash: str) -> Threshold` — `val` содержит `split`, `y_match`, `score_raw`, `score_direction`; все `split` должны быть `validation`; критерий получает унифицированные оценки и метки и возвращает порог.
  - `save_threshold(t: Threshold, path: Path) -> None`; `load_threshold(path: Path) -> Threshold` (`split_used != "validation"` или файла нет → `ProtocolViolationError`).

- [x] **Step 1: Failing-тесты**

```python
def test_rejects_non_validation(): with pytest.raises(ThresholdLeakageError): select_threshold(df_with_test_rows, crit, ...)
def test_uses_unified_score(): lower_is_match scores are negated before criterion
def test_roundtrip(tmp_path): save→load equal
def test_load_missing_or_wrong_split(tmp_path): ProtocolViolationError
def test_only_validation_rows_pass(): df all validation → Threshold.split_used == "validation"
```

- [x] **Step 2:** FAIL. **Step 3:** реализовать. **Step 4:** PASS; commit `feat: validation-only threshold selection`.

---

# Фаза 5. Запуск и воспроизводимость

**Выход фазы:** раннер с resume без перезаписи, фиксация seeds/окружения, блокировка test.

### Task 15: Seeds, окружение, журнал запуска

**Files:**
- Create: `src/expcore/seeding.py`, `src/expcore/runlog.py`, `src/expcore/timing.py`, `tests/unit/test_seeding_runlog.py`

**Interfaces:**
- Produces:
  - `STREAM_INDEX = {"sampler": 0, "augment": 1, "init": 2, "bootstrap": 3}`.
  - `init_seeds(seed: int) -> None` (`random`, `np.random.seed`, `cv2.setRNGSeed`, TensorFlow при наличии: `tf.keras.utils.set_random_seed`).
  - `stream(seed: int, name: str, unit_id: int) -> np.random.Generator`.
  - `capture_rng_state() -> dict` (JSON-сериализуемо; OpenCV — только seed + пометка «состояние не читается»).
  - `collect_environment(*, deterministic_ops: bool | None, extra: Mapping | None = None) -> dict` (Python, ОС, CPU, ОЗУ, GPU при наличии, версии библиотек, переменные `PYTHONHASHSEED`, `TF_DETERMINISTIC_OPS`, `CUDA_VISIBLE_DEVICES`, `nondeterminism_notes`, `library_primitives`).
  - `make_run_id(module: str, profile: str, config_hash: str, seed: int) -> str`.
  - `git_state(repo: Path) -> tuple[str | None, str | None]` — `(commit, dirty_diff_hash)`.
  - `class RunRecord`: `start(...)`, `finish(status, failure=None)`, запись `run.json` по §5.5 с `attempts`; `JsonlLogger(path: Path, run_id: str)` с методом `log(level, event, **payload)`.
  - `timing.timed(fn, *args) -> tuple[Any, float]` (мс, `perf_counter_ns`); `peak_memory_mb() -> tuple[float | None, str | None]`.

- [x] **Step 1: Failing-тесты**

```python
def test_streams_independent_of_call_order(): a = stream(11,"sampler",3).random(); stream(11,"sampler",0).random(); assert stream(11,"sampler",3).random() == a
def test_streams_differ_by_name_and_unit(): ...
def test_init_seeds_reproducible(): init_seeds(11); x = np.random.rand(); init_seeds(11); assert np.random.rand() == x
def test_rng_state_json_serializable(): dump_json(capture_rng_state(), tmp)
def test_run_id_format(): make_run_id("geometric","pilot","a"*64,23) == "geometric-pilot-aaaaaaaaaaaa-s23"
def test_run_json_valid_and_status_transitions(tmp_path): start→running (finished_at null) → finish completed; validate_run_json passes; attempts has 1 entry
def test_failed_then_resume_appends_attempt(...)
def test_confirmatory_requires_git_commit(...): git_commit None in confirmatory → ProtocolViolationError
def test_peak_memory_null_reason_without_psutil(monkeypatch): (None, "psutil_unavailable")
```

- [x] **Step 2:** FAIL. **Step 3:** реализовать. **Step 4:** PASS; commit `feat: seeding, run record, environment, timing`.

### Task 16: Раннер и resume

**Files:**
- Create: `src/expcore/resume.py`, `tests/unit/test_resume.py`

**Interfaces:**
- Consumes: `io`, `runlog`, `timing`, `schemas`, `registry`, `loaders`.
- Produces:
  - `@dataclass class RunContext(run_id, seed, profile, run_dir: Path, cfg: Config)`.
  - `class Runner: __init__(self, ctx: RunContext, predictor: PairPredictor | QueryPredictor, images: ImageStore)`; `run(self, items: Sequence[PairInput | QueryInput], *, method_id: str) -> Path` — делит на единицы по `cfg.runner.chunk_size` (`unit_id` — номер чанка по отсортированному списку), на каждую единицу пишет `predictions/part-{unit_id:06d}.parquet` (атомарно), обновляет `progress.json` (`unit_id`, `n_rows`, `sha256`), в конце пишет `_SUCCESS`.
  - Поведение: единица с совпадающим shard-хэшем пропускается; несовпадение хэша завершённой → `ResultTamperedError`; файлы `*.tmp-*` игнорируются и удаляются; технические исключения алгоритма → `TechnicalError`, строка в `errors.jsonl`, `status=failed`; `elapsed_ms` измеряется раннером; `Decision` преобразуется в строку `predictions` (`q_valid = outcome == decided`, `timing_mode`).
  - `read_predictions(run_dir: Path) -> pd.DataFrame`; `predictions_content_hash(df: pd.DataFrame) -> str` (для `T-REP-02`).

- [x] **Step 1: Failing-тесты** (`T-REP-03`, `tests/fakes.py`: `RejectAllPredictor`, `RandomScorePredictor(seed-stream)`, `OracleLabelPredictor`)

```python
def test_full_run_writes_valid_predictions(ctx, pairs, store): Runner(...).run(...); validate_predictions(read_predictions(ctx.run_dir), pairs, require_complete=True)
def test_resume_skips_completed_units(...): crash predictor on unit 2 → TechnicalError; rerun with fixed predictor → units 0,1 shard files unchanged (mtime/sha), final complete; attempts == 2
def test_completed_run_not_rewritten(...): second run() leaves all shards byte-identical
def test_tampered_shard_raises(...): modify shard → ResultTamperedError
def test_leftover_tmp_ignored(...):                      # Review Focus 3
    (run_dir/"predictions"/"part-000001.parquet.tmp-123").write_bytes(b"junk"); rerun → completes, junk file removed, results valid
def test_results_independent_of_resume_history(...): RandomScorePredictor with interrupted+resumed run == uninterrupted run (content hash)
def test_technical_error_not_in_predictions(...): INV-09
def test_elapsed_ms_recorded(...): all non-null, >= 0
```

- [x] **Step 2:** FAIL. **Step 3:** реализовать; случайность предиктора — через `ctx` и `stream(seed, "sampler", unit_id)`. Помимо shard-файлов раннер пишет в каталог запуска `config.resolved.yaml`, `environment.json`, `rng_state.json` (в начале и на границах единиц), `file_order.json` (порядок пар/запросов), `timers.json`, `run.log.jsonl`; `AugmentationLogger(run_dir).log(unit_id, sample_id, aug_name, params)` копит строки в `augmentations.parquet`. Добавить тест `test_run_dir_contains_all_artifacts` (перечень файлов из §7.6 присутствует, JSON-файлы проходят `load_json`). **Step 4:** PASS; commit `feat: runner with atomic shards and resume`.

### Task 17: Версии протокола и блокировка test

**Files:**
- Create: `src/expcore/protocol.py`, `configs/protocol_registry.json`, `tests/unit/test_protocol.py`

**Interfaces:**
- Consumes: `io`, `SplitGuard`.
- Produces:
  - `load_registry(path: Path) -> list[dict]`; `register_version(path: Path, version: str, parent: str | None) -> None` (новая версия; дубликат → `ProtocolViolationError`).
  - `write_plan_lock(out_dir: Path, version: str, runs: Sequence[Mapping]) -> str` (возвращает `plan_hash`); `check_run_in_plan(out_dir: Path, config_hash: str) -> None` → `ProtocolViolationError`.
  - `open_test(registry_path: Path, out_dir: Path, version: str, run_id: str) -> None` — создаёт `test_opened.json`, ставит `test_opened=true`.
  - `is_post_test(registry_path: Path, version: str) -> bool` — родитель с открытым test.
  - `assert_plan_mutable(registry_path: Path, version: str) -> None` — версия с открытым test → `ProtocolViolationError` (код 5).
  - `SplitGuard.allow_test(self) -> None` — вызывается только после `check_run_in_plan` и `open_test`.

- [x] **Step 1: Failing-тесты** (`T-REP-04`)

```python
def test_confirmatory_run_must_be_in_plan(...): ProtocolViolationError without plan.lock.json
def test_first_test_access_creates_marker(...): test_opened.json exists; registry test_opened True
def test_plan_immutable_after_test_open(...): assert_plan_mutable raises
def test_child_version_marked_post_test(...): is_post_test True
def test_new_version_required_for_grid_change(...): register_version duplicate → error
def test_guard_denies_test_before_open(...): TestAccessError
```

- [x] **Step 2:** FAIL. **Step 3:** реализовать. **Step 4:** PASS; commit `feat: protocol versions and test lock`.

---

# Фаза 6. CLI

**Выход фазы:** все команды §8 работают на fixtures; коды завершения соответствуют §8.3.

### Task 18: Общий CLI и каркасы P1–P3

**Files:**
- Create: `src/expcore/cli_common.py`, `src/{photometric,geometric}/{__main__.py,cli.py}`, `src/metric/{__main__.py,cli.py}`, `tests/unit/test_cli_skeletons.py`

**Interfaces:**
- Consumes: `load_config`, `load_manifest`, `Runner`, `registry`, `RunRecord`.
- Produces:
  - `base_parser(prog: str) -> argparse.ArgumentParser` — `--config` обязателен, `--data-root`, `--manifest-dir`, `--out-root` необязательны.
  - `run_cli(main: Callable[[argparse.Namespace], None], argv: Sequence[str]) -> int` — перехват `ProtocolError`, запись `errors.jsonl`/`run.json.failure`, возврат `exit_code`; иные исключения → 4.
  - `photometric.cli.main(argv) -> int` (`run`), `geometric.cli.main(argv) -> int` (`run`), `metric.cli.main(argv) -> int` (`train|embed|search`; TensorFlow импортируется только внутри обработчиков). Каждый: загрузка конфига и замороженного манифеста, проверка профиля/split, `init_seeds`, каталог запуска, `RunRecord`, затем `get_algorithm(module, method_id)`; не найден → код 6, `run.json.status=failed`, `failure.type="NotImplementedAlgorithm"`.

- [x] **Step 1: Failing-тесты** (`T-CLI-01`, часть `T-CLI-02`)

```python
@pytest.mark.parametrize("mod,cmd", [("photometric",["run"]),("geometric",["run"]),("metric",["train"]),("metric",["embed"]),("metric",["search"]),("experiments",["plan"]),...])
def test_help_exits_zero(mod, cmd): assert subprocess.run([sys.executable,"-m",mod,*cmd,"--help"]).returncode == 0
def test_skeleton_returns_6(fixture_cfg_path, frozen_manifest): assert main(["run","--config",str(fixture_cfg_path)]) == 6; run.json status failed
def test_missing_config_exit_2(): ...
def test_unfrozen_manifest_exit_3(...): ...
def test_registered_fake_algorithm_runs(...): register fake → exit 0, _SUCCESS exists
def test_metric_import_does_not_import_tensorflow(): import metric.cli; assert "tensorflow" not in sys.modules
def test_no_absolute_dev_paths_in_source(): grep src for "/home/", "C:\\" → none
def test_technical_error_exit_4(...): predictor raising RuntimeError → exit 4, run.json status failed, errors.jsonl non-empty, predictions have no error rows
def test_confirmatory_run_opens_test_via_protocol(...): confirmatory run with plan.lock → test_opened.json created before first test image load
```

- [x] **Step 2:** FAIL. **Step 3:** реализовать; для confirmatory обработчик вызывает `protocol.check_run_in_plan` и `protocol.open_test` до `SplitGuard.allow_test()`; `metric`-обработчики вызывают `tf.config.experimental.enable_op_determinism()`, если функция доступна, и передают результат в `collect_environment(deterministic_ops=…)`. **Step 4:** PASS; commit `feat: common CLI and P1-P3 skeletons`.

### Task 19: `experiments build-manifests` и `validate-data`

**Files:**
- Create: `src/experiments/__main__.py`, `src/experiments/cli.py`, `src/experiments/validate.py`, `tests/unit/test_validate_data.py`

**Interfaces:**
- Consumes: `build_manifests`, `load_manifest`, `check_integrity`, `assign_splits`, `scan`.
- Produces: `experiments.cli.main(argv) -> int` с подкомандами `build-manifests`, `validate-data`, `plan`, `run`, `aggregate`, `report` (последние четыре — Tasks 20–21); `validate.validate_data(cfg: Config, data_root: Path, manifest_dir: Path) -> dict` — шаги §4.8 (хэш, наличие файлов и SHA-256, схемы, повторное разбиение vs сохранённое, пересечения), пишет `validation_report.json`.

- [x] **Step 1: Failing-тесты**

```python
def test_build_then_validate_ok(...): rc_build == 0 and rc_validate == 0 and report["ok"] is True
def test_validate_without_manifest_exit_3(...)
def test_validate_detects_deleted_image(...): exit 3, report lists file
def test_validate_detects_modified_image_sha(...)
def test_validate_detects_split_drift(...): edit a split in images.parquet (and fake the frozen hash) → recomputed split mismatch → exit 3
def test_validate_blocks_on_cross_split_duplicates(...): T-DAT-06 end-to-end, exit 3
def test_provenance_fields_present(...): T-DAT-08 — datasets.json has source_url, download_date, archive_name, archive_sha256, license, license_file, discovered_counts
def test_repo_contains_no_dataset_images(): T-DAT-08 — scan repo for .png/.jpg/.ppm outside .venv/data/out → none
```

- [x] **Step 2:** FAIL. **Step 3:** реализовать. **Step 4:** PASS; commit `feat: build-manifests and validate-data commands`.

### Task 20: `experiments plan` и `run`

**Files:**
- Create: `src/experiments/plan.py`, `tests/unit/test_plan_run.py`

**Interfaces:**
- Consumes: `protocol`, `config_hash`, `make_run_id`, модульные `main`.
- Produces:
  - `make_plan(cfg: Config, module_configs: Sequence[Path]) -> list[PlannedRun]`; `@dataclass(frozen=True) class PlannedRun(module, command, seed, config_path: str, config_hash, run_id)`; развёртка по `cfg.seeds` профиля; `write_plan(out_dir: Path, runs: Sequence[PlannedRun], cfg: Config) -> None` (`plan.json`; для confirmatory ещё `plan.lock.json` через `protocol.write_plan_lock` и `assert_plan_mutable`).
  - `run_plan(out_dir: Path, ...) -> int` — исполняет запланированные запуски через модульные `main`; confirmatory проверяет `check_run_in_plan`; сводный код — максимальный из кодов.

- [x] **Step 1: Failing-тесты**

```python
def test_plan_seeds_by_profile(...): pilot → {11,23,37}; confirmatory → all five
def test_plan_lock_written_for_confirmatory_only(...)
def test_plan_after_test_open_exits_5(...)
def test_run_executes_plan_with_fake_algorithm(...): all run dirs have _SUCCESS
def test_confirmatory_run_outside_plan_exits_5(...)
def test_run_ids_match_plan(...)
```

- [x] **Step 2:** FAIL. **Step 3:** реализовать. **Step 4:** PASS; commit `feat: experiments plan and run`.

### Task 21: `experiments aggregate` и `report`

**Files:**
- Create: `src/experiments/aggregate.py`, `src/experiments/report.py`, `src/expcore/reporting.py`, `tests/unit/test_aggregate_report.py`

**Interfaces:**
- Consumes: `metrics`, `read_predictions`, `validate_predictions`, `load_pair_evaluation`, `thresholds`.
- Produces:
  - `reporting.DISCLAIMER: str` (текст §1.6); `reporting.dataset_display_name(dataset_id: str, completeness: dict | None) -> str` («HPatches, поднабор `<имя>`» при `subset`).
  - `aggregate.aggregate(plan_dir: Path, out_path: Path) -> pd.DataFrame` — для каждого завершённого запуска: `validate_predictions(..., require_complete=True)` (иначе `DataValidationError`, код 3), метрики по `(method_id, relation_id, split, seed)`, строки несут `n_pairs, n_source_groups, n_seeds`; confirmatory-таблицы берут только `confirmatory/` и не `oracle_mode`; oracle-запуски — отдельная секция; пишет `metrics.parquet`.
  - `report.render_report(metrics: pd.DataFrame, plan: Sequence[PlannedRun], completeness: dict) -> str` — Markdown: оговорка, метрики, матрица 2×3 и покрытие по классам для каждого `(method, relation, split)`, подпись положительного класса и правила инверсии, секция «изменчивость по seeds»; отсутствие в отчёте завершённого запуска плана → `ProtocolViolationError`.

- [x] **Step 1: Failing-тесты** (`T-AGG-01`, `T-REP-01`)

```python
def test_aggregate_rejects_incomplete_predictions(...): Review Focus 2 — missing pair → exit 3, not counted as reject
def test_aggregate_counts_independent_units(...): n_source_groups < n_pairs, n_seeds == 3 for pilot plan
def test_oracle_runs_separated(...)
def test_report_has_disclaimer_matrix_and_class_coverage(...): DISCLAIMER in text; "reject" column present; coverage+/- present
def test_report_includes_worse_than_baseline_runs(...): RejectAllPredictor run appears (no filtering)
def test_report_fails_if_run_missing(...)
def test_hpatches_subset_named_in_report(...): T-DAT-07
def test_report_states_positive_class(...)
```

- [x] **Step 2:** FAIL. **Step 3:** реализовать. **Step 4:** PASS; commit `feat: aggregate and report`.

---

# Фаза 7. Примитивы, сквозные проверки, поставка

**Выход фазы:** сквозной smoke воспроизводим; комплект поставки собран; lock-файлы получены проверочной установкой.

### Task 22: Реестр библиотечных примитивов и контрольные проверки

**Files:**
- Create: `src/expcore/primitives.py`, `tests/unit/test_primitives.py`

**Interfaces:**
- Produces: `PRIMITIVES: tuple[dict, ...]` (`{name, library, check_id}`); `primitive_versions() -> list[dict]` (добавляет `version`); `run_check(check_id: str) -> None` (бросает `AssertionError` при расхождении). Проверки из §10: `PRIM-SIFT`, `PRIM-COLOR`, `PRIM-LINALG`, `PRIM-ADAM` (`tf`), `PRIM-BACKBONE` (`tf`).

- [x] **Step 1: Failing-тесты**

```python
@pytest.mark.parametrize("cid", ["PRIM-SIFT","PRIM-COLOR","PRIM-LINALG"])
def test_primitive_check_passes(cid): run_check(cid)
@pytest.mark.tf
def test_adam_first_step(): run_check("PRIM-ADAM")        # θ₁ = θ₀ − lr·ĝ/(|ĝ|+ε)
@pytest.mark.tf
def test_backbone_shape(): run_check("PRIM-BACKBONE")
def test_registry_lists_all_checks_and_versions_in_env(): collect_environment()["library_primitives"] covers PRIMITIVES
def test_tf_checks_skip_with_reason_when_missing(): ...
```

- [x] **Step 2:** FAIL. **Step 3:** реализовать проверки на синтетических данных: шахматная доска + поворот 90° для SIFT/`findHomography` (допуск на ошибку гомографии задать в коде константой с комментарием), D65-белая точка `L=100`, `lstsq`/SVD на известной системе. **Step 4:** PASS; commit `feat: primitives registry with control checks`.

### Task 23: Сквозной smoke и воспроизводимость

**Files:**
- Create: `tests/smoke/test_end_to_end.py`

**Interfaces:**
- Consumes: всё предыдущее; `tests/fakes.py`.

- [x] **Step 1: Failing-тесты** (`@pytest.mark.smoke`; `T-REP-02`, `T-DAT-05`, критерии приёмки §11.3 п.2, 3, 6)

```python
def test_pipeline_build_validate_plan_run_aggregate_report(tmp_path): all exit codes 0; report contains 2×3 matrix and DISCLAIMER
def test_two_fresh_process_runs_identical_predictions(tmp_path):
    h1 = run_smoke_in_subprocess(tmp_path/"a"); h2 = run_smoke_in_subprocess(tmp_path/"b"); assert h1 == h2
def test_corrupted_fixture_makes_validate_exit_3(tmp_path): ...
def test_oracle_marked_everywhere(tmp_path): run.json oracle_mode True, method_id endswith "+oracle", report section separate
def test_pilot_profile_cannot_touch_test_split_e2e(tmp_path): exit 5
```

- [x] **Step 2:** `pytest -m smoke -v` → FAIL на недостающих связках. **Step 3:** исправить только интеграционные расхождения (новых модулей не добавлять). **Step 4:** PASS; commit `test: end-to-end smoke and reproducibility`.

### Task 24: Схемы, шаблоны, README, DECISIONS

**Files:**
- Create: `scripts/gen_schemas.py`, `schemas/*.schema.json`, `configs/smoke/{photometric,geometric,metric,experiments}.yaml`, `README.md`, `DECISIONS.md`, `tests/unit/test_delivery.py`

**Interfaces:**
- Produces: `gen_schemas.generate(out_dir: Path) -> list[Path]` — JSON Schema файлов контракта (`datasets`, `images`, `pairs`, `predictions`, `run`, `thresholds`, `metrics`) из `expcore.schemas`.

- [x] **Step 1: Failing-тесты**

```python
def test_committed_schemas_match_generated(tmp_path): generate(tmp_path) == contents of repo schemas/
def test_readme_has_disclaimer_and_all_commands():   # T-REP-01, критерий §11.3 п.5
    text = Path("README.md").read_text(); assert DISCLAIMER in text and all(c in text for c in ALL_COMMANDS)
def test_decisions_has_all_D_entries(): D-01…D-20 present
def test_smoke_configs_load(): every configs/smoke/*.yaml loads with load_config
def test_full_template_loads(): configs/templates/full.yaml
```

- [x] **Step 2:** FAIL. **Step 3:** реализовать скрипт; написать README (установка, команды §8, порядок работы, оговорка) и `DECISIONS.md` (раздел 2 спеки); сгенерировать `schemas/`. **Step 4:** PASS; commit `docs: schemas, configs, README, decisions`.

### Task 25: Проверочная установка и lock-файлы

**Files:**
- Create: `requirements.lock`, `requirements-metric.lock`, `scripts/verify_lock.py`

- [x] **Step 1:** Чистое окружение A: `pip install -e ".[dev]"` → `pytest -m "unit or smoke" -v` → все PASS.
- [x] **Step 2:** Зафиксировать `pip freeze` в `requirements.lock` вместе со строкой с версией Python.
- [x] **Step 3:** Чистое окружение B: `pip install -e ".[dev,metric]"` → `pytest -m "unit or smoke or tf" -v`; если TensorFlow/Keras не ставится на выбранной версии Python или `PRIM-BACKBONE` падает — выбрать совместимую версию Python и повторить (результат не угадывается, а берётся из этого прогона); зафиксировать `requirements-metric.lock`.
- [x] **Step 4:** `scripts/verify_lock.py`: устанавливает lock в чистое окружение (`pip install --no-deps -r …`) и запускает `pytest -m "unit or smoke"`; коды: 0 — OK. Запустить обе проверки.
- [x] **Step 5:** Commit `chore: lock files from verified installs`; пройтись по критериям приёмки §11.3 п.1–6, результат записать в `DECISIONS.md` одной строкой.
