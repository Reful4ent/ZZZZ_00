# Экспериментальный комплекс: общий протокол (ТЗ 00)

> Численные настройки являются назначенным проектным протоколом (редакция 1.0 от 23.10.2026), а не опубликованными результатами или доказанно оптимальными параметрами. Реализация выполнена с нуля по главам 1 и 2; совпадение с неизвестной реализацией авторских статей не заявляется.

Если HPatches неполон, везде вместо «HPatches» указывается «HPatches, поднабор `<имя>`».

Пакеты: `expcore` (общий протокол), `photometric` (P1), `geometric` (P2), `metric` (P3), `experiments` (P4).
Алгоритмы программ подключаются через реестр (`expcore.registry.register_algorithm`); без них команды
модулей — каркасы и завершаются кодом 6. Спецификация — `2026-10-24-common-protocol-design.md`,
журнал решений — `DECISIONS.md`.

## Установка

```bash
python -m venv .venv
# Linux/macOS: . .venv/bin/activate      Windows: .venv\Scripts\activate
pip install -r requirements.lock --no-deps && pip install -e . --no-deps          # CPU-профиль
pip install -r requirements-metric.lock --no-deps && pip install -e . --no-deps   # с TensorFlow (P3)
```

Без lock-файлов: `pip install -e ".[dev]"` (и `.[dev,metric]` для TensorFlow; `.[mem]` — пиковая память через psutil).
Проверка: `pytest -m "unit or smoke"` (тесты `tf` пропускаются с причиной, если TensorFlow не установлен).

## Команды

У всех команд: `--config PATH` (обязателен), необязательные `--data-root`, `--manifest-dir`, `--out-root`
(пути не входят в `config_hash`), `--protocol-registry` (по умолчанию `configs/protocol_registry.json`);
`--help` возвращает 0. Профиль, seed и параметры задаются только в конфиге.

| Команда | Назначение |
|---|---|
| `python -m experiments build-manifests --config …` | Сканирует данные, разбивает, проверяет дубликаты, замораживает манифест (`--overrides overrides.yaml` — `exclude_image`) |
| `python -m experiments validate-data --config …` | Проверяет замороженный манифест: хэш, файлы и SHA-256, схемы, повторное разбиение, пересечения частей |
| `python -m experiments plan --config …` | Матрица запусков по seeds профиля (`plan.json`; для confirmatory — `plan.lock.json`) |
| `python -m experiments run --config …` | Выполняет план через команды модулей |
| `python -m experiments aggregate --config …` | Метрики §6 по завершённым запускам → `metrics.parquet` |
| `python -m experiments report --config …` | Markdown-отчёт по всем завершённым запускам плана |
| `python -m photometric run --config …` | P1 (каркас) |
| `python -m geometric run --config …` | P2 (каркас) |
| `python -m metric train --config …` | P3 (каркас; TensorFlow загружается лениво) |
| `python -m metric embed --config …` | P3 (каркас) |
| `python -m metric search --config …` | P3 (каркас) |

Коды завершения: 0 — успех, 2 — конфигурация/аргументы, 3 — данные (`DataValidationError`), 4 — техническая
ошибка, 5 — нарушение протокола, 6 — алгоритм не зарегистрирован.

## Порядок работы

`build-manifests → validate-data → pilot (plan → run → aggregate → report) → plan (confirmatory) → run → aggregate → report`

1. Данные раскладываются в `<data_root>/kadid10k`, `hpatches`, `mvtec_ad`, `sop` (раскладки поставщиков).
2. Манифест замораживается до любых результатов; исправление — только новой версией (`manifest_version` + `--overrides`).
3. Pilot работает на `train/dev/validation`; test недоступен.
4. Confirmatory запускается только по замороженному плану; первое обращение к test записывается
   (`test_opened.json`), после этого план версии менять нельзя — изменения дают новую версию протокола
   (`configs/protocol_registry.json`), а её результаты помечаются как отдельное исследование.
5. Завершённые результаты не перезаписываются (флага `--force` нет); прерванный запуск продолжается тем же `run_id`.

## Smoke на синтетических данных

```bash
python -m tests.fixtures_builder data/smoke
python -m experiments build-manifests --config configs/smoke/experiments.yaml
python -m experiments validate-data   --config configs/smoke/experiments.yaml
python -m experiments plan            --config configs/smoke/experiments.yaml
python -m experiments run             --config configs/smoke/experiments.yaml   # 6, пока нет алгоритмов
```

## Поставка

`configs/` — шаблон `templates/full.yaml` (все ключи обязательны), smoke-конфиги, реестр версий протокола;
`schemas/` — JSON Schema файлов контракта, генерируются `python scripts/gen_schemas.py`; данные и результаты
в репозиторий не входят и передаются путями.
