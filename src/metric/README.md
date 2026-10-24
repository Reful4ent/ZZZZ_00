# metric (P3): прокси-метрическое обучение

Программа P3 по ТЗ 03: ResNet50 → Dense(128) → L2, косинусная softmax cross-entropy по прокси, калибровка очередями центров. Спека: `TZ_03-metric-proxy-design.md`; решения — `DECISIONS.md` (раздел «Решения P3»).

TensorFlow подключается только внутри обработчиков команд; `import metric` его не загружает. Окружение: `requirements-metric.lock` (TensorFlow 2.21.0, Keras 3.15.1).

## Команды

Все команды принимают общие аргументы ТЗ 00 (`--config`, `--data-root`, `--manifest-dir`, `--out-root`, `--protocol-registry`, `--profile`) и `--help`.

| Команда | Действие |
|---|---|
| `python -m metric train --config C` | Обучение по конфигу (запуск ТЗ 00). Лучший энкодер — `best_checkpoint.json`. |
| `python -m metric resume --checkpoint <run_dir>/checkpoints/last` | Продолжение с полного checkpoint; конфиг берётся из `config.resolved.yaml` запуска. |
| `python -m metric embed --config C --split {validation,test}` | Эмбеддинги split лучшим энкодером обучающего запуска (`params.degradation` — для E7). Для `imagenet_fixed`/`rgb_hist` `--split` необязателен: без него один базовый запуск встраивает все допустимые split. |
| `python -m metric search --config C --gallery G.npz --queries Q.npz [--pairs] [--thresholds T.json]` | Блочный поиск, Recall@k. С `--pairs` — `pair_scores.parquet` и пары в `predictions.parquet`; на validation выбирается `tD` (`thresholds.json`). На test с `--pairs` порог обязателен: `--thresholds` указывает на validation `thresholds.json`. Без `--pairs` в `predictions.parquet` попадают запросы. |
| `python -m metric select --config E` | `selected_config.json` по tuning-запускам; `E` — конфиг `experiments` сетки. |

## Порядок работы

1. `python -m experiments build-manifests --config …` — манифесты и пары. Пока `build-manifests` не импортирует `metric` (запрос №1 к ТЗ 00), пары P3 строятся только в процессе с `import metric`:
   `python -c "import metric, sys; from experiments.cli import main; sys.exit(main(['build-manifests', '--config', 'CONFIG.yaml']))"`.
2. `python -m experiments validate-data --config …`.
3. Pilot E1: `python -m experiments plan|run` над `configs/metric/e1_tune.yaml` (λ ∈ {0, 0.01, 0.1, 1}, три seed профиля pilot). Ячейки лежат в `configs/metric/cells/`.
4. `python -m metric select --config configs/metric/e1_tune.yaml` — выбор λ* по Recall@1 на validation. Аналогично для E4 (`e4_contrastive`, `e4_triplet`, `e4_classifier`).
5. Конфиги, зависящие от выбора (`e1_confirm`, `e2_k`, `e3_nwarm`, `e4_confirm`, `e5_limited`, `e6_noaug`, `e7_degrade`), создаёт код:
   ```python
   import json
   from pathlib import Path
   from metric.expconfigs import materialize
   sel = {e: json.loads(Path(f"<out_root>/pilot/metric/select/{e}/selected_config.json").read_text(encoding="utf-8"))
          for e in ("e1_tune", "e4_contrastive", "e4_triplet", "e4_classifier")}
   materialize("e1_confirm", Path("configs/metric"), selected=sel)
   materialize("e5_limited", Path("configs/metric"), selected=sel, n_cohort_images=N)  # N_cohort_all, классы с >=4 изображениями
   ```
   Запись: `<exp_id>.yaml` (конфиг `experiments`) и `cells/<exp_id>/*.yaml`.
6. `python -m experiments plan` для confirmatory пишет `plan.lock.json` (ТЗ 00, §7.5; test открывается только запусками из него), затем `run` для E2–E7. `aggregate`/`report` над запусками P3 — работа P4.

## Каталог запуска

`<out_root>/<profile>/metric/<run_id>/`. Базовый запуск — `train` (или `embed` нетренируемого метода) своего конфига. `embed` и `search` обученной модели — отдельные запуски ТЗ 00 со своим `run.json`; `stage.json` называет стадию (команда, split, degradation_id) и `base_run_id` обучающего запуска. `select` пишет в `<out_root>/<profile>/metric/select/<имя конфига>/selected_config.json`.

## Выходы (§13.2)

- train: `checkpoints/last/`, `checkpoints/best/` (в нём `encoder.keras`: backbone → dense, до L2), `training_steps/part-NNN.parquet`, `epochs.parquet`, `queue_state/epoch-NNN.parquet`, `batches.parquet`, `proxy_index.parquet`, `sampler_exclusions.parquet`, `best_checkpoint.json`, `memory.json`.
- embed: `embeddings.npz` (`embeddings`, `image_ids`, `weights_hash`, `split`, `degradation_id`, `method_id`, `d`).
- search: `retrieval.parquet`, `retrieval_summary.json`, `pair_scores.parquet` и `predictions.parquet` с `--pairs`, `thresholds.json` на validation.
- select: `selected_config.json`.
- общие файлы запуска ТЗ 00: `run.json`, `config.resolved.yaml`, `environment.json`, `rng_state.json`, `run.log.jsonl`, `errors.jsonl`, `timers.json`, `_SUCCESS`.

## Оговорки (§1.6)

1. Общая оговорка ТЗ 00 (ТЗ00-спека §1.6; текст — `expcore.reporting.DISCLAIMER`).
2. Базовый критерий — косинусная softmax cross-entropy (Г2-150), `loss_id=proxy_cos_softmax_v1`; это конкретизация общей модели главы 2, а не подтверждённая копия функции авторской рукописи.
3. Численные настройки P3 — начальный проектный профиль, а не настройки, восстановленные из статьи.
4. Эпоха — назначенное число равномерно семплированных пакетов, а не гарантированный однократный проход каждого изображения.
5. Снижение Ldev само по себе не является успешным результатом.
6. Выбор λ по Recall@1 не означает оптимальности бинарного порога.
7. Для E5: Обучение при малом числе примеров на класс; не доказывает few-shot-адаптацию на новые классы.
8. Для `base_minimal`: Вариант без очередей измеряет стоимость обучения; его меньшие накладные расходы не сравниваются с калибровкой без этого пояснения.
9. Если обучение шло на поднаборе классов: в метаданных указываются фактические число классов, изображений и шагов.

Тексты 1–9 попадают в поле `disclaimers` файлов `best_checkpoint.json` и `selected_config.json` (`metric.disclaimers`).

## Известные ограничения

- Открытые запросы к ТЗ 00 (`TZ_03-requests-to-TZ00.md`): `build-manifests` не импортирует `metric`; нет общего `prepare_run`; нет отдельного потока `degrade` (E7 использует `stream(seed, "augment", 2**31 + i)`).
- Интеграция с `experiments aggregate` — работа P4: запуски `train` не несут predictions, стадийные запуски не входят в `plan`.
- Детерминизм на GPU не проверен; проверки выполнены на CPU.
- TensorFlow 2.21 на нативной Windows не поддерживает GPU; обучение на реальном SOP — на Linux/WSL2 с GPU. Pilot на реальном SOP не выполнялся (см. `DECISIONS.md`); оценка GPU-часов спеки §10 остаётся прогнозом.
- Версии baseline-методов `<method>_v1` — соглашение `metric.expconfigs`.
