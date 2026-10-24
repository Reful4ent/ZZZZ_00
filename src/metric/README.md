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
   materialize("e2_k", Path("configs/metric"), selected=sel)   # pilot: e2_k, e3_nwarm, e6_noaug — планируются по отдельности
   materialize("confirmatory", Path("configs/metric"), selected=sel, n_cohort_images=N)  # N_cohort_all, классы с >=4 изображениями
   materialize("e7_degrade", Path("configs/metric"), selected=sel)   # ячейки ручных search-стадий E7
   ```
   Запись: `<exp_id>.yaml` (конфиг `experiments`) и `cells/<exp>/*.yaml`.
6. Confirmatory планируется **один раз**: `python -m experiments plan --config configs/metric/confirmatory.yaml`. Это единственный `plan.lock.json` профиля confirmatory (ТЗ 00, §7.5: один lock на (profile, protocol_version); повторный `plan` другого конфига заменил бы lock, а после открытия test изменение плана запрещено). В него входят все ячейки E1-confirm, E4-confirm, E4-fixed, E5; отдельные `e1_confirm.yaml`, `e4_confirm.yaml`, `e4_fixed.yaml`, `e5_limited.yaml` не планировать. `e4_fixed` в общем плане разворачивается по всем 5 seed (детерминированные повторы).
7. `python -m experiments run` — сначала завершить **всё** confirmatory-обучение; только затем `embed`/`search` на test. Повторный `metric train` (и `experiments run`) неудачного или прерванного запуска продолжает его с `checkpoints/last` (событие `auto_resume` в `run.log.jsonl`); после успешной попытки `failure.json` удаляется (история — в `errors.jsonl` и `run.json`).
8. Стадии обученных моделей (`embed`, `search`) запускаются вручную с **развёрнутым конфигом плана** для нужного seed: `<out_root>/<profile>/plan/<protocol_version>/configs/NN-metric-train-<ячейка>-sS.yaml` (запланированный train хеширует `seeds: {seed: s, seeds: null}`; конфиг ячейки даёт другой `run_id`, и обучающий запуск не находится). E7 — не элементы плана: это ручные `search`-стадии (`--gallery` без деградации, `--queries` с `params.degradation` из `cells/e7_degrade/`), после всего обучения. `search --pairs` с деградированными запросами запрещён (§9.7). `aggregate`/`report` над запусками P3 — работа P4.

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
