"""Mandatory P3 disclaimers (spec 1.6) for best_checkpoint.json and selected_config.json. TF-free."""
from expcore.reporting import DISCLAIMER

ALWAYS = (
    f"Общая оговорка ТЗ 00 (ТЗ00-спека §1.6): {DISCLAIMER}",
    "Базовый критерий — косинусная softmax cross-entropy (Г2-150), `loss_id=proxy_cos_softmax_v1`; это "
    "конкретизация общей модели главы 2, а не подтверждённая копия функции авторской рукописи.",
    "Численные настройки P3 — начальный проектный профиль, а не настройки, восстановленные из статьи.",
    "Эпоха — назначенное число равномерно семплированных пакетов, а не гарантированный однократный проход "
    "каждого изображения.",
    "Снижение Ldev само по себе не является успешным результатом.",
    "Выбор λ по Recall@1 не означает оптимальности бинарного порога.",
)
E5_NOTE = "Обучение при малом числе примеров на класс; не доказывает few-shot-адаптацию на новые классы."
BASE_MINIMAL_NOTE = ("Вариант без очередей измеряет стоимость обучения; его меньшие накладные "
                     "расходы не сравниваются с калибровкой без этого пояснения.")


def disclaimers(method_id: str, params: dict, subset: tuple[int, int, int] | None = None) -> list[str]:
    """subset: actual (classes, images, steps) of a run trained on a class subset (item 9)."""
    out = list(ALWAYS)
    if params["data"]["per_class_limit"] is not None:
        out.append(E5_NOTE)
    if method_id == "base_minimal":
        out.append(BASE_MINIMAL_NOTE)
    if params["data"]["class_subset"] is not None and subset is not None:
        out.append("Обучение на поднаборе классов: {} классов, {} изображений, {} шагов.".format(*subset))
    return out
