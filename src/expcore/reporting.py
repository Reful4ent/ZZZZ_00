"""Fixed report texts (spec §1.6, §4.3)."""

DISCLAIMER = ("Численные настройки являются назначенным проектным протоколом (редакция 1.0 от 23.10.2026), "
              "а не опубликованными результатами или доказанно оптимальными параметрами. Реализация выполнена "
              "с нуля по главам 1 и 2; совпадение с неизвестной реализацией авторских статей не заявляется.")

DATASET_NAMES = {"kadid10k": "KADID-10k", "hpatches": "HPatches", "mvtec_ad": "MVTec AD",
                 "sop": "Stanford Online Products"}


def dataset_display_name(dataset_id: str, completeness: dict | None) -> str:
    """An incomplete HPatches delivery is always named as a subset."""
    name = DATASET_NAMES.get(dataset_id, dataset_id)
    if completeness and completeness.get("status") == "subset":
        return f"{name}, поднабор `{completeness.get('subset_name')}`"
    return name
