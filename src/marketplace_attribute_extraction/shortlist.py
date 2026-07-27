"""Dynamic attribute shortlist for marketplace queries."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

GENERIC_ATTRIBUTES = ("Бренд", "Модель")

SHORTLIST_STOPWORDS = {
    "и", "или", "для", "на", "в", "с", "по", "из", "от",
    "тип", "наличие", "поддержка", "режим", "функция",
    "характеристика", "значение", "максимальный", "минимальный",
}

# query regex, attribute regex, boost, diagnostic label
ATTRIBUTE_QUERY_HINT_RULES = [
    (r"\b(оператив\w*|озу|ram)\b", r"оператив|\bram\b", 45.0, "RAM hint"),
    (r"\b(встроенн\w*\s+памят\w*|rom|накопител\w*|ssd|hdd)\b",
     r"встроенн.*памят|\brom\b|накопител|диск", 45.0, "storage hint"),
    (r"\b(аккумулятор\w*|батаре\w*|мач|mah)\b",
     r"аккумулятор|батаре|емкость", 40.0, "battery hint"),
    (r"\b(диагонал\w*|дюйм\w*|inch|inches)\b",
     r"диагонал|размер.*экран", 40.0, "diagonal hint"),
    (r"\b(разрешени\w*|4k|8k|full\s*hd|uhd|qhd|\d{3,4}\s*[xх×]\s*\d{3,4})\b",
     r"разрешени", 40.0, "resolution hint"),
    (r"\b(цвет\w*|черн\w*|бел\w*|красн\w*|син\w*|сер\w*|зелен\w*|золот\w*)\b",
     r"цвет", 32.0, "color hint"),
    (r"\b(вес\w*|масса)\b", r"вес|масса", 38.0, "weight hint"),
    (r"\b(высот\w*)\b", r"высот", 38.0, "height hint"),
    (r"\b(ширин\w*)\b", r"ширин", 38.0, "width hint"),
    (r"\b(глубин\w*)\b", r"глубин", 38.0, "depth hint"),
    (r"\b(длин\w*)\b", r"длин", 38.0, "length hint"),
    (r"\b(диаметр\w*)\b", r"диаметр", 38.0, "diameter hint"),
    (r"\b(мощност\w*|ватт\w*|киловатт\w*)\b",
     r"мощност|потребляем", 38.0, "power hint"),
    (r"\b(частот\w*|герц\w*|гц|hz)\b",
     r"частот|обновлен", 38.0, "frequency hint"),
    (r"\b(камера\w*|мегапиксел\w*|мп)\b",
     r"камер|мегапиксел|разрешение.*фото", 36.0, "camera hint"),
    (r"\b(sim|сим)\b", r"sim|сим", 35.0, "SIM hint"),
    (r"\b(процессор\w*|cpu|чип\w*)\b",
     r"процессор|чипсет", 35.0, "CPU hint"),
    (r"\b(ядр\w*)\b", r"ядер|ядр", 35.0, "core-count hint"),
    (r"\b(объем\w*|обьем\w*|литр\w*)\b",
     r"объем|обьем|вместимост", 28.0, "volume hint"),
    (r"\b(гаранти\w*)\b", r"гаранти", 35.0, "warranty hint"),
    (r"\b(страна|производств\w*)\b",
     r"страна|производств", 30.0, "country hint"),
]

ATTRIBUTE_UNIT_HINT_RULES = [
    (r"(?<!\w)\d+(?:[.,]\d+)?\s*(?:тб|tb|гб|gb|мб|mb)(?!\w)",
     r"памят|\bram\b|\brom\b|накопител|видеопамят|диск", 28.0, "memory unit"),
    (r"(?<!\w)\d+(?:[.,]\d+)?\s*(?:мач|mah)(?!\w)",
     r"аккумулятор|батаре|емкость", 42.0, "battery unit"),
    (r"(?<!\w)\d+(?:[.,]\d+)?\s*(?:гц|hz)(?!\w)",
     r"частот|обновлен", 34.0, "display frequency unit"),
    (r"(?<!\w)\d+(?:[.,]\d+)?\s*(?:кгц|khz|мгц|mhz|ггц|ghz)(?!\w)",
     r"частот|процессор|cpu", 32.0, "processor frequency unit"),
    (r"(?<!\w)\d+(?:[.,]\d+)?\s*[\"″]",
     r"диагонал|размер.*экран", 40.0, "inch quote"),
    (r"(?<!\w)\d+(?:[.,]\d+)?\s*(?:вт|w|квт|kw)(?!\w)",
     r"мощност|потребляем", 35.0, "power unit"),
    (r"(?<!\w)\d+(?:[.,]\d+)?\s*(?:кг|kg|г|гр|gram)(?!\w)",
     r"вес|масса|загрузк", 28.0, "mass unit"),
    (r"(?<!\w)\d+(?:[.,]\d+)?\s*(?:мм|mm|см|cm|м|meter)(?!\w)",
     r"высот|ширин|глубин|длин|диаметр|размер|толщин", 24.0, "length unit"),
    (r"(?<!\w)\d+(?:[.,]\d+)?\s*(?:л|литр|ml|мл)(?!\w)",
     r"объем|обьем|вместимост|резервуар", 28.0, "volume unit"),
    (r"(?<!\w)\d+(?:[.,]\d+)?\s*(?:об/мин|rpm)(?!\w)",
     r"оборот|скорост|отжим", 34.0, "rotation unit"),
    (r"(?<!\w)\d+(?:[.,]\d+)?\s*(?:мп|mp)(?!\w)",
     r"камер|мегапиксел|разрешение.*фото", 38.0, "camera unit"),
]


def normalize_shortlist_text(value: Any) -> str:
    value = unicodedata.normalize("NFKC", str(value or ""))
    value = value.lower().replace("ё", "е")
    value = re.sub(r"[^\w\d]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def shortlist_tokens(value: Any) -> set[str]:
    return {
        token
        for token in normalize_shortlist_text(value).split()
        if len(token) >= 3 and token not in SHORTLIST_STOPWORDS
    }


def rank_attributes_for_query(
    query: str,
    category_id: str,
    attributes: Iterable[str],
    attribute_priority_lookup: dict[str, dict[str, dict[str, float]]] | None = None,
) -> list[dict[str, Any]]:
    """Rank category attributes using priors and lexical/unit hints."""
    query_raw = str(query)
    query_normalized = normalize_shortlist_text(query_raw)
    query_token_set = shortlist_tokens(query_raw)
    priors = (attribute_priority_lookup or {}).get(str(category_id), {})

    rows: list[dict[str, Any]] = []
    for list_position, attribute_name in enumerate(attributes, start=1):
        attribute_normalized = normalize_shortlist_text(attribute_name)
        attribute_token_set = shortlist_tokens(attribute_name)
        prior = priors.get(attribute_name, {})
        rank = int(prior.get("rank", list_position))
        statistical_score = float(prior.get("score", 0.0))
        coverage = float(prior.get("coverage", 0.0))

        score = 3.0 / max(rank, 1) + 2.0 * statistical_score + 0.5 * coverage
        reasons = [f"category_prior_rank={rank}"]

        if attribute_name in GENERIC_ATTRIBUTES:
            score += 100.0
            reasons.append("mandatory_generic")

        overlap = query_token_set & attribute_token_set
        if overlap:
            score += 8.0 + 3.0 * len(overlap)
            reasons.append("token_overlap=" + ",".join(sorted(overlap)))

        for query_pattern, attribute_pattern, boost, reason in ATTRIBUTE_QUERY_HINT_RULES:
            if (
                re.search(query_pattern, query_normalized, flags=re.IGNORECASE)
                and re.search(attribute_pattern, attribute_normalized, flags=re.IGNORECASE)
            ):
                score += boost
                reasons.append(reason)

        for query_pattern, attribute_pattern, boost, reason in ATTRIBUTE_UNIT_HINT_RULES:
            if (
                re.search(query_pattern, query_raw, flags=re.IGNORECASE)
                and re.search(attribute_pattern, attribute_normalized, flags=re.IGNORECASE)
            ):
                score += boost
                reasons.append(reason)

        rows.append(
            {
                "attribute": attribute_name,
                "score": float(score),
                "category_rank": rank,
                "list_position": list_position,
                "reasons": reasons,
            }
        )

    return sorted(
        rows,
        key=lambda row: (-row["score"], row["category_rank"], row["list_position"]),
    )


def select_attribute_shortlist(
    query: str,
    category_id: str,
    attributes: Iterable[str],
    shortlist_size: int | None = 5,
    attribute_priority_lookup: dict[str, dict[str, dict[str, float]]] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Select GLiNER2 labels while preserving the original schema order."""
    unique_attributes = list(dict.fromkeys(attributes or []))

    if shortlist_size is None or shortlist_size >= len(unique_attributes):
        return unique_attributes, [
            {
                "attribute": attribute,
                "score": None,
                "category_rank": position,
                "list_position": position,
                "reasons": ["full_schema"],
            }
            for position, attribute in enumerate(unique_attributes, start=1)
        ]

    shortlist_size = int(shortlist_size)
    if shortlist_size <= 0:
        raise ValueError("shortlist_size must be positive or None")

    ranking = rank_attributes_for_query(
        query=query,
        category_id=str(category_id),
        attributes=unique_attributes,
        attribute_priority_lookup=attribute_priority_lookup,
    )
    selected_set = {row["attribute"] for row in ranking[:shortlist_size]}
    selected = [attribute for attribute in unique_attributes if attribute in selected_set]
    return selected, ranking
