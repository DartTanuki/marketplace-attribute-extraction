from marketplace_attribute_extraction.shortlist import select_attribute_shortlist


def test_ram_hint_enters_shortlist():
    attributes = [
        "Бренд",
        "Модель",
        "Страна",
        "Встроенная память (ROM)",
        "Оперативная память (RAM)",
        "Емкость аккумулятора",
    ]
    selected, _ = select_attribute_shortlist(
        "Samsung Galaxy S26 8 ГБ оперативы",
        "16434",
        attributes,
        shortlist_size=5,
    )
    assert "Оперативная память (RAM)" in selected
    assert "Бренд" in selected
    assert "Модель" in selected


def test_full_schema_mode():
    attributes = ["Бренд", "Модель", "Цвет"]
    selected, ranking = select_attribute_shortlist(
        "черный телефон",
        "1",
        attributes,
        shortlist_size=None,
    )
    assert selected == attributes
    assert all(row["reasons"] == ["full_schema"] for row in ranking)
