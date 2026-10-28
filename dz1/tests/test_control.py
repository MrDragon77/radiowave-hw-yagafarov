"""Проверка контрольного варианта 0.

Поменяйте DZ на своё задание (dz1 ... dz6). Тест сравнивает ваши результаты с ответами
из текста задания. Допуски (rtol, atol) записаны в control_answers.json.
Если вы решаете не на Python, сравните таблицу результатов вручную и этот тест не запускайте.
"""
import json
import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from solution import solve  # noqa: E402

DZ = "dz1"

DATA = json.loads(pathlib.Path(__file__).with_name("control_answers.json").read_text(encoding="utf-8"))[DZ]


@pytest.fixture(scope="module")
def result():
    return solve(DZ, 0, DATA.get("inputs"))


@pytest.mark.parametrize("a", DATA["answers"], ids=lambda a: a["key"])
def test_control(result, a):
    assert a["key"] in result, f"В результате нет ключа {a['key']} ({a['desc']})"
    got = result[a["key"]]
    if isinstance(a["value"], str):
        assert got == a["value"]
    else:
        assert math.isclose(got, a["value"], rel_tol=a["rtol"], abs_tol=a["atol"]), \
            f"{a['desc']}: ожидалось {a['value']} {a['unit']}, получено {got}"
