import re

from pylatexenc.latexwalker import LatexWalker, LatexWalkerError  # type: ignore


def validate_latex(latex_str: str) -> bool:
    """Проверяет синтаксическую корректность LaTeX-формулы."""
    if not latex_str or len(latex_str.strip()) < 4:
        return False

    # 1. Жесткая проверка баланса фигурных скобок
    if latex_str.count("{") != latex_str.count("}"):
        return False

    # 2. Жесткая проверка совпадения имен окружений
    begins = re.findall(r"\\begin\{([^}]+)\}", latex_str)
    ends = re.findall(r"\\end\{([^}]+)\}", latex_str)
    if sorted(begins) != sorted(ends):
        return False

    try:
        walker = LatexWalker(latex_str)
        walker.get_latex_nodes()
        return True
    except (LatexWalkerError, Exception):
        return False
