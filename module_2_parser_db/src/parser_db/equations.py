"""Модуль для валидации математических формул (LaTeX)."""

from pylatexenc.latexwalker import LatexWalker, LatexWalkerError  # type: ignore


def validate_latex(latex_str: str) -> bool:
    """Проверяет синтаксическую корректность LaTeX-формулы.

    Использует AST-парсер для точной проверки баланса окружений,
    математических блоков и корректности экранирования.
Ц
    Args:
        latex_str: Строка с формулой.

    Returns:
        True, если синтаксис формулы корректен, иначе False.
    """
    if not latex_str or len(latex_str.strip()) < 4:
        return False

    try:
        walker = LatexWalker(latex_str)
        # Попытка построить узлы. Упадет с ошибкой, если синтаксис сломан.
        walker.get_latex_nodes()
        return True
    except LatexWalkerError:
        return False
    except Exception:
        # Отлов непредвиденных ошибок, чтобы парсинг статьи не крашнулся
        return False
