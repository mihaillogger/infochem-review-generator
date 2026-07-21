"""Модуль для валидации и лечения математических формул (LaTeX)."""


def validate_latex(latex_str: str) -> bool:
    """Проверяет баланс скобок и базовые аномалии OCR в LaTeX-формуле.

    Отсекает короткий мусор и проверяет целостность LaTeX-окружений.
    Игнорирует экранированные скобки (например, \\{).

    Args:
        latex_str (str): Строка с формулой.

    Returns:
        bool: True, если синтаксис (скобки и окружения) сбалансирован, иначе False.
    """
    if not latex_str or len(latex_str.strip()) < 4:
        return False

    if latex_str.count(r"\begin") != latex_str.count(r"\end"):
        return False

    stack: list[str] = []
    brackets = {"{": "}", "[": "]", "(": ")"}
    inverse_brackets = {v: k for k, v in brackets.items()}

    skip_next = False
    for char in latex_str:
        if skip_next:
            skip_next = False
            continue
        if char == "\\":
            skip_next = True
            continue

        if char in brackets:
            stack.append(char)
        elif char in inverse_brackets:
            if not stack:
                return False
            top = stack.pop()
            if top != inverse_brackets[char]:
                return False

    return len(stack) == 0


def fix_latex_brackets(latex_str: str) -> str:
    """Автоматически исправляет нехватку закрывающих скобок в конце строки.

    Также игнорирует экранированные скобки, чтобы не лечить то, что не сломано.

    Args:
        latex_str (str): Битая строка с формулой.

    Returns:
        str: Исправленная строка с добавленными в конец скобками.
    """
    stack: list[str] = []
    brackets = {"{": "}", "[": "]", "(": ")"}

    skip_next = False
    for char in latex_str:
        if skip_next:
            skip_next = False
            continue
        if char == "\\":
            skip_next = True
            continue

        if char in brackets:
            stack.append(char)
        elif char in brackets.values():
            if stack:
                stack.pop()

    fixed_str = latex_str
    while stack:
        unclosed = stack.pop()
        fixed_str += brackets[unclosed]

    return fixed_str
