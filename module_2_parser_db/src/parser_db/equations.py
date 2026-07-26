"""Модуль для валидации математических формул (LaTeX)."""

import re


def validate_latex(latex_str: str) -> bool:
    """Проверяет баланс скобок и окружений в LaTeX-формуле.

    Отсекает короткий мусор и проверяет целостность LaTeX-окружений
    с помощью стека. Игнорирует экранированные скобки.

    Args:
        latex_str: Строка с формулой.

    Returns:
        True, если синтаксис (скобки и окружения) сбалансирован, иначе False.
    """
    if not latex_str or len(latex_str.strip()) < 4:
        return False

    # Валидация окружений \begin{} ... \end{}
    env_stack: list[str] = []
    env_pattern = re.compile(r"\\(?:begin|end)\{([^}]+)\}")
    
    for match in env_pattern.finditer(latex_str):
        full_match = match.group(0)
        env_name = match.group(1)
        if full_match.startswith(r"\begin"):
            env_stack.append(env_name)
        elif full_match.startswith(r"\end"):
            if not env_stack or env_stack[-1] != env_name:
                return False
            env_stack.pop()

    if env_stack:
        return False

    # Валидация фигурных, квадратных и круглых скобок
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