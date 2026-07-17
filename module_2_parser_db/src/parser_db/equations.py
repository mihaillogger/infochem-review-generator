"""
Модуль для валидации и лечения математических формул (LaTeX).
"""

def validate_latex(latex_str: str) -> bool:
    """
    Проверяет баланс скобок в LaTeX-формуле.

    Args:
        latex_str (str): Строка с формулой.

    Returns:
        bool: True, если синтаксис (скобки) сбалансирован, иначе False.
    """
    stack = []
    brackets = {'{': '}', '[': ']', '(': ')'}
    inverse_brackets = {v: k for k, v in brackets.items()}
    
    for char in latex_str:
        if char in brackets:
            stack.append(char)
        elif char in inverse_brackets:
            if not stack:
                return False  # Закрывающая скобка без открывающей
            top = stack.pop()
            if top != inverse_brackets[char]:
                return False  # Несовпадение типов скобок
                
    return len(stack) == 0


def fix_latex_brackets(latex_str: str) -> str:
    """
    Пытается автоматически исправить нехватку закрывающих скобок в конце строки.
    (Частая галлюцинация OCR-моделей).

    Args:
        latex_str (str): Битая строка с формулой.

    Returns:
        str: Исправленная строка.
    """
    stack = []
    brackets = {'{': '}', '[': ']', '(': ')'}
    
    # Собираем все незакрытые скобки
    for char in latex_str:
        if char in brackets:
            stack.append(char)
        elif char in brackets.values():
            if stack:
                stack.pop()
                
    # Дописываем недостающие скобки в конец
    fixed_str = latex_str
    while stack:
        unclosed = stack.pop()
        fixed_str += brackets[unclosed]
        
    return fixed_str