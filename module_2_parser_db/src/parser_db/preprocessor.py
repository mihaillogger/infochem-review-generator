"""Модуль предобработки распарсенного текста перед чанкингом."""

import re
from typing import Any

from transformers import AutoTokenizer

from parser_db.config import settings
from parser_db.schemas import Paragraph, VisualMeta

tokenizer = AutoTokenizer.from_pretrained(settings.EMBEDDING_MODEL_NAME, trust_remote_code=True)


def count_tokens(text: str) -> int:
    """
    Подсчитывает точное количество токенов в тексте.

    Args:
        text: Исходный текст.

    Returns:
        Количество токенов согласно словарю модели.
    """
    return len(tokenizer.encode(text))


def split_recursively(text: str, max_tokens: int) -> list[str]:
    """
    Классический рекурсивный сплиттер для огромных кусков текста,
    которые не влезают в окно контекста модели.

    Args:
        text (str): Исходный огромный текст.
        max_tokens (int): Максимально допустимое число токенов.

    Returns:
        list[str]: Список разрезанных кусков, каждый из которых <= max_tokens.
    """
    if count_tokens(text) <= max_tokens:
        return [text]

    separators = ["\n\n", "\n", ". ", " "]
    for sep in separators:
        parts = text.split(sep)
        if len(parts) > 1:
            result = []
            current_piece = ""
            for part in parts:
                part_text = part + sep if sep != " " else part

                if count_tokens(current_piece + part_text) > max_tokens:
                    if current_piece:
                        result.append(current_piece.strip())
                    current_piece = part_text
                else:
                    current_piece += part_text

            if current_piece:
                result.append(current_piece.strip())

            # Проверяем, удалось ли успешно разбить текст
            if all(count_tokens(p) <= max_tokens for p in result):
                return result

    # Если даже по пробелам не бьется, режем по токенам
    tokens = tokenizer.encode(text)
    chunks = [tokens[i : i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [str(tokenizer.decode(chunk)) for chunk in chunks]


def build_sandwiches(paragraphs: list[Paragraph]) -> list[dict[str, Any]]:
    """
    Группирует параграфы методом 'Сэндвича', склеивая таблицы и формулы
    с соседними текстовыми блоками.

    Args:
        paragraphs (list[Paragraph]): Список параграфов секции.

    Returns:
        list[dict]: Список сформированных блоков с предварительной разметкой.
    """
    blocks: list[dict[str, Any]] = []
    skip_next = False

    for i, para in enumerate(paragraphs):
        if skip_next:
            skip_next = False
            continue

        if para.type in ["table", "equation"]:
            sandwich_text = []
            meta = {
                "contains_table": para.type == "table",
                "contains_math": para.type == "equation",
                "raw_table_markup": para.content if para.type == "table" else None,
                "raw_math_markup": [para.content] if para.type == "equation" else None,
                "is_broken_table": para.is_broken and para.type == "table",
                "is_broken_math": para.is_broken and para.type == "equation",
                "fallback_table_path": para.image_fallback_path
                if (para.is_broken and para.type == "table")
                else None,
                "fallback_math_path": para.image_fallback_path
                if (para.is_broken and para.type == "equation")
                else None,
            }

            # Добавляем предыдущий абзац
            if i > 0 and paragraphs[i - 1].type == "text" and not blocks[-1].get("is_sandwich"):
                sandwich_text.append(str(blocks.pop()["text"]))

            sandwich_text.append(para.content)

            # Добавляем следующий абзац
            if i < len(paragraphs) - 1 and paragraphs[i + 1].type == "text":
                sandwich_text.append(paragraphs[i + 1].content)
                skip_next = True

            blocks.append({"text": "\n\n".join(sandwich_text), "is_sandwich": True, **meta})
        else:
            blocks.append(
                {
                    "text": para.content,
                    "is_sandwich": False,
                    "contains_table": False,
                    "contains_math": False,
                    "raw_table_markup": None,
                    "raw_math_markup": None,
                    "is_broken_table": False,
                    "is_broken_math": False,
                    "fallback_table_path": None,
                    "fallback_math_path": None,
                }
            )

    return blocks


def extract_validated_visuals(text: str, document_visuals: list[VisualMeta]) -> set[str]:
    """
    Ищет в тексте ссылки только на те иллюстрации, которые реально существуют
    в метаданных текущего документа.
    """
    found_ids = set()

    for visual in document_visuals:
        # Экранируем спецсимволы
        escaped_id = re.escape(visual.id).replace(r"\ ", r"\s*")

        pattern = re.compile(rf"\b{escaped_id}\b", re.IGNORECASE)

        if pattern.search(text):
            found_ids.add(visual.id)

    return found_ids
