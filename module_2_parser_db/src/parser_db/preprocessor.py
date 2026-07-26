"""Модуль предобработки распарсенного текста перед чанкингом."""

import re
from typing import TypedDict

import structlog
from transformers import AutoTokenizer

from parser_db.config import settings
from parser_db.schemas import Paragraph, VisualMeta

logger = structlog.get_logger(__name__)

tokenizer = AutoTokenizer.from_pretrained(settings.EMBEDDING_MODEL_NAME, trust_remote_code=True)


class SandwichBlock(TypedDict):
    text: str
    is_sandwich: bool
    contains_table: bool
    contains_math: bool
    raw_table_markup: str | None
    raw_math_markup: list[str] | None
    is_broken_table: bool
    is_broken_math: bool
    fallback_table_path: str | None
    fallback_math_path: str | None


def count_tokens(text: str) -> int:
    """
    Подсчитывает точное количество токенов в тексте.

    Args:
        text: Исходный текст.

    Returns:
        Количество токенов согласно словарю модели.
    """
    return len(tokenizer.encode(text, add_special_tokens=False))


def split_recursively(text: str, max_tokens: int) -> tuple[list[str], bool]:
    """
    Классический рекурсивный сплиттер для огромных кусков текста,
    которые не влезают в окно контекста модели.

    Args:
        text (str): Исходный огромный текст.
        max_tokens (int): Максимально допустимое число токенов.

    Returns:
        tuple[list[str], bool]: Список разрезанных кусков, каждый из которых <= max_tokens
        и флаг (True, если пришлось резать по токенам).
    """
    if count_tokens(text) <= max_tokens:
        return [text], False

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
                return result, False

    # Если даже по пробелам не бьется, режем по токенам
    logger.warning("fallback_token_split_used", text_length=len(text))

    tokens = tokenizer.encode(text)
    chunks = [tokens[i : i + max_tokens] for i in range(0, len(tokens), max_tokens)]

    return [str(tokenizer.decode(chunk)) for chunk in chunks], True


def build_sandwiches(paragraphs: list[Paragraph]) -> list[SandwichBlock]:
    """
    Группирует параграфы методом 'Сэндвича', склеивая таблицы и формулы
    с соседними текстовыми блоками.

    Args:
        paragraphs (list[Paragraph]): Список параграфов секции.

    Returns:
        list[SandwichBlock]: Список сформированных блоков с предварительной разметкой.
    """
    blocks: list[SandwichBlock] = []
    skip_next = False

    for i, para in enumerate(paragraphs):
        if skip_next:
            skip_next = False
            continue

        if para.type in ["table", "equation"]:
            sandwich_text = []
            meta: SandwichBlock = {
                "text": "",
                "is_sandwich": True,
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

            text_to_embed = para.enriched_summary if para.enriched_summary else para.content
            sandwich_text.append(text_to_embed)

            # Добавляем следующий абзац
            if i < len(paragraphs) - 1 and paragraphs[i + 1].type == "text":
                sandwich_text.append(paragraphs[i + 1].content)
                skip_next = True

            meta["text"] = "\n\n".join(sandwich_text)
            blocks.append(meta)
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


def build_visuals_patterns(document_visuals: list[VisualMeta]) -> dict[str, re.Pattern[str]]:
    """
    Один раз компилирует регулярные выражения для всех картинок документа.

    Args:
        document_visuals: Список визуальных элементов документа.

    Returns:
        dict: Словарь вида {'Fig. 1': скомпилированный_паттерн}
    """
    patterns = {}
    for visual in document_visuals:
        escaped_id = re.escape(visual.id).replace(r"\ ", r"\s*")
        patterns[visual.id] = re.compile(rf"\b{escaped_id}\b", re.IGNORECASE)

    return patterns


def extract_visual_ids(text: str, patterns: dict[str, re.Pattern[str]]) -> set[str]:
    """
    Ищет ссылки на иллюстрации по заранее скомпилированным паттернам.

    Args:
        text: Текст текущего чанка/блока.
        patterns: Словарь скомпилированных паттернов от build_visuals_patterns.

    Returns:
        set: Множество найденных точных ID (например, {'Fig. 1', 'Table 2'}).
    """
    found_ids: set[str] = set()

    if not text:
        return found_ids

    for exact_id, pattern in patterns.items():
        if pattern.search(text):
            found_ids.add(exact_id)

    return found_ids
