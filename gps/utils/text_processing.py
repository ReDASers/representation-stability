"""
Text processing utilities for adversarial detection.

This module provides functions for tokenizing text, mapping between sub-tokens and words,
and preprocessing text to fit within model token limits. It supports various HuggingFace
tokenizers and handles edge cases like special tokens and subword tokenization.
"""
from __future__ import annotations

import logging
import re
import string
from functools import lru_cache
from typing import Dict, List, Tuple, Union

import nltk
from transformers import PreTrainedTokenizer

logger = logging.getLogger("adversarial_detector")


# Constants for text processing
WORD_LIKE_PATTERN = re.compile(
    r"[A-Za-z\u00C0-\u017F0-9]+(?:[.\-'][A-Za-z\u00C0-\u017F0-9]+)*"
)
"""Pattern to identify word-like tokens: letters (including accents), digits, and internal punctuation."""

PUNCT_TO_STRIP = "".join(c for c in string.punctuation if c not in {".", "-", "'"})
"""Characters to strip from the beginning and end of token candidates."""

_UNKNOWN_LEN_SENTINEL = 1000000000000000019884624838656
"""Sentinel value used by HuggingFace tokenizers to indicate unknown max_length."""

_SPECIAL_TOKENS_OFFSET_MODELS = {"bert", "roberta"}
"""Model types that include special tokens affecting offset mapping calculations."""


def tokenize_texts(
    texts: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int | None = None,
):
    """
    Tokenize a list of input texts using the provided HuggingFace tokenizer.

    Parameters
    ----------
    texts : List[str]
        Input strings to tokenize.
    tokenizer : PreTrainedTokenizer
        HuggingFace tokenizer instance.
    max_length : int, optional
        Maximum length for truncation. If None, uses tokenizer.model_max_length.

    Returns
    -------
    BatchEncoding
        Tokenized inputs with dynamic padding and truncation applied.
    """
    if max_length is None and tokenizer.model_max_length not in (None, _UNKNOWN_LEN_SENTINEL):
        max_length = tokenizer.model_max_length
        logger.debug("tokenize_texts: using tokenizer.model_max_length=%s", max_length)

    return tokenizer(
        texts,
        truncation=True,
        padding="longest",
        max_length=max_length,
        return_tensors="pt",
    )


def split_into_words(text: str) -> List[str]:
    """
    Extract word-like tokens from text, preserving internal punctuation.

    Parameters
    ----------
    text : str
        Input string to split into words.

    Returns
    -------
    List[str]
        Word-like substrings extracted from the input text.
    """
    if not text:
        return []

    # Extract candidate substrings and their positions
    candidates = (
        (m.group().strip(PUNCT_TO_STRIP), m.start())
        for m in WORD_LIKE_PATTERN.finditer(text)
    )
    
    # Keep order and drop empty results
    words_with_pos = [(w, pos) for w, pos in candidates if w]
    words_with_pos.sort(key=lambda t: t[1])
    
    return [w for w, _ in words_with_pos]


def map_subtokens_to_words(
    text: str,
    tokenizer: PreTrainedTokenizer,
    *,
    return_token_to_word: bool = False,
    max_length: int | None = None,
) -> Tuple[List[str], Union[Dict[int, List[int]], Dict[int, int]]]:
    """
    Map between sub-token indices and word indices for tokenized text.

    Uses tokenizer offset mappings when available (fast path) or manual reconstruction 
    (slow path) to build mappings between token and word indices.

    Parameters
    ----------
    text : str
        Input string to tokenize and map.
    tokenizer : PreTrainedTokenizer
        HuggingFace tokenizer instance.
    return_token_to_word : bool, default False
        If True, return token-to-word mapping. Otherwise, return word-to-tokens mapping.
    max_length : int, optional
        Maximum number of tokens for truncation.

    Returns
    -------
    Tuple[List[str], Union[Dict[int, List[int]], Dict[int, int]]]
        Extracted words and the requested mapping dictionary.
    """
    words = split_into_words(text)
    if not words:
        return [], {}

    word_spans = _find_word_spans(text, words)

    try:  # Fast path: use offset mappings from the tokenizer
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=max_length is not None,
            max_length=max_length,
            return_offsets_mapping=True,
            return_length=True,
        )
        if max_length and inputs["length"][0] >= max_length:
            logger.warning(
                "Text was truncated to %d tokens – some words may be unmapped.",
                max_length,
            )

        offsets = inputs["offset_mapping"][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        word_to_tokens: Dict[int, List[int]] = {i: [] for i in range(len(words))}
        token_to_word: Dict[int, int] = {}

        for t_idx, (start, end) in enumerate(offsets):
            if start == end == 0 or tokens[t_idx] in tokenizer.all_special_tokens:
                continue
            for w_idx, (w_start, w_end) in enumerate(word_spans):
                if start < w_end and end > w_start:
                    word_to_tokens[w_idx].append(t_idx)
                    token_to_word[t_idx] = w_idx
                    break

        if unmapped := [words[i] for i, toks in word_to_tokens.items() if not toks]:
            logger.debug("%d words could not be mapped (showing up to 5): %s", len(unmapped), unmapped[:5])

        mapping = token_to_word if return_token_to_word else word_to_tokens
        return words, mapping

    except (KeyError, AttributeError):  # Slow path: manual reconstruction
        logger.debug("Offset mapping unavailable - falling back to manual mapping.")
        return _manual_token_word_mapping(
            text,
            tokenizer,
            words,
            word_spans,
            return_token_to_word,
            max_length,
        )


def truncate_text_preserve_structure(
    text: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    *,
    buffer: int = 50,
) -> str:
    """
    Truncate text to fit within token limits while preserving word boundaries.

    Uses binary search over word boundaries to maximize preserved content while 
    reserving buffer space for special tokens.

    Parameters
    ----------
    text : str
        Input string to truncate.
    tokenizer : PreTrainedTokenizer
        HuggingFace tokenizer instance.
    max_length : int
        Maximum allowed number of tokens.
    buffer : int, default 50
        Number of tokens to reserve for special tokens and padding.

    Returns
    -------
    str
        Truncated string fitting within the token limit.
    """
    if tokenizer(text, return_length=True, add_special_tokens=True)["length"][0] <= max_length:
        return text  

    target = max_length - buffer
    words = split_into_words(text)
    left, right, best = 1, len(words), ""

    while left <= right:
        mid = (left + right) // 2
        candidate = " ".join(words[:mid])
        if tokenizer(candidate, return_length=True, add_special_tokens=True)["length"][0] <= target:
            best = candidate
            left = mid + 1
        else:
            right = mid - 1

    return best


def preprocess_texts_for_model_limit(
    texts: List[str],
    model,
    tokenizer: PreTrainedTokenizer,
    max_length: int | None,
):
    """
    Process texts to ensure they fit within model token limits.

    Parameters
    ----------
    texts : List[str]
        Input strings to process.
    model : Any
        HuggingFace model instance with a .config attribute.
    tokenizer : PreTrainedTokenizer
        HuggingFace tokenizer instance.
    max_length : int, optional
        Maximum tokens per text. If None, derived from model and tokenizer.

    Returns
    -------
    Tuple[List[str], int]
        Processed texts and count of texts that were truncated.
    """
    if max_length is None:
        logger.debug("max_length not provided - deriving from model/tokenizer.")
        max_length = get_model_max_length(model, tokenizer)

    processed, truncated = [], 0
    for text in texts:
        clipped = truncate_text_preserve_structure(text, tokenizer, max_length)
        processed.append(clipped)
        truncated += text != clipped

    if truncated:
        logger.warning("Truncated %d/%d texts to <= %d tokens.", truncated, len(texts), max_length)
    return processed, truncated


# ---------------------------------------------------------------------------
# Private helper functions
# ---------------------------------------------------------------------------


def _find_word_spans(text: str, words: List[str]) -> List[Tuple[int, int]]:
    """
    Compute character start and end positions for each word in the original text.

    Assumes words appear in order without overlap (best-effort matching).

    Parameters
    ----------
    text : str
        Original text string.
    words : List[str]
        Word substrings extracted from the text.

    Returns
    -------
    List[Tuple[int, int]]
        Character span (start, end) indices for each word.
    """
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for word in words:
        while cursor < len(text):
            cursor = text.find(word, cursor)
            if cursor == -1:
                break  # Word not found (should not happen)
            end = cursor + len(word)
            spans.append((cursor, end))
            cursor = end
            break
    return spans


def _clean_subword_token(token: str) -> Tuple[str, bool]:
    """
    Remove subword tokenizer markers and identify word continuations.

    Handles WordPiece (##), SentencePiece (▁), and BPE (Ġ) token prefixes.

    Parameters
    ----------
    token : str
        Single token string from the tokenizer.

    Returns
    -------
    Tuple[str, bool]
        Token text without prefix markers and whether it continues the previous word.
    """
    if token.startswith("##"):
        return token[2:], True  # WordPiece continuation
    if token.startswith("▁"):
        return token[1:], False  # SentencePiece new word
    if token.startswith("Ġ") or token.startswith(" "):
        return token.lstrip("Ġ "), False  # GPT-2/BPE new word
    return token, False


def _manual_token_word_mapping(
    text: str,
    tokenizer: PreTrainedTokenizer,
    words: List[str],
    word_spans: List[Tuple[int, int]],
    return_token_to_word: bool,
    max_length: int | None,
):
    """
    Manually reconstruct token-word mappings when offset mappings are unavailable.

    Slower fallback that matches subword tokens to words by concatenation, handling
    special tokens and various subword tokenization schemes.

    Parameters
    ----------
    text : str
        Input text string.
    tokenizer : PreTrainedTokenizer
        HuggingFace tokenizer instance.
    words : List[str]
        Word substrings to map against.
    word_spans : List[Tuple[int, int]]
        Character spans for each word.
    return_token_to_word : bool
        If True, return token-to-word mapping. Otherwise, return word-to-tokens mapping.
    max_length : int, optional
        Maximum number of tokens for truncation.

    Returns
    -------
    Tuple[List[str], Union[Dict[int, List[int]], Dict[int, int]]]
        Original word list and the requested token-word mapping.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=max_length is not None, max_length=max_length)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    word_to_tokens: Dict[int, List[int]] = {i: [] for i in range(len(words))}
    token_to_word: Dict[int, int] = {}

    idx = 0  # Token index (special tokens are skipped automatically)
    for w_idx, word in enumerate(words):
        combined = ""
        while idx < len(tokens):
            token = tokens[idx]
            if token in tokenizer.all_special_tokens:
                idx += 1
                continue

            clean_tok, continuation = _clean_subword_token(token)
            # If we hit a fresh word but have partial match, break
            if not continuation and combined:
                break

            combined += clean_tok
            word_to_tokens[w_idx].append(idx)
            token_to_word[idx] = w_idx
            idx += 1

            if combined.lower() == word.lower():
                break  # Word fully matched
            if len(combined) > len(word):
                logger.debug("Overshot when reconstructing word '%s'.", word)
                break

    mapping = token_to_word if return_token_to_word else word_to_tokens
    return words, mapping


# ---------------------------------------------------------------------------
# Model configuration utilities
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def get_model_max_length(model, tokenizer: PreTrainedTokenizer) -> int:
    """
    Determine a safe maximum sequence length for the model and tokenizer.

    Uses tokenizer.model_max_length if available, otherwise inspects model.config
    attributes like max_position_embeddings or n_positions. Reserves space for
    special tokens when needed.

    Parameters
    ----------
    model : Any
        HuggingFace model instance with a .config attribute.
    tokenizer : PreTrainedTokenizer
        HuggingFace tokenizer instance.

    Returns
    -------
    int
        Maximum sequence length, capped at 128 for memory safety.
    """
    if tokenizer.model_max_length not in (None, _UNKNOWN_LEN_SENTINEL):
        return min(tokenizer.model_max_length, 128)

    cfg = model.config
    if hasattr(cfg, "max_position_embeddings"):
        length = cfg.max_position_embeddings
        if getattr(cfg, "model_type", None) in _SPECIAL_TOKENS_OFFSET_MODELS:
            length -= 2  # Reserve space for CLS + SEP tokens
        return min(length, 128)

    if hasattr(cfg, "n_positions"):
        return min(cfg.n_positions, 128)

    return 128  # Safe default for memory constraints
