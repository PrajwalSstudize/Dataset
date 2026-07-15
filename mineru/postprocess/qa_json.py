# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

DocumentType = Literal["questions", "answers", "none"]
ModelCaller = Callable[[str, list[dict[str, Any]]], str]
logger = logging.getLogger(__name__)

QUESTION_OPTION_RE = re.compile(
    r"(?P<label>\([a-dA-D]\))\s*(?P<text>.*?)(?=\s*\([a-dA-D]\)\s*|$)",
    re.DOTALL,
)
NUMBERED_BLOCK_RE = re.compile(r"(?m)^\s*(?P<number>\d{1,4})\.\s*")
ANSWER_HEAD_RE = re.compile(r"^\s*(?:\((?P<paren>[a-dA-D])\)|(?P<bare>[a-dA-D]))(?=\s|$|[\).:-])")
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\((?P<src>[^)]+)\)")
HTML_IMAGE_RE = re.compile(
    r"<img\b[^>]*?\bsrc\s*=\s*(?P<quote>[\"'])(?P<src>.*?)(?P=quote)",
    re.IGNORECASE,
)
SUB_TAG_RE = re.compile(r"</?\s*sub\s*>", re.IGNORECASE)
SUP_TAG_RE = re.compile(r"</?\s*sup\s*>", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"</?(?!img\b)[a-zA-Z][^>]*>")
MATH_SYMBOL_REPLACEMENTS = {
    "\uf06c": "λ",
    "\u03bb": "λ",
    "\uf06d": "μ",
    "\u03bc": "μ",
    "\u00b5": "μ",
    "\uf070": "π",
    "\u03c0": "π",
    "\uf077": "ω",
    "\u03c9": "ω",
    "\uf066": "φ",
    "\u03c6": "φ",
    "\u03d5": "φ",
    "\uf071": "θ",
    "\u03b8": "θ",
    "\uf0de": "=>",
    "\uf0b4": "therefore",
    "\u212b": "Å",
}

# Strips the "www.igyanam.com" watermark/credit line that shows up in some
# source markdown (with or without protocol/www, tolerant of OCR spacing
# around the dots, e.g. "www . igyanam . com").
WATERMARK_RE = re.compile(
    r"(?:https?://)?\s*(?:www\s*\.\s*)?igyanam\s*\.\s*com\b",
    re.IGNORECASE,
)

# --- Merged-question detection -------------------------------------------------
# Matches a numbered-question marker that appears *inside* already-extracted text
# (i.e. not the leading number MinerU/the model was asked to strip), signalling
# that two or more questions were merged into a single object.
EMBEDDED_QUESTION_NUMBER_RE = re.compile(
    r"(?:\n|\s+)(?P<number>\d{1,4})\.\s+(?=[A-Z(])"
)


@dataclass
class ParsedBlock:
    number: int
    text: str
    images: list[str] = field(default_factory=list)


def generate_qa_json(
    parse_dir: str | Path,
    file_stem: str,
    document_type: DocumentType,
    *,
    model_base_url: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    model_caller: ModelCaller | None = None,
    allow_rule_fallback: bool = True,
) -> Path | None:
    """Generate final exam JSON next to MinerU's Markdown output."""
    if document_type == "none":
        return None

    parse_dir = Path(parse_dir)
    markdown_path = parse_dir / f"{file_stem}.md"
    if not markdown_path.is_file():
        raise FileNotFoundError(f"Missing markdown file: {markdown_path}")

    markdown_text = markdown_path.read_text(encoding="utf-8")
    payload = extract_with_model(
        markdown_text,
        parse_dir,
        document_type,
        model_base_url=model_base_url,
        model_name=model_name,
        api_key=api_key,
        model_caller=model_caller,
    )
    if payload is None:
        if not allow_rule_fallback:
            raise RuntimeError("QA JSON model extraction failed and fallback is disabled.")
        logger.warning("QA JSON model extraction unavailable; using rule-based fallback.")
        payload = extract_with_rules(markdown_text, parse_dir, document_type)

    if document_type == "questions":
        output_path = parse_dir / f"{file_stem}_questions.json"
    elif document_type == "answers":
        output_path = parse_dir / f"{file_stem}_answers.json"
    else:
        raise ValueError(f"Unsupported QA document type: {document_type}")

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def extract_with_rules(
    markdown_text: str,
    parse_dir: str | Path,
    document_type: Literal["questions", "answers"],
) -> dict[str, Any]:
    if document_type == "questions":
        return {"questions": parse_questions(markdown_text, parse_dir)}
    return {"answers": parse_answers(markdown_text, parse_dir)}


def extract_with_model(
    markdown_text: str,
    parse_dir: str | Path,
    document_type: Literal["questions", "answers"],
    *,
    model_base_url: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    model_caller: ModelCaller | None = None,
) -> dict[str, Any] | None:
    model_base_url = normalize_model_base_url(
        model_base_url or os.getenv("MINERU_QA_JSON_BASE_URL")
    )
    model_name = model_name or os.getenv("MINERU_QA_JSON_MODEL")
    api_key = api_key or os.getenv("MINERU_QA_JSON_API_KEY", "EMPTY")

    if model_caller is None and not model_base_url:
        return None

    image_sources = collect_image_sources(markdown_text)
    prompt = build_extraction_prompt(markdown_text, document_type, image_sources)
    messages = build_model_messages(prompt, image_sources, parse_dir)

    try:
        raw_content = (
            model_caller(model_name or "", messages)
            if model_caller is not None
            else call_openai_compatible_model(
                model_base_url=model_base_url,
                model_name=model_name,
                api_key=api_key,
                messages=messages,
            )
        )
        payload = parse_model_json(raw_content)
        payload = normalize_model_payload(payload, parse_dir, document_type)
        validate_payload(payload, document_type)
        return payload
    except Exception as exc:
        logger.warning(f"QA JSON model extraction failed: {exc}")
        return None


def normalize_model_base_url(base_url: str | None) -> str | None:
    if not base_url:
        return None
    base_url = base_url.strip().rstrip("/")
    if not base_url:
        return None
    if base_url.endswith("/v1"):
        return base_url
    return f"{base_url}/v1"


def call_openai_compatible_model(
    *,
    model_base_url: str,
    model_name: str | None,
    api_key: str,
    messages: list[dict[str, Any]],
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=model_base_url)
    if not model_name:
        model_name = discover_openai_compatible_model(client)
    request_payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0,
    }
    try:
        completion = client.chat.completions.create(
            **request_payload,
            response_format={"type": "json_object"},
        )
    except Exception:
        completion = client.chat.completions.create(**request_payload)
    content = completion.choices[0].message.content
    if content is None:
        raise ValueError("Model returned empty content.")
    return content


def discover_openai_compatible_model(client: Any) -> str:
    models = client.models.list()
    for model in getattr(models, "data", []) or []:
        model_id = getattr(model, "id", None)
        if model_id:
            return model_id
    raise ValueError("No model name configured and server did not return any models.")


def build_extraction_prompt(
    markdown_text: str,
    document_type: Literal["questions", "answers"],
    image_sources: list[str],
) -> str:
    image_instruction = (
        "Images found in the Markdown are listed by their exact path. If an image belongs to an item, "
        "set that item's img field to that exact image path. If no image belongs to it, set img to null."
    )
    cleanup_instruction = (
        "Clean OCR/math artifacts in question, option, and solution text: convert private-use glyphs such as "
        " to λ/lambda,  to μ/mu, π/omega/phi/theta symbols to readable math text, remove malformed repeated "
        "<sub>/<sup> tags, and rewrite subscript/superscript notation as readable plain text like I_max, I_min, "
        "S_1, S_2, λ/2, or H_2O. Do not leave raw broken HTML tags in the JSON. Also remove any occurrence of "
        "the watermark/credit text \"www.igyanam.com\" (in any casing, spacing, or with/without \"www.\"/protocol) "
        "from every field -- it must never appear anywhere in the output."
    )
    if document_type == "questions":
        schema_instruction = """Return only valid JSON using this exact schema:
{
  "questions": [
    {
      "page_no": null,
      "question_number": 1,
      "question": "question text",
      "options": ["(a) option", "(b) option", "(c) option", "(d) option"],
      "img": null
    }
  ]
}"""
        task_instruction = (
            "Extract every physics multiple-choice question. Options may be inline or split across lines. "
            "Keep formulas semantically faithful while cleaning OCR artifacts. Do not solve the questions."
            "Make sure that there should be only one question in single JSON object. Therefor, one question data per JSON"
        )
    else:
        schema_instruction = """Return only valid JSON using this exact schema:
{
  "answers": [
    {
      "Index": "1",
      "correctOption": "a",
      "SolutionData": "solution text",
      "img": null
    }
  ]
}"""
        task_instruction = (
            "Extract every answer key and its solution. Each numbered answer continues until the next numbered answer. "
            "The correct option must be a lowercase letter a, b, c, or d when present."
        )

    image_list = "\n".join(f"- {source}" for source in image_sources) or "- none"
    return f"""{task_instruction}

{schema_instruction}

Rules:
- Return JSON only. Do not wrap in Markdown.
- Do not invent missing items.
- Preserve item numbering from the source.
- Normalize option labels to "(a)", "(b)", "(c)", "(d)" for questions.
- Never place more than one question's text or option set inside a single JSON question object. Each object in the "questions" array must contain exactly one question and, at most, one set of (a)-(d) options.
- {cleanup_instruction}
- {image_instruction}

Image paths:
{image_list}

Markdown:
{markdown_text}
"""


def build_model_messages(
    prompt: str,
    image_sources: list[str],
    parse_dir: str | Path,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image_source in image_sources[:20]:
        image_data = encode_image(image_source, parse_dir)
        if image_data:
            content.append({"type": "image_url", "image_url": {"url": image_data}})
    return [{"role": "user", "content": content}]


def parse_model_json(raw_content: str) -> dict[str, Any]:
    if "</think>" in raw_content:
        raw_content = raw_content.split("</think>", 1)[1]
    try:
        import json_repair

        payload = json_repair.loads(raw_content)
    except ImportError:
        payload = json.loads(raw_content)
    if not isinstance(payload, dict):
        raise ValueError("Model JSON output must be an object.")
    return payload


def normalize_model_payload(
    payload: dict[str, Any],
    parse_dir: str | Path,
    document_type: Literal["questions", "answers"],
) -> dict[str, Any]:
    if document_type == "questions":
        items = payload.get("questions")
        if not isinstance(items, list):
            raise ValueError("Model output missing questions list.")
        normalized = [normalize_model_question(item, parse_dir) for item in items]
        payload["questions"] = split_merged_question_items(normalized)
        return payload

    items = payload.get("answers")
    if not isinstance(items, list):
        raise ValueError("Model output missing answers list.")
    payload["answers"] = [normalize_model_answer(item, parse_dir) for item in items]
    return payload


def normalize_model_question(item: Any, parse_dir: str | Path) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("Question item must be an object.")
    return {
        "page_no": item.get("page_no"),
        "question_number": int(item.get("question_number")),
        "question": clean_exam_text(str(item.get("question", ""))),
        "options": normalize_options(item.get("options", [])),
        "img": normalize_model_img(item.get("img"), parse_dir),
    }


def normalize_model_answer(item: Any, parse_dir: str | Path) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("Answer item must be an object.")
    correct_option = str(item.get("correctOption", "") or "").strip().lower()
    correct_option = correct_option[:1] if correct_option[:1] in {"a", "b", "c", "d"} else ""
    return {
        "Index": str(item.get("Index", "")).strip(),
        "correctOption": correct_option,
        "SolutionData": clean_exam_text(str(item.get("SolutionData", ""))),
        "img": normalize_model_img(item.get("img"), parse_dir),
    }


def normalize_options(options: Any) -> list[str]:
    if not isinstance(options, list):
        return []
    normalized = []
    for option in options:
        option_text = clean_exam_text(str(option))
        match = re.match(r"^\(?([a-dA-D])\)?[\).:-]?\s*(.*)$", option_text)
        if match:
            option_text = clean_exam_text(f"({match.group(1).lower()}) {match.group(2).strip()}".strip())
        if option_text:
            normalized.append(option_text)
    return normalized


def normalize_model_img(value: Any, parse_dir: str | Path) -> str | list[str] | None:
    if value is None or value == "":
        return None
    if isinstance(value, list):
        encoded = [normalize_model_img(item, parse_dir) for item in value]
        encoded = [item for item in encoded if item]
        if not encoded:
            return None
        return encoded[0] if len(encoded) == 1 else encoded
    if not isinstance(value, str):
        return None
    if value.startswith("data:"):
        return value
    return encode_image(value, parse_dir)


def validate_payload(
    payload: dict[str, Any],
    document_type: Literal["questions", "answers"],
) -> None:
    if document_type == "questions":
        questions = payload.get("questions")
        if not isinstance(questions, list):
            raise ValueError("questions must be a list.")
        for item in questions:
            if not isinstance(item.get("question_number"), int):
                raise ValueError("question_number must be an integer.")
            if not isinstance(item.get("question"), str):
                raise ValueError("question must be a string.")
            if not isinstance(item.get("options"), list):
                raise ValueError("options must be a list.")
        return

    answers = payload.get("answers")
    if not isinstance(answers, list):
        raise ValueError("answers must be a list.")
    for item in answers:
        if not isinstance(item.get("Index"), str):
            raise ValueError("Index must be a string.")
        if not isinstance(item.get("correctOption"), str):
            raise ValueError("correctOption must be a string.")
        if not isinstance(item.get("SolutionData"), str):
            raise ValueError("SolutionData must be a string.")


def parse_questions(markdown_text: str, parse_dir: str | Path) -> list[dict[str, Any]]:
    questions = []
    for block in split_numbered_blocks(markdown_text):
        body = strip_image_markup(block.text)
        option_matches = list(QUESTION_OPTION_RE.finditer(body))
        if option_matches:
            question_text = body[: option_matches[0].start()]
            options = [
                normalize_space(f"{match.group('label').lower()} {match.group('text')}")
                for match in option_matches
                if normalize_space(match.group("text"))
            ]
        else:
            question_text = body
            options = []

        questions.append(
            {
                "page_no": None,
                "question_number": block.number,
                "question": clean_exam_text(question_text),
                "options": [clean_exam_text(option) for option in options],
                "img": encode_images(block.images, parse_dir),
            }
        )
    return split_merged_question_items(questions)


def parse_answers(markdown_text: str, parse_dir: str | Path) -> list[dict[str, Any]]:
    answers = []
    for block in split_numbered_blocks(markdown_text):
        body = normalize_answer_text(strip_image_markup(block.text))
        match = ANSWER_HEAD_RE.match(body)
        correct_option = ""
        if match:
            correct_option = (match.group("paren") or match.group("bare")).lower()
            body = body[match.end() :]

        body = re.sub(r"^\s*[\).:-]+\s*", "", body)
        answers.append(
            {
                "Index": str(block.number),
                "correctOption": correct_option,
                "SolutionData": clean_exam_text(body),
                "img": encode_images(block.images, parse_dir),
            }
        )
    return answers


def split_numbered_blocks(markdown_text: str) -> list[ParsedBlock]:
    text = normalize_ocr_symbols(markdown_text)
    matches = list(NUMBERED_BLOCK_RE.finditer(text))
    blocks: list[ParsedBlock] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        raw_body = text[start:end].strip()
        if not raw_body:
            continue
        body, images = extract_image_sources(raw_body)
        blocks.append(ParsedBlock(number=int(match.group("number")), text=body, images=images))
    return blocks


def extract_image_sources(text: str) -> tuple[str, list[str]]:
    images = [match.group("src").strip() for match in MARKDOWN_IMAGE_RE.finditer(text)]
    images.extend(match.group("src").strip() for match in HTML_IMAGE_RE.finditer(text))
    return text, [image for image in images if image]


def collect_image_sources(text: str) -> list[str]:
    _, image_sources = extract_image_sources(text)
    seen = set()
    unique_sources = []
    for source in image_sources:
        if source in seen:
            continue
        seen.add(source)
        unique_sources.append(source)
    return unique_sources


def strip_image_markup(text: str) -> str:
    text = MARKDOWN_IMAGE_RE.sub(" ", text)
    return HTML_IMAGE_RE.sub(" ", text)


def normalize_ocr_symbols(text: str) -> str:
    return (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\uff08", "(")
        .replace("\uff09", ")")
        .replace("\u00a0", " ")
    )


def normalize_answer_text(text: str) -> str:
    text = re.sub(r"^\s*answer\s*[:.-]\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_exam_text(text: str) -> str:
    text = normalize_ocr_symbols(str(text))
    text = WATERMARK_RE.sub("", text)
    text = re.sub(r"[(\[{]\s*[)\]}]", "", text)
    for source, replacement in MATH_SYMBOL_REPLACEMENTS.items():
        text = text.replace(source, replacement)
    text = normalize_repeated_scripts(text, "sub")
    text = normalize_repeated_scripts(text, "sup")
    text = SUB_TAG_RE.sub("_", text)
    text = SUP_TAG_RE.sub("^", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"_+", "_", text)
    text = re.sub(r"\^+", "^", text)
    text = re.sub(r"\s*([_^])\s*", r"\1", text)
    text = re.sub(r"([_^])([,.;:)\]}])", r"\2", text)
    text = re.sub(r"([(\[{])([_^])", r"\1", text)
    return normalize_space(text)


def normalize_repeated_scripts(text: str, tag_name: Literal["sub", "sup"]) -> str:
    marker = "_" if tag_name == "sub" else "^"
    open_tag = re.compile(rf"(?:<\s*{tag_name}\s*>)+", re.IGNORECASE)
    close_tag = re.compile(rf"(?:<\s*/\s*{tag_name}\s*>)+", re.IGNORECASE)
    text = open_tag.sub(marker, text)
    return close_tag.sub("", text)


def encode_images(image_sources: list[str], parse_dir: str | Path) -> str | None:
    encoded_images = []
    for source in image_sources:
        encoded_image = encode_image(source, parse_dir)
        if encoded_image:
            encoded_images.append(encoded_image)
    if not encoded_images:
        return None
    if len(encoded_images) == 1:
        return encoded_images[0]
    return encoded_images


def encode_image(image_source: str, parse_dir: str | Path) -> str | None:
    if image_source.startswith("data:"):
        return image_source
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", image_source):
        return None

    image_path = (Path(parse_dir) / image_source).resolve(strict=False)
    parse_root = Path(parse_dir).resolve(strict=False)
    try:
        image_path.relative_to(parse_root)
    except ValueError:
        return None
    if not image_path.is_file():
        return None

    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


# --- Merge guard: enforce one question per JSON object -------------------------


def split_merged_question_items(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure no single question object contains more than one question.

    Detects two symptoms of a merge:
      1. An embedded numbered-question marker inside the `question` text
         (e.g. "...final answer.\\n7. A block slides down...").
      2. An `options` list containing more than one full (a)-(d) cycle.

    Any offending item is split into multiple standalone question objects,
    and the full list is renumbered sequentially afterwards.
    """
    expanded: list[dict[str, Any]] = []
    for item in questions:
        expanded.extend(_split_single_question_item(item))
    return _renumber_questions(expanded)


def _split_single_question_item(item: dict[str, Any]) -> list[dict[str, Any]]:
    text = item.get("question", "") or ""
    options = item.get("options", []) or []

    text_segments = _split_embedded_question_text(text)
    option_groups = _split_option_cycles(options)

    if len(text_segments) <= 1 and len(option_groups) <= 1:
        return [item]

    segment_count = max(len(text_segments), len(option_groups))

    if len(text_segments) < segment_count:
        # Options show more (a)-(d) cycles than we found question-text boundaries
        # for. We can't know where the missing question's text lives, so we must
        # never silently drop the orphaned options -- keep them, flagged for
        # manual review, instead of discarding them.
        fallback_text = text_segments[-1] if text_segments else text
        while len(text_segments) < segment_count:
            text_segments.append(
                f"{fallback_text} [REVIEW: extra option set detected without matching question text]"
            )

    option_groups = _align_option_groups_to_segments(text_segments, option_groups, segment_count)

    raw_images = item.get("img")
    image_list = raw_images if isinstance(raw_images, list) else ([raw_images] if raw_images else [])

    split_items = []
    for index in range(segment_count):
        question_text = text_segments[index].strip()
        if not question_text:
            continue
        parsed_segment = _parse_question_segment(question_text, option_groups[index], index + 1)
        split_items.append(
            {
                "page_no": item.get("page_no"),
                "question_number": item.get("question_number"),
                "question": parsed_segment["question"],
                "options": parsed_segment["options"],
                "img": image_list[index] if index < len(image_list) else None,
            }
        )

    if not split_items:
        return [item]

    logger.warning(
        "Detected a merged question object (question_number=%s); split into %d separate question objects.",
        item.get("question_number"),
        len(split_items),
    )
    return split_items


def _align_option_groups_to_segments(
    text_segments: list[str],
    option_groups: list[list[str]],
    segment_count: int,
) -> list[list[str]]:
    if len(option_groups) == 1 and segment_count > 1:
        earlier_segments_have_inline_options = any(
            _segment_contains_option_cycle(segment) for segment in text_segments[:-1]
        )
        if earlier_segments_have_inline_options:
            return ([[]] * (segment_count - 1)) + option_groups
    return _pad_to_length(option_groups, segment_count, default=[])


def _segment_contains_option_cycle(text: str) -> bool:
    labels = {match.group(1).lower() for match in re.finditer(r"\(([a-dA-D])\)", text)}
    return {"a", "b", "c", "d"}.issubset(labels)


def _parse_question_segment(question_text: str, options: list[str], question_number: int) -> dict[str, Any]:
    question_text = re.sub(r"^\s*\d{1,4}\.\s*", "", question_text)
    synthetic_markdown = "\n".join(
        [f"{question_number}. {question_text}", *options]
    )
    parsed_questions = parse_questions(synthetic_markdown, ".")
    if parsed_questions:
        parsed_question = dict(parsed_questions[0])
        parsed_question["question_number"] = question_number
        return parsed_question
    return {
        "page_no": None,
        "question_number": question_number,
        "question": clean_exam_text(question_text),
        "options": [clean_exam_text(option) for option in options],
        "img": None,
    }


def _split_embedded_question_text(text: str) -> list[str]:
    if not text:
        return [text]
    matches = list(EMBEDDED_QUESTION_NUMBER_RE.finditer(text))
    if not matches:
        return [text]
    segments = []
    start = 0
    for match in matches:
        segment = text[start : match.start()]
        if segment.strip():
            segments.append(segment)
        start = match.end()
    tail = text[start:]
    if tail.strip():
        segments.append(tail)
    return segments if len(segments) > 1 else [text]


def _split_option_cycles(options: list[str]) -> list[list[str]]:
    if not options:
        return [[]]
    groups: list[list[str]] = []
    current: list[str] = []
    seen_labels: set[str] = set()
    for option in options:
        label_match = re.match(r"^\(([a-dA-D])\)", option)
        label = label_match.group(1).lower() if label_match else None
        if label and label in seen_labels:
            groups.append(current)
            current = []
            seen_labels = set()
        if label:
            seen_labels.add(label)
        current.append(option)
    if current:
        groups.append(current)
    return groups or [[]]


def _pad_to_length(items: list[Any], length: int, default: Any) -> list[Any]:
    items = list(items)
    while len(items) < length:
        items.append(default)
    return items


def _renumber_questions(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    renumbered = []
    for index, item in enumerate(questions, start=1):
        item = dict(item)
        item["question_number"] = index
        renumbered.append(item)
    return renumbered
