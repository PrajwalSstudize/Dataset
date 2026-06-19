# Copyright (c) Opendatalab. All rights reserved.
import base64
import json

from mineru.postprocess.qa_json import clean_exam_text, extract_with_model, generate_qa_json, parse_answers, parse_questions


QUESTION_SAMPLE = """1. If two light waves having same frequency have intensity ratio 4 : 1 and they interfere, the ratio of maximum to minimum
intensity in the pattern will be
(a) 9 : 1 (b) 3 : 1 (c) 25 : 9 (d) 16 : 25
2. In Young's double slit experiment using sodium light (lambda = 5898A), 92 fringes are seen. If given colour (lambda = 5461A) is
used, how many fringes will be seen
(a) 62 (b) 67 (c) 85 (d) 99
3. Two beams of light having intensities I and 4I interfere to produce a fringe pattern on a screen. The phase difference between
the beams is pi/2 at point A and pi at point B. Then the difference between the resultant intensities at A and B is
(a) 2I (b) 4I (c) 5I (d) 7I
4. If two waves represented by y1 = 4 sin omega t and y2 = 3 sin(omega t + pi/3)
interfere at a point, the amplitude of the resulting wave will be about
(a) 7 (b) 6 (c) 5 (d) 3.
"""


ANSWER_SAMPLE = """1. (a)
By using Imax / Imin = 9 / 1.
2. (d)
By using n1 lambda1 = n2 lambda2
=> 92 x 5898 = n2 x 5461
=> n2 = 99
3. (b)
By using I = I1 + I2 + 2 sqrt(I1 I2) cos phi.
At point A : Resultant intensity = 5I
At point B : Resultant intensity = I.
Hence the difference = 4I
"""


def test_parse_questions_with_inline_options_and_decimal_like_option():
    questions = parse_questions(QUESTION_SAMPLE, ".")

    assert len(questions) == 4
    assert questions[0]["question_number"] == 1
    assert questions[0]["question"].startswith("If two light waves")
    assert questions[0]["options"] == [
        "(a) 9 : 1",
        "(b) 3 : 1",
        "(c) 25 : 9",
        "(d) 16 : 25",
    ]
    assert questions[3]["question_number"] == 4
    assert questions[3]["options"][-1] == "(d) 3."


def test_parse_answers_until_next_numbered_answer():
    answers = parse_answers(ANSWER_SAMPLE, ".")

    assert len(answers) == 3
    assert answers[0]["Index"] == "1"
    assert answers[0]["correctOption"] == "a"
    assert answers[1]["correctOption"] == "d"
    assert "n2 = 99" in answers[1]["SolutionData"]
    assert "Hence the difference = 4I" in answers[2]["SolutionData"]


def test_generate_question_json_with_base64_image(tmp_path):
    parse_dir = tmp_path / "sample" / "auto"
    image_dir = parse_dir / "images"
    image_dir.mkdir(parents=True)
    image_bytes = b"fake-png"
    image_path = image_dir / "q1.png"
    image_path.write_bytes(image_bytes)
    (parse_dir / "sample.md").write_text(
        "1. What is shown?\n![](images/q1.png)\n(a) A (b) B (c) C (d) D\n",
        encoding="utf-8",
    )

    output_path = generate_qa_json(parse_dir, "sample", "questions")
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    expected_base64 = base64.b64encode(image_bytes).decode("ascii")
    assert payload["questions"][0]["img"] == f"data:image/png;base64,{expected_base64}"


def test_model_extraction_is_used_before_rule_fallback(tmp_path):
    parse_dir = tmp_path / "sample" / "auto"
    image_dir = parse_dir / "images"
    image_dir.mkdir(parents=True)
    image_bytes = b"model-image"
    (image_dir / "q1.png").write_bytes(image_bytes)
    markdown = "1. OCR text that the model will structure\n![](images/q1.png)\n"

    def fake_model_caller(model_name, messages):
        assert model_name == "qa-model"
        assert messages[0]["role"] == "user"
        return json.dumps(
            {
                "questions": [
                    {
                        "page_no": None,
                        "question_number": 1,
                        "question": "Structured by model",
                        "options": ["a first", "b second", "c third", "d fourth"],
                        "img": "images/q1.png",
                    }
                ]
            }
        )

    payload = extract_with_model(
        markdown,
        parse_dir,
        "questions",
        model_name="qa-model",
        model_caller=fake_model_caller,
    )

    expected_base64 = base64.b64encode(image_bytes).decode("ascii")
    assert payload["questions"][0]["question"] == "Structured by model"
    assert payload["questions"][0]["options"][0] == "(a) first"
    assert payload["questions"][0]["img"] == f"data:image/png;base64,{expected_base64}"


def test_clean_exam_text_normalizes_math_glyphs_and_broken_sub_tags():
    cleaned = clean_exam_text("Given \uf06c = 5898Å and \uf06d = 1.5, I<sub><sub>max</sub></sub> / I<sub>min</sub>")

    assert "λ = 5898Å" in cleaned
    assert "μ = 1.5" in cleaned
    assert "I_max" in cleaned
    assert "I_min" in cleaned
    assert "<sub>" not in cleaned


def test_model_output_cleanup_applies_to_questions_and_answers():
    question_payload = extract_with_model(
        "1. raw",
        ".",
        "questions",
        model_name="qa-model",
        model_caller=lambda model, messages: json.dumps(
            {
                "questions": [
                    {
                        "page_no": None,
                        "question_number": 1,
                        "question": "Find \uf06c when I<sub><sub>max</sub></sub> is given",
                        "options": ["a \uf06d", "b I<sub>min</sub>"],
                        "img": None,
                    }
                ]
            }
        ),
    )
    answer_payload = extract_with_model(
        "1. raw",
        ".",
        "answers",
        model_name="qa-model",
        model_caller=lambda model, messages: json.dumps(
            {
                "answers": [
                    {
                        "Index": "1",
                        "correctOption": "a",
                        "SolutionData": "Use \uf06c/2 and I<sub><sub>min</sub></sub>",
                        "img": None,
                    }
                ]
            }
        ),
    )

    assert "λ" in question_payload["questions"][0]["question"]
    assert "I_max" in question_payload["questions"][0]["question"]
    assert question_payload["questions"][0]["options"][0] == "(a) μ"
    assert "λ/2" in answer_payload["answers"][0]["SolutionData"]
    assert "I_min" in answer_payload["answers"][0]["SolutionData"]
