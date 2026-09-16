#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
    ### YOUR CODE HERE
    import os

    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_deepseek import ChatDeepSeek
    from pydantic import BaseModel, Field

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is missing. Put it in .env before running hw1.py."
        )

    class ReceiptExtraction(BaseModel):
        """Amounts needed to answer both homework questions for one receipt."""

        amount_paid_after_rounding: Decimal = Field(
            description="The final payment/tender amount after ROUNDING"
        )
        subtotal_after_discounts_before_rounding: Decimal = Field(
            description="The receipt's SUBTOTAL before the ROUNDING line"
        )
        rounding_adjustment: Decimal = Field(
            description="The signed ROUNDING amount, or 0 when no rounding is printed"
        )
        discount_lines: list[Decimal] = Field(
            description=(
                "Signed HKD amounts for every discount/promotion/coupon line before "
                "SUBTOTAL; exclude ROUNDING"
            )
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You extract monetary fields from one Hong Kong supermarket receipt.
Read the receipt image carefully and return only one valid JSON object matching the
provided schema. Do not wrap the JSON in Markdown fences and do not add prose.

Rules:
1. amount_paid_after_rounding is the final tender/payment amount immediately after
   the ROUNDING line (for example OCTOPUS, VISA, CASH, or another payment method).
   Do not use change, card balance, amount deducted repeated later, points, or dates.
2. subtotal_after_discounts_before_rounding is the SUBTOTAL shown immediately before
   ROUNDING. It already includes all discounts.
3. rounding_adjustment is only the signed amount on the ROUNDING line.
4. discount_lines must be a JSON array of numbers containing every negative discount,
   promotion, coupon, member, app, percentage-off, multi-buy saving, or packaging-
   damage reduction printed in the item/discount area before SUBTOTAL. Keep each
   signed amount exactly as printed; for example: [-12.40, -5.39].
5. Never include ROUNDING in discount_lines. Never include payment, change, balances,
   points, dates, quantities, unit prices, or ordinary positive item prices.
6. A zero-value coupon is not a discount. If there are no discounts, return an empty list.
7. Use HKD decimal amounts without a currency symbol or thousands separator. Do not
   guess an unreadable value; re-check the image and the arithmetic around SUBTOTAL.""",
            ),
            MessagesPlaceholder("receipt_message"),
        ]
    )

    model = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        api_key=api_key,
        temperature=0,
        max_retries=2,
    )
    extractor = model.with_structured_output(
        ReceiptExtraction,
        method="json_mode",
    )
    return prompt | extractor


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    ### YOUR CODE HERE
    from langchain_core.messages import HumanMessage

    requests = []
    for image in images:
        requests.append(
            {
                "receipt_message": [
                    HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": (
                                    f"Extract the required fields from {image.name}. "
                                    "Return one structured record for this receipt only."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data_url(image)},
                            },
                        ]
                    )
                ]
            }
        )

    # Receipts are independent, so bounded parallel extraction reduces total latency.
    extractions = chain.batch(
        requests,
        config={"max_concurrency": min(4, len(requests))},
        return_exceptions=True,
    )

    def value(record: Any, field: str) -> Any:
        if isinstance(record, dict):
            return record[field]
        return getattr(record, field)

    def amounts_are_consistent(record: Any) -> bool:
        """Check the receipt identity: subtotal + rounding = final payment."""
        try:
            paid = Decimal(str(value(record, "amount_paid_after_rounding")))
            subtotal = Decimal(
                str(value(record, "subtotal_after_discounts_before_rounding"))
            )
            rounding = Decimal(str(value(record, "rounding_adjustment")))
        except (AttributeError, InvalidOperation, KeyError, TypeError, ValueError):
            return False
        cents = Decimal("0.01")
        return paid.quantize(cents) == (subtotal + rounding).quantize(cents)

    # Retry only failed or internally inconsistent receipts once, rather than
    # paying to rerun the whole folder.
    for index, extraction in enumerate(extractions):
        failed = isinstance(extraction, BaseException)
        if failed or not amounts_are_consistent(extraction):
            try:
                extractions[index] = chain.invoke(requests[index])
            except Exception as exc:
                if failed:
                    raise RuntimeError(
                        f"Could not extract amounts from {images[index].name}"
                    ) from exc

    total_paid = Decimal("0")
    total_without_discounts = Decimal("0")

    for extraction in extractions:
        paid = Decimal(str(value(extraction, "amount_paid_after_rounding")))
        subtotal = Decimal(
            str(value(extraction, "subtotal_after_discounts_before_rounding"))
        )
        discount_total = Decimal("0")

        for discount in value(extraction, "discount_lines"):
            discount_total += abs(Decimal(str(discount)))

        total_paid += paid
        total_without_discounts += subtotal + discount_total

    cents = Decimal("0.01")
    total_paid = total_paid.quantize(cents)
    total_without_discounts = total_without_discounts.quantize(cents)

    return {
        QUERY_1: f"HK${total_paid:.2f}",
        QUERY_2: f"HK${total_without_discounts:.2f}",
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
