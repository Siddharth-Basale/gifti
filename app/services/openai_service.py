import json
from typing import Literal

from openai import OpenAI

from app.config import get_openai_api_key

# Tier 1: cheap models (GPT Image: better instruction following than DALL-E 2, no unwanted text)
TIER1_CHAT_MODEL = "gpt-3.5-turbo"
TIER1_IMAGE_MODEL = "gpt-image-1-mini"
TIER1_IMAGE_QUALITY = "low"  # $0.005/image at 1024x1024
TIER1_IMAGE_SIZE = "1024x1024"

# Tier 2: premium models
TIER2_CHAT_MODEL = "gpt-4o-mini"
TIER2_IMAGE_MODEL = "gpt-image-1.5"
TIER2_IMAGE_QUALITY = "high"
TIER2_IMAGE_SIZE = "1024x1024"


def get_client() -> OpenAI:
    return OpenAI(api_key=get_openai_api_key())


Tier = Literal["tier1", "tier2"]


def generate_description_and_tag(
    client: OpenAI,
    giftcard_name: str,
    prompt: str,
    tier: Tier,
) -> dict:
    """Return descriptions (medium x2, short x2), tags (~10), and 5 refactored gift card name suggestions."""
    model = TIER1_CHAT_MODEL if tier == "tier1" else TIER2_CHAT_MODEL
    system = (
        "You are a gift card copywriter for digital wallet gift cards .\n\n"
        "Given a gift card name and a customer prompt, you must return ALL of the following in one JSON object:\n\n"
        "1. descriptions_medium: array of exactly 2 different medium-length descriptions (4-5 sentences each). "
        "Both should be based on the prompt and gift card name; vary tone or angle slightly.\n"
        "2. descriptions_short: array of exactly 2 different short descriptions (1 sentence each). "
        "Same theme, more concise.\n"
        "3. tags: array of about 10 tags. Each tag is 1-3 words. Mix promotion type, occasion, and vibe.\n"
        "4. giftcard_name_suggestions: array of exactly 5 refactored/improved versions of the gift card name. "
        "Keep the intent but make them clearer, catchier, or more professional.\n\n"
        "Context: Infer business domain and promotion type (birthday, festival, sale, thank-you, etc.) from the prompt. "
        "Make all copy fit that type.\n\n"
        "Reply ONLY with valid JSON in exactly this shape (no other keys):\n"
        "{\"descriptions_medium\":[\"...\",\"...\"],\"descriptions_short\":[\"...\",\"...\"],"
        "\"tags\":[\"...\",\"...\",...],\"giftcard_name_suggestions\":[\"...\",\"...\",\"...\",\"...\",\"...\"]}"
    )
    user = f"Gift card name: {giftcard_name}\nPrompt: {prompt}"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    data = json.loads(raw)
    return {
        "descriptions_medium": [s.strip() for s in data.get("descriptions_medium", [])][:2],
        "descriptions_short": [s.strip() for s in data.get("descriptions_short", [])][:2],
        "tags": [s.strip() for s in data.get("tags", [])][:12],
        "giftcard_name_suggestions": [s.strip() for s in data.get("giftcard_name_suggestions", [])][:5],
    }


def generate_image(
    client: OpenAI,
    giftcard_name: str,
    description: str,
    tier: Tier,
) -> tuple[str, str]:
    """Return (image_base64, media_type). GPT Image models always return base64."""
    image_prompt = (
        f"Background image that visually depicts the following: {description}. "
        f"Theme or subject: {giftcard_name}. "
        "Purely visual scene, no cards, no text or writing of any kind. Clean and professional."
    )
    if tier == "tier1":
        resp = client.images.generate(
            model=TIER1_IMAGE_MODEL,
            prompt=image_prompt,
            size=TIER1_IMAGE_SIZE,
            quality=TIER1_IMAGE_QUALITY,
            n=1,
        )
    else:
        resp = client.images.generate(
            model=TIER2_IMAGE_MODEL,
            prompt=image_prompt,
            size=TIER2_IMAGE_SIZE,
            quality=TIER2_IMAGE_QUALITY,
            n=1,
        )
    # GPT Image models always return b64_json (no url)
    b64 = resp.data[0].b64_json
    return (b64, "image/png")
