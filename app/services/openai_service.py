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
) -> dict[str, str]:
    """Return {"description": str, "tag": str}."""
    model = TIER1_CHAT_MODEL if tier == "tier1" else TIER2_CHAT_MODEL
    system = (
        "You are a gift card copywriter. You produce copy for digital wallet gift cards "
        "(Google Wallet / Apple Wallet): one short description and one small tag.\n\n"
        "Context and rules:\n"
        "- Business domain and intent come from the customer's prompt.\n"
        "- Infer the promotion type from the prompt when possible Make the description and tag fit that type.\n"
        "- Description should be 4-5 sentences, based on the prompt and gift card name.\n"
        "- Tag must be 1–3 words.\n\n"
        "Reply ONLY with valid JSON in exactly this shape: "
        "{\"description\":\"...\",\"tag\":\"...\"}"
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
    description = data.get("description", "").strip()
    tag = data.get("tag", "").strip()
    return {"description": description, "tag": tag}


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
