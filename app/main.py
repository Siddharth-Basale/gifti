from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

from app.config import get_openai_api_key
from app.services.openai_service import (
    get_client,
    generate_description_and_tag,
    generate_image,
    Tier,
)

app = FastAPI(title="Gift Card API", description="Two-tier gift card description and image generation.")


class DescribeRequest(BaseModel):
    giftcard_name: str
    prompt: str


class DescribeResponse(BaseModel):
    descriptions_medium: list[str]  # 2 choices
    descriptions_short: list[str]   # 2 choices
    tags: list[str]                 # ~10 tags
    giftcard_name_suggestions: list[str]  # 5 refactored names


class ImageRequest(BaseModel):
    giftcard_name: str
    description: str


class ImageResponse(BaseModel):
    image_base64: str
    media_type: str = "image/png"


def openai_client():
    get_openai_api_key()  # validate key exists
    return get_client()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/tier1/describe", response_model=DescribeResponse)
def tier1_describe(
    body: DescribeRequest,
    client=Depends(openai_client),
):
    try:
        result = generate_description_and_tag(
            client, body.giftcard_name, body.prompt, "tier1"
        )
        return DescribeResponse(
            descriptions_medium=result["descriptions_medium"],
            descriptions_short=result["descriptions_short"],
            tags=result["tags"],
            giftcard_name_suggestions=result["giftcard_name_suggestions"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tier1/image", response_model=ImageResponse)
def tier1_image(
    body: ImageRequest,
    client=Depends(openai_client),
):
    try:
        b64, media_type = generate_image(
            client, body.giftcard_name, body.description, "tier1"
        )
        return ImageResponse(image_base64=b64, media_type=media_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tier2/describe", response_model=DescribeResponse)
def tier2_describe(
    body: DescribeRequest,
    client=Depends(openai_client),
):
    try:
        result = generate_description_and_tag(
            client, body.giftcard_name, body.prompt, "tier2"
        )
        return DescribeResponse(
            descriptions_medium=result["descriptions_medium"],
            descriptions_short=result["descriptions_short"],
            tags=result["tags"],
            giftcard_name_suggestions=result["giftcard_name_suggestions"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tier2/image", response_model=ImageResponse)
def tier2_image(
    body: ImageRequest,
    client=Depends(openai_client),
):
    try:
        b64, media_type = generate_image(
            client, body.giftcard_name, body.description, "tier2"
        )
        return ImageResponse(image_base64=b64, media_type=media_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
