import json
import httpx
from pydantic import BaseModel, Field
from fastapi import FastAPI
import fastapi
import transformers
from typing import Dict, Any, Optional

app = FastAPI()


class PredictRequest(BaseModel):
    hf_pipeline: str
    model_deployed_url: str
    inputs: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


@app.post(path="/predict")
async def predict(request: PredictRequest):
     """
    Translates the input into V2 protocol and sends it to model_deployed_url.

    Args:
        request: The request object.

    Returns:
        The response from model_deployed_url.
    """

    inputs = transformers.hf_to_v2_inference_protocol(request.hf_pipeline, request.inputs, request.parameters)
    response = httpx.post(
        request.model_deployed_url,
        headers={"Content-Type": "application/json"},
        json=inputs,
    )

    return response.json()