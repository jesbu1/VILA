import argparse
import base64
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from typing import List, Literal, Optional, Union, get_args

import requests
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image as PILImage
from PIL.Image import Image
from pydantic import BaseModel

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from server import ChatMessage, TextContent, ImageURL, ImageContent, get_literal_values, load_image



IMAGE_CONTENT_BASE64_REGEX = re.compile(r"^data:image/(png|jpe?g);base64,(.*)$")
PATH_MODEL_NAME = "vila_3b_oxe_no_droid"
PATH_MASK_MODEL_NAME = "vila_3b_oxe_no_droid_path_mask"



class ChatCompletionRequest(BaseModel):
    model: Literal[
        "VILA1.5-3B",
        "VILA1.5-3B-AWQ",
        "VILA1.5-3B-S2",
        "VILA1.5-3B-S2-AWQ",
        "Llama-3-VILA1.5-8B",
        "Llama-3-VILA1.5-8B-AWQ",
        "VILA1.5-13B",
        "VILA1.5-13B-AWQ",
        "VILA1.5-40B",
        "VILA1.5-40B-AWQ",
        "HAMSTER-13B",
        "vila_3b_oxe_no_droid",
        "vila_3b_oxe_no_droid_path_mask"
    ]
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 0.9
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = False
    use_cache: Optional[bool] = True
    num_beams: Optional[int] = 1



model_path_only = None
model_path_mask = None
tokenizer = None
image_processor = None
context_len = None


VILA_MODELS = get_literal_values(ChatCompletionRequest, "model")


def normalize_image_tags(model, qs: str) -> str:
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)

    if DEFAULT_IMAGE_TOKEN not in qs:
        print("No image was found in input messages. Continuing with text only prompt.")
    return qs


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_path_mask, model_path_only, tokenizer, image_processor, context_len
    disable_torch_init()
    
    if not app.args.model_paths:
        raise ValueError("At least one model path must be provided via --model-paths")
    for model_path in app.args.model_paths:
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, None)
        if "mask" in model_path:
            model_path_mask = model
            model_name = PATH_MASK_MODEL_NAME
        else:
            model_path_only = model
            model_name = PATH_MODEL_NAME
        print(f"Model {model_name} loaded successfully. Context length: {context_len}")
    yield


app = FastAPI(lifespan=lifespan)


# Load model upon startup
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        global model_path_mask, model_path_only, tokenizer, image_processor, context_len

        if request.model != PATH_MASK_MODEL_NAME and request.model != PATH_MODEL_NAME:
            raise ValueError(
                f"The endpoint is configured to use the model {PATH_MASK_MODEL_NAME} or {PATH_MODEL_NAME}, "
                f"but the request model is {request.model}"
            )
        
        if request.model == PATH_MASK_MODEL_NAME:
            model = model_path_mask
        else:
            model = model_path_only
            
        config, device = model.config, model.device
        max_tokens = request.max_tokens
        temperature = request.temperature
        top_p = request.top_p
        use_cache = request.use_cache
        num_beams = request.num_beams

        messages = request.messages
        conv_mode = app.args.conv_mode

        images = []

        conv = conv_templates[conv_mode].copy()
        user_role = conv.roles[0]
        assistant_role = conv.roles[1]

        for message in messages:
            if message.role == "user":
                prompt = ""

                if isinstance(message.content, str):
                    prompt += message.content
                if isinstance(message.content, list):
                    for content in message.content:
                        if content.type == "text":
                            prompt += content.text
                        if content.type == "image_url":
                            image = load_image(content.image_url.url)
                            images.append(image)
                            prompt += IMAGE_PLACEHOLDER

                normalized_prompt = normalize_image_tags(model, prompt)
                conv.append_message(user_role, normalized_prompt)
            if message.role == "assistant":
                prompt = message.content
                conv.append_message(assistant_role, prompt)

        prompt_text = conv.get_prompt()
        print("Prompt input: ", prompt_text)

        # support generation with text only inputs
        if len(images) == 0:
            images_input = None
        else:
            images_tensor = process_images(images, image_processor, config).to(device, dtype=torch.float16)
            images_input = [images_tensor]

        input_ids = (
            tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(device)
        )

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_input,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_tokens,
                use_cache=use_cache,
                stopping_criteria=[stopping_criteria],
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        print("\nAssistant: ", outputs)

        resp_content = [TextContent(type="text", text=outputs)]
        return {
            "id": uuid.uuid4().hex,
            "object": "chat.completion",
            "created": time.time(),
            "model": request.model,
            "choices": [{"message": ChatMessage(role="assistant", content=resp_content)}],
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


if __name__ == "__main__":

    host = os.getenv("VILA_HOST", "0.0.0.0")
    port = os.getenv("VILA_PORT", 8000)
    # model_path = os.getenv("VILA_MODEL_PATH", "Efficient-Large-Model/VILA1.5-3B") # Removed single path env var
    conv_mode = os.getenv("VILA_CONV_MODE", "vicuna_v1")
    workers = os.getenv("VILA_WORKERS", 1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=host)
    parser.add_argument("--port", type=int, default=port)
    parser.add_argument("--model-paths", type=str, nargs='+', required=True, help="List of paths to the models")
    parser.add_argument("--conv-mode", type=str, default=conv_mode)
    parser.add_argument("--workers", type=int, default=workers)
    app.args = parser.parse_args()

    uvicorn.run(app, host=host, port=port, workers=workers, log_level="debug")
