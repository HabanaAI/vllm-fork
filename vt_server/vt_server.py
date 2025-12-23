import argparse
import torch
import base64
from io import BytesIO
from urllib.parse import urlparse
#from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel
import numpy as np
import uvicorn


# Initialize FastAPI app
app = FastAPI(title="SigLIP Vision Encoder Service")

# Global variables for model and processor
model = None
processor = None
device = None
model_name_to_display = "SigLIP"


class TensorRequest(BaseModel):
    tensor: str  # Base64 encoded numpy array
    shape: list[int]  # Original tensor shape


class EmbeddingResponse(BaseModel):
    embeddings: str  # Base64 encoded numpy array
    shape: list[int]  # Shape of the embeddings tensor


@app.on_event("startup")
async def load_model():
    global model, processor, device

    # load SigLIP model
    full_model_name = "google/gemma-3-12b-it"
    full_model = AutoModel.from_pretrained(full_model_name)
    model = full_model.vision_tower

    device = torch.device("hpu")
    model = model.to(device)
    model.eval()

    print(f"+-+ loaded: hidden_size={model.config.hidden_size}, num_layers={model.config.num_hidden_layers}, image_size={model.config.image_size}")
    

@app.get("/health")
async def health_check():
    res = {
        "status": "ready" if model is not None else "Model not loaded",
        "model": model_name_to_display,
        "device": str(device),
    }
    return res


@app.post("/encode", response_model=EmbeddingResponse)
async def encode_images(request: TensorRequest):
    print("+-+ vt_server.py::encode_images() enter")

    # Decode base64 tensor
    tensor_bytes = base64.b64decode(request.tensor)
    buffer = BytesIO(tensor_bytes)
    pixel_values = np.load(buffer)
    pixel_values = torch.from_numpy(pixel_values).to(device)
    print(f"+-+ pixel_values, shape: {pixel_values.shape}, dtype: {pixel_values.dtype}, device={pixel_values.device}")

    # Run inference
    with torch.no_grad():
        output = model(pixel_values)
    embeddings = output.last_hidden_state
    embeddings_cpu = embeddings.to(torch.float16).to("cpu").numpy()
    print(f"+-+ embeddings, shape: {embeddings.shape}, dtype: {embeddings.dtype}, device={embeddings.device}")

    # Prepare response
    buffer = BytesIO()
    np.save(buffer, embeddings_cpu)
    embeddings_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    shape = list(embeddings_cpu.shape)
    
    res = EmbeddingResponse(
        embeddings=embeddings_base64,
        shape=shape
    )

    print("+-+ vt_server.py::encode_images() exit")
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vision Encoder Service')
    parser.add_argument("--server-url", default="http://localhost:8000", help="Server URL")
    args = parser.parse_args()
    
    # Parse the URL to extract host and port
    parsed_url = urlparse(args.server_url)
    host = parsed_url.hostname or "0.0.0.0"
    port = parsed_url.port or 8000
    
    uvicorn.run(app, host=host, port=port)
