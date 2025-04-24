import runpod
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import torchvision.transforms as T
import base64
import traceback

# Initialize the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model directly from torch hub and ensure it's on GPU
print("Loading DinoV2 model from torch hub")
dinov2_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14_reg")
dinov2_model.to(device)  # Explicitly move to GPU
dinov2_model.eval()
print(
    f"Model loaded successfully and is on device: {next(dinov2_model.parameters()).device}"
)


# Image transformation for DinoV2 Giant
def get_transform():
    return T.Compose(
        [
            T.Resize(518),
            T.CenterCrop(518),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def compute_image_embedding(image_bytes):
    """
    Compute embedding using DinoV2 Giant with registers
    """
    try:
        # Keep original RGB channels
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        print(f"Image loaded, size: {pil_image.size}")

        # Transform image
        transform = get_transform()
        transformed_img = transform(pil_image).unsqueeze(0).to(device)
        print(f"Image transformed, shape: {transformed_img.shape}")

        # Generate embedding
        with torch.no_grad(), torch.cuda.amp.autocast():
            embedding = dinov2_model(transformed_img)
            print(f"Embedding generated, shape: {embedding.shape}")

        # Convert to numpy
        numpy_embedding = embedding.cpu().numpy().squeeze()
        return numpy_embedding

    except Exception as e:
        print(f"Error in embedding computation: {e}")
        print(traceback.format_exc())
        raise Exception(f"Failed to compute embedding: {e}")


def handler(event):
    try:
        print(f"Handler received event: {event}")
        input_data = event["input"]

        # Get and decode base64 image
        image_b64 = input_data.get("image")
        if not image_b64:
            return {"error": "No image provided for embedding"}

        try:
            print("Decoding base64 image")
            image_bytes = base64.b64decode(image_b64)
            print(f"Image decoded: {len(image_bytes)} bytes")
        except Exception as e:
            print(f"Failed to decode base64: {e}")
            # Fallback to raw data in case it's already binary
            image_bytes = image_b64

        embedding = compute_image_embedding(image_bytes)
        return {"embedding": embedding.tolist()}

    except Exception as e:
        print(f"Handler error: {e}")
        print(traceback.format_exc())
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
