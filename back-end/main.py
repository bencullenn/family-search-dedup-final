import os
import logging
import cv2
import numpy as np
import pytesseract
import math
import json
import requests
import base64
from PIL import Image as PILImage
from io import BytesIO
import torch
from torch import nn
import torchvision.transforms as T
import time

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text
from supabase import create_client, Client
from datetime import datetime, timezone
from tqdm import tqdm
from collections import defaultdict

from helpers.BinaryComparator import BinaryComparator
import asyncio
from concurrent.futures import ThreadPoolExecutor
from schema import User
from model import FlagStatus
from database import get_db
import model

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("fastapi")
logger.setLevel(logging.INFO)
# Remove existing handlers to prevent duplicate logs
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s: %(message)s")  # Simplified format
handler.setFormatter(formatter)
logger.addHandler(handler)

# Print out pytorch information
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")

supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Uncomment this to use DinoV2 for local embedding
"""device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
dinov2_model.to(device)
dinov2_model.eval()"""

# Define image transformation
transform_image = T.Compose(
    [T.ToTensor(), T.Resize(224), T.CenterCrop(224), T.Normalize([0.5], [0.5])]
)


@app.get("/api/binary-compare")
def binary_compare(sourceId: str = Query("001_01.jpg")):
    logger.info("Running binary compare")
    same_photos = []
    comparator = BinaryComparator()
    for root, dirs, files in os.walk("public/duplicates/exact-duplicates"):
        for file in files:
            file_path = os.path.join(root, file)
            src_path = os.path.join(root, sourceId)
            if comparator.compareImages(src_path, file_path):
                same_photos.append(
                    {
                        "src": src_path.replace("public/", "/"),
                        "alt": "Matching photo",
                        "score": 1.0,
                    }
                )
    logger.info(f"Photo matches: {same_photos}")
    return {"result": same_photos, "src_path": src_path}


@app.get("/api/people")
def get_people(db: Session = Depends(get_db)):
    people = (
        db.query(model.Person)
        .order_by(model.Person.first_name, model.Person.last_name)
        .all()
    )

    for person in people:
        person.first_name = person.first_name.replace("_", " ")

    return people


@app.get("/api/clear/{person_id}")
def clear_stack(person_id):
    try:
        # Fetch stacks associated with the person
        stack_result = (
            supabase.table("stack").select("id").eq("person", person_id).execute()
        )

        stack_ids = [stack["id"] for stack in stack_result.data]

        if stack_ids:
            # Remove references from image table first
            for stack_id in stack_ids:
                supabase.table("image").update({"stack": None}).eq(
                    "stack", stack_id
                ).execute()

            # Now delete from stack
            supabase.table("stack").delete().eq("person", person_id).execute()

    except Exception as e:
        logger.error(f"Error clearing stack: {e}")
        raise HTTPException(status_code=500, detail="Database error")


@app.get("/api/photos")
def get_photos():
    photos_directory = os.path.join(
        os.getcwd(), "public", "duplicates", "exact-duplicates"
    )
    try:
        file_names = os.listdir(photos_directory)
        photos = [
            {
                "id": file_name,
                "src": f"/duplicates/exact-duplicates/{file_name}",
                "alt": file_name,
            }
            for file_name in file_names
        ]
        return photos
    except Exception as error:
        print(f"Error reading photos directory: {error}")
        raise HTTPException(status_code=500, detail="Failed to fetch photos")


@app.get("/api/image")
async def get_image(image_id: int, db: Session = Depends(get_db)):
    image = db.query(model.Image).filter(model.Image.id == image_id).first()

    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")

    response = supabase.storage.from_("test-photos").create_signed_url(image.path, 60)

    return response["signedURL"]


def compute_orb_features(image_bytes: bytes) -> str:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create(nfeatures=500)
    keypoints, descriptors = orb.detectAndCompute(img, None)
    if descriptors is None or len(descriptors) == 0:
        raise HTTPException(status_code=500, detail="No ORB features detected")

    # Convert all descriptors to a binary matrix.
    # Each ORB descriptor is originally a 32-byte vector; unpacking will yield 256 bits.
    descriptors_unpacked = np.unpackbits(descriptors, axis=1)  # shape: (N, 256)

    # Aggregate the descriptors into one by doing a bitwise majority vote.
    # For each bit position, if at least half of the descriptors have a 1, set the bit to 1.
    aggregated_bits = (
        np.sum(descriptors_unpacked, axis=0) >= (descriptors_unpacked.shape[0] / 2)
    ).astype(int)
    bit_string = "".join(map(str, aggregated_bits))  # Convert to a string of 0s and 1s

    return bit_string


def count_words(image_bytes: bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return len(text)


# Deprecated
def compute_image_embedding(image_bytes: bytes):
    """
    Compute deep learning embedding for an image using DinoV2.
    Processes images in grayscale but makes them compatible with DinoV2's 3-channel input requirement.
    Returns the raw embedding vector for storage in PGVector.
    """
    try:
        # Convert bytes to PIL Image and ensure it's in grayscale mode
        pil_image = PILImage.open(BytesIO(image_bytes)).convert(
            "L"
        )  # L = grayscale mode

        # Convert to tensor
        grayscale_tensor = T.ToTensor()(pil_image)  # [1, H, W]

        # Replicate the grayscale channel to create a 3-channel grayscale image
        three_channel = grayscale_tensor.repeat(3, 1, 1)  # [3, H, W]

        # Complete the rest of the transformation
        transformed_img = (
            T.Compose(
                [
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )(three_channel)
            .unsqueeze(0)
            .to(device)
        )

        # Generate embedding
        with torch.no_grad():
            embedding = dinov2_model(transformed_img)

        try:
            # First attempt the standard conversion
            return embedding.cpu().numpy()[
                0
            ]  # Get the first item to remove batch dimension
        except Exception as numpy_error:
            # If standard conversion fails, try alternative approach
            logger.warning(f"Standard numpy conversion failed: {numpy_error}")

            try:
                # Alternative 1: Manual conversion using PyTorch's tolist()
                embedding_list = embedding.cpu().detach().tolist()[0]
                return np.array(embedding_list, dtype=np.float32)
            except Exception as alt_error:
                # Alternative 2: Use detach() more carefully
                logger.warning(f"Alternative 1 failed: {alt_error}")
                detached = embedding.cpu().detach()
                numpy_array = np.zeros(detached.shape[1], dtype=np.float32)
                # Manually copy values
                for i in range(detached.shape[1]):
                    numpy_array[i] = float(detached[0, i].item())
                return numpy_array

    except Exception as e:
        logger.error(f"Error computing image embedding: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to compute image embedding: {e}"
        )


def compute_image_embedding_hq(image_bytes: bytes):
    """
    Compute high-quality deep learning embedding for an image using a RunPod endpoint.
    This function sends the image to a remote API instead of loading the model locally.
    Returns a tuple of (original_embedding, hq_embedding) for storage in PGVector.
    """
    try:
        # Uncomment this to use DinoV2 for local embedding
        # original_embedding = compute_image_embedding(image_bytes)

        # Convert image bytes to base64
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Prepare the request to the RunPod endpoint
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('RUNPOD_API_KEY')}",
        }

        data = {"input": {"image": base64_image}}

        # Make the API request to submit the job
        response = requests.post(
            "https://api.runpod.ai/v2/vmmhm63d4av6zk/run",
            headers=headers,
            json=data,
        )

        # Check if the request was successful
        if response.status_code != 200:
            logger.error(f"RunPod API error: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to submit job: API returned status code {response.status_code}",
            )

        # Get the job ID from the response
        job_id = response.json()["id"]
        logger.info(f"Job submitted successfully. Job ID: {job_id}")

        # Poll for completion
        while True:
            # Check job status
            status_response = requests.get(
                f"https://api.runpod.ai/v2/vmmhm63d4av6zk/status/{job_id}",
                headers=headers,
            )

            if status_response.status_code != 200:
                logger.error(
                    f"RunPod status API error: {status_response.status_code} - {status_response.text}"
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to check job status: API returned status code {status_response.status_code}",
                )

            status_data = status_response.json()
            status = status_data["status"]

            if status == "COMPLETED":
                # Extract the embedding from the response
                hq_embedding = status_data["output"]["embedding"]
                return np.array(hq_embedding, dtype=np.float32)
            elif status == "FAILED":
                logger.error(f"Job failed: {status_data}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Job failed: {status_data.get('error', 'Unknown error')}",
                )
            else:
                # Wait before checking again
                time.sleep(2)

    except Exception as e:
        logger.error(f"Error computing image embedding with RunPod: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to compute image embedding: {str(e)}"
        )


@app.post("/api/create-stacks")
async def create_matches(
    person_id: int = Form(...),
    match_threshold: float = Form(0.95),
    recompute: bool = Form(False),
    db: Session = Depends(get_db),
):
    try:
        # Get all the images for the person that don't have a stack
        if recompute:
            images = db.query(model.Image).filter(model.Image.person == person_id).all()
        else:
            images = (
                db.query(model.Image)
                .filter(model.Image.person == person_id)
                .filter(model.Image.stack is None)
                .all()
            )

        # For each one run it through the match endpoint with a threshold of 70%
        for image in tqdm(images, desc="Processing images"):
            try:
                # Skip images with invalid paths
                if not image.path or image.path.startswith("None/"):
                    logger.warning(f"Skipping image with invalid path: {image.path}")
                    continue

                # Download the image using the path
                try:
                    image_bytes = supabase.storage.from_("test-photos").download(
                        image.path
                    )
                except Exception as download_error:
                    logger.error(
                        f"Failed to download image {image.path}: {str(download_error)}"
                    )
                    continue

                image_file = UploadFile(file=BytesIO(image_bytes), filename=image.path)

                try:
                    # Compute ORB features for the image
                    orb_features = compute_orb_features(image_bytes)

                    # Compute image embeddings using both methods
                    hq_embedding = compute_image_embedding_hq(image_bytes)

                    # Update both ORB features and image embeddings in the database
                    image.orb_features = orb_features
                    image.image_embedding_hq = hq_embedding
                    db.commit()
                except Exception as compute_error:
                    logger.error(
                        f"Failed to compute features for image {image.path}: {str(compute_error)}"
                    )
                    continue

                try:
                    # Get matches using the image embeddings
                    result = await get_matches(
                        file=image_file,
                        person_id=person_id,
                        match_threshold=match_threshold,
                        match_count=2,
                        db=db,
                    )

                    # Filter out the current image from the result
                    result = [r for r in result if r[0] != image.path]

                    logger.info(f"Match result for image {image.id}: {result}")

                    # If no matches are found, create a new stack
                    if len(result) == 0 or image.exclude:
                        logger.info(f"Creating new stack for image {image.id}")
                        new_stack = model.Stack(
                            person=person_id,
                            name=image.path,
                            created_at=datetime.now(timezone.utc),
                            thumbnail=image.id,
                        )
                        db.add(new_stack)
                        db.commit()
                        db.refresh(new_stack)

                        # Update the image to have the new stack_id
                        image.stack = new_stack.id
                        db.commit()

                    # Add the current image to the stack by updating the stack_id in the database
                    else:
                        logger.info(
                            f"Adding image {image.id} to existing stack {result[0][2]}"
                        )
                        stack_id = result[0][2]
                        image.stack = stack_id
                        db.commit()

                except Exception as match_error:
                    logger.error(
                        f"Failed to process matches for image {image.path}: {str(match_error)}"
                    )
                    continue

            except Exception as image_error:
                logger.error(f"Failed to process image {image.id}: {str(image_error)}")
                continue

        return {"result": "Stacks created", "status": "success"}

    except Exception as e:
        logger.error(f"Error in create_matches: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"message": "Failed to create stacks", "error": str(e)},
        )


@app.post("/api/matches")
async def get_matches(
    file: UploadFile = File(...),
    person_id: int = Form(...),
    match_threshold: float = Form(0.7),  # Changed to float, default 0.7
    match_count: int = Form(10),
    db: Session = Depends(get_db),
) -> list:
    # Read and decode the uploaded image file
    file_bytes = await file.read()
    if file_bytes is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Compute image embeddings
    hq_embedding = compute_image_embedding_hq(file_bytes)
    print("HQ embedding: ", hq_embedding)

    # Convert to a list - using the high-quality embedding for matching
    image_embedding_list = hq_embedding.tolist()

    # Execute the SQL function using image embeddings
    stmt = text("""
        SELECT * FROM match_image_dino_hq(
            :person_id,
            CAST(:query_embedding AS vector(1536)),
            :match_threshold,
            :match_count
        )
    """)

    # Log the values being sent to SQL for debugging
    logger.info(f"Matching with threshold: {match_threshold}, count: {match_count}")

    result = db.execute(
        stmt,
        {
            "person_id": person_id,
            "query_embedding": image_embedding_list,
            "match_threshold": float(match_threshold),  # Ensure it's a float
            "match_count": match_count,
        },
    )

    matches = result.fetchall()
    logger.info(f"Raw matches from database: {matches}")

    # Get person details for folder path
    person = db.query(model.Person).filter(model.Person.id == person_id).first()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    logger.info(f"Person found: {person.first_name} {person.last_name}")

    # Transform matches into the expected format
    formatted_matches = []
    for match in matches:
        try:
            path = match[1]  # Get the path from the match
            logger.info(f"Original path: {path}")

            if not path or path.startswith("None/"):
                # Skip invalid paths
                logger.warning(f"Skipping invalid path: {path}")
                continue

            if not path.startswith("whole-people/"):
                # Fix path if it's not properly formatted
                path = f"whole-people/{person.first_name.replace(' ', '_')}_{person.last_name}/{path}"

            logger.info(f"Attempting to get signed URL for path: {path}")

            # Get signed URL for the image
            signed_url = supabase.storage.from_("test-photos").create_signed_url(
                path,  # Use the corrected path
                1800,
            )

            logger.info(f"Successfully got signed URL for {path}")

            formatted_matches.append(
                [
                    path,  # title/path
                    signed_url["signedURL"],  # url
                    match[2],  # stack_id
                ]
            )
        except Exception as e:
            logger.error(f"Failed to get signed URL for match {match}: {str(e)}")
            continue

    logger.info(f"Final formatted matches: {formatted_matches}")
    return formatted_matches


@app.post("/api/upload-image")
async def upload_orb(
    title: str = Form(None),
    person: str = Form(...),
    stack_id: int = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    try:
        # Read file bytes
        file_bytes = await file.read()
        if file_bytes is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Get person details
        person = db.query(model.Person).filter(model.Person.id == int(person)).first()
        if person is None:
            raise HTTPException(status_code=404, detail="Person not found")

        # Format the folder name
        folder_name = (
            f"whole-people/{person.first_name.replace(' ', '_')}_{person.last_name}"
        )
        file_path = f"{folder_name}/{file.filename}"

        # Upload to Supabase storage
        response = supabase.storage.from_("test-photos").upload(
            file_path, file_bytes, file_options={"content-type": file.content_type}
        )

        if isinstance(response, dict) and "error" in response:
            raise HTTPException(status_code=500, detail=str(response["error"]))

        # Compute ORB features (keeping original functionality)
        bit_string = compute_orb_features(file_bytes)

        # Compute image embeddings
        hq_embedding = compute_image_embedding_hq(file_bytes)

        # exclude = count_words(image_bytes=file_bytes) > 100

        # Create new image record
        new_image = model.Image(
            orb_features=bit_string,
            image_embedding_hq=hq_embedding,
            stack=stack_id,
            path=file_path,
            person=person.id,
            title=title,
            created_at=datetime.now(timezone.utc),
        )

        db.add(new_image)
        db.commit()
        db.refresh(new_image)

        return {
            "success": True,
            "detail": "Image uploaded successfully",
            "image_id": new_image.id,
            "path": file_path,
        }

    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        db.rollback()
        # Try to delete the uploaded file if database operation failed
        try:
            supabase.storage.from_("test-photos").remove([file_path])
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/person-images")
async def get_person_images(
    person_id: int, user_id: int = Query(...), db: Session = Depends(get_db)
):
    logger.debug("=== Starting get_person_images ===")
    logger.debug(f"Parameters: person_id={person_id}, user_id={user_id}")

    # Get person details first
    person = db.query(model.Person).filter(model.Person.id == person_id).first()
    if not person:
        logger.error(f"Person {person_id} not found")
        raise HTTPException(status_code=404, detail="Person not found")

    # First get all images to see what we're working with
    all_images = db.query(model.Image).filter(model.Image.person == person_id).all()
    logger.debug(f"Total images found for person {person_id}: {len(all_images)}")
    logger.debug(f"First few images: {[img.id for img in all_images[:5]]}")

    # Update query to handle NULL flag_status correctly
    images = (
        db.query(model.Image.id, model.Image.path, model.Image.stack)
        .filter(model.Image.person == person_id)
        .filter(
            (model.Image.flag_status.is_(None))  # Include images with no flag
            | (
                model.Image.flag_status != FlagStatus.FLAGGED
            )  # Include non-flagged images
            | (
                (model.Image.flag_status == FlagStatus.FLAGGED)
                & (
                    model.Image.flagged_by != user_id
                )  # Include flagged images by other users
            )
        )
        .all()
    )

    logger.debug(f"Images after filtering: {len(images)}")
    if images:
        logger.debug(f"First few filtered image IDs: {[img.id for img in images[:5]]}")

    # Log stack information
    stacks = set(img.stack for img in images if img.stack is not None)
    logger.debug(f"Stack IDs: {list(stacks)}")

    async def get_signed_url(img):
        try:
            path = img.path
            if not path.startswith("whole-people/"):
                path = f"whole-people/{person.first_name.replace(' ', '_')}_{person.last_name}/{path}"

            with ThreadPoolExecutor() as executor:
                signed_url = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda: supabase.storage.from_("test-photos").create_signed_url(
                        path, 1800
                    ),
                )
            return (img.stack, img.id, signed_url["signedURL"])
        except Exception as e:
            logger.error(f"Failed to get signed URL for image {img.path}: {str(e)}")
            return None

    # Gather all signed URL requests concurrently
    results = await asyncio.gather(*[get_signed_url(img) for img in images])

    # Process results into stacks
    person_images = defaultdict(list)
    for result in results:
        if result:
            stack_id, img_id, url = result
            # Include flag status in the response
            person_images[stack_id].append(
                {
                    "id": img_id,
                    "url": url,
                    "flag_status": None,  # Default to None
                    "flagged_by": None,  # Default to None
                }
            )

    # Filter out empty stacks and convert to list
    result = [
        [
            {
                "id": item["id"],
                "url": item["url"],
                "flag_status": item.get("flag_status"),
                "flagged_by": item.get("flagged_by"),
            }
            for item in urls
        ]
        for urls in person_images.values()
        if urls
    ]

    return result


@app.get("/api/users")
def get_users(db: Session = Depends(get_db)):
    try:
        result = supabase.table("user").select("*").execute()
        return result.data
    except Exception as e:
        logger.error(f"Error fetching users: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch users")


@app.post("/api/flag-image")
async def flag_image(
    image_id: int = Form(...), user_id: int = Form(...), db: Session = Depends(get_db)
):
    try:
        # Update the image record
        result = (
            supabase.table("image")
            .update({"flag_status": "FLAGGED", "flagged_by": user_id})
            .eq("id", image_id)
            .execute()
        )

        if not result.data:
            raise HTTPException(status_code=404, detail="Image not found")

        return {"success": True}
    except Exception as e:
        logger.error(f"Error flagging image: {e}")
        raise HTTPException(status_code=500, detail="Failed to flag image")


@app.get("/api/flagged-images")
async def get_flagged_images(
    person_id: int, user_id: int = Query(...), db: Session = Depends(get_db)
):
    try:
        logger.info(f"Fetching flagged images for person {person_id} by user {user_id}")

        # Get images that are flagged by others
        images = (
            db.query(model.Image)
            .filter(model.Image.person == person_id)
            .filter(
                (model.Image.flag_status == FlagStatus.FLAGGED)
                & (model.Image.flagged_by != user_id)
            )
            .all()
        )

        logger.info(f"Found {len(images)} flagged images")

        if not images:
            return []

        # Get the stack information and signed URLs
        result = []
        for img in images:
            stack_images = (
                db.query(model.Image).filter(model.Image.stack == img.stack).all()
            )

            stack_urls = []
            for stack_img in stack_images:
                try:
                    signed_url = supabase.storage.from_(
                        "test-photos"
                    ).create_signed_url(stack_img.path, 1800)
                    stack_urls.append(
                        {
                            "id": stack_img.id,
                            "url": signed_url["signedURL"],
                            "is_flagged": stack_img.id == img.id,
                            "flagged_by": stack_img.flagged_by,
                        }
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to get signed URL for image {stack_img.path}: {str(e)}"
                    )

            if stack_urls:
                result.append(
                    {
                        "stack_id": img.stack,
                        "flagged_image_id": img.id,
                        "flagged_by": img.flagged_by,
                        "images": stack_urls,
                    }
                )

        return result
    except Exception as e:
        logger.error(f"Error getting flagged images: {e}")
        raise HTTPException(status_code=500, detail="Failed to get flagged images")


@app.post("/api/unflag-image")
async def unflag_image(image_id: int = Form(...), db: Session = Depends(get_db)):
    try:
        # Remove flag and flagged_by
        result = (
            supabase.table("image")
            .update({"flag_status": None, "flagged_by": None})
            .eq("id", image_id)
            .execute()
        )

        if not result.data:
            raise HTTPException(status_code=404, detail="Image not found")

        return {"success": True}
    except Exception as e:
        logger.error(f"Error unflagging image: {e}")
        raise HTTPException(status_code=500, detail="Failed to unflag image")


@app.post("/api/remove-from-stack")
async def remove_from_stack(image_id: int = Form(...), db: Session = Depends(get_db)):
    try:
        # Remove stack assignment and clear flag
        result = (
            supabase.table("image")
            .update({"stack": None, "flag_status": None, "flagged_by": None})
            .eq("id", image_id)
            .execute()
        )

        if not result.data:
            raise HTTPException(status_code=404, detail="Image not found")

        return {"success": True}
    except Exception as e:
        logger.error(f"Error removing image from stack: {e}")
        raise HTTPException(status_code=500, detail="Failed to remove image from stack")
