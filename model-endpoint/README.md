# DinoV2 Model Endpoint

## Overview

This container uses the DinoV2 model to generate image embeddings. The model is used to process grayscale images and generate deep learning embeddings that can be used for various downstream tasks.

## Building and Deploying Container to RunPod

### Build Container

1. Install Docker Desktop if you don't already have it: https://www.docker.com/products/docker-desktop/

2. Create a repository to upload and share your containers with RunPod. I used a free public repository on Docker Hub (https://hub.docker.com/repositories) for simplicity.

3. Once inside the docker directory, run the following command to build the image:

```bash
docker build --platform linux/amd64 --tag <username>/<repo>:<tag> .
```

4. Push the Docker image to your container registry:

```bash
docker push <username>/<repo>:<tag>
```

### Deploying to RunPod

1. Open RunPod dashboard.

2. Navigate to the Serverless tab and select the purple "New Endpoint" button.

3. Select the Docker Image option under the "Custom Source" section and input the path to the Docker image you pushed. Click "Next" to continue.

4. Select a GPU with sufficient VRAM for your needs. The DinoV2 model is relatively lightweight compared to larger models.

5. Click "Create Endpoint" and your endpoint will be deployed in a few minutes.

6. Once the endpoint is deployed, you can test it using the client in the 'client' directory. A README is included in the client directory for more details on how to use it.

## API Usage

The endpoint accepts POST requests with the following structure:

```json
{
  "input": {
    "image": "<base64_encoded_image_bytes>"
  }
}
```

The response will contain the image embedding as a list of floating-point numbers:

```json
{
    "embedding": [0.123, 0.456, ...]
}
```

## Technical Details

### Image Processing

The handler processes images in the following way:

1. Converts input image bytes to a PIL Image
2. Ensures the image is in grayscale mode
3. Transforms the image to be compatible with DinoV2's requirements:
   - Resizes to 224x224
   - Converts to a 3-channel format (replicating grayscale channel)
   - Normalizes pixel values
4. Generates embeddings using the DinoV2 model
5. Returns the embedding as a numpy array converted to a list

### Error Handling

The handler includes robust error handling for:

- Image processing errors
- Model inference errors
- Numpy conversion errors

If any errors occur during processing, they will be caught and returned with appropriate error messages.

## Dependencies

The main dependencies for this project are:

- PyTorch
- PIL (Python Imaging Library)
- NumPy
- RunPod

These are specified in the requirements.txt file and will be installed during the container build process.
