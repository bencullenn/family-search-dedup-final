# family-search-dedup

This is the code for our family search deduplication project.

## Installation

### Front End

- Go into the front-end folder and run `npm install` to install all the dependencies.
- Run `npm run dev` to start the front end server.

### Back End

- If you don't have UV installed, install it from [here](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer).
- Go into back end folder and run `uv sync` to create a virtual environment and install all the dependencies.
- Run `uv pip install torch torchvision` to install the torch and torchvision libraries (they don't play nice with uv sync).
- Run `fastapi dev main.py` to start the back end server.
