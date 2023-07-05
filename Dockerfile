# Use an official detectron2 runtime as a parent image
FROM gaseooo/detectron2:1.0.1

# Set the working directory in the container to /app
WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y \
    wget \
    git

# Install python dependencies
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
RUN python -m pip install --upgrade pip

# Clone the EVA/det repository and install it
RUN git clone https://github.com/baaivision/EVA.git
WORKDIR /app/EVA/EVA-01/det
RUN python -m pip install -e .

# Go back to the working directory
WORKDIR /app

# At runtime, run python command
CMD ["python3"]

