# Use the official PyTorch Docker image as the base
FROM pytorch/pytorch:latest

# Install Jupyter Notebook and JupyterLab
RUN pip install jupyter jupyterlab

# Install hugging face libraries
RUN pip install diffusers[torch] huggingface_hub
RUN pip install transformers
RUN pip install xformers==0.0.16
RUN pip install bitsandbytes-cuda117
RUN pip install accelerate


RUN pip install torchvision ftfy tensorboard Jinja2

# Set the working directory to /app
WORKDIR /app

# Copy the contents of the current directory into the container at /app
COPY . /app

# Set the default command to run JupyterLab when the container starts
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

