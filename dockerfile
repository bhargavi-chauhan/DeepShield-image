FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Install dependencies
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Start container in bash so user can run any script
CMD ["/bin/bash"]