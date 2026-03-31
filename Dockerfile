FROM rayproject/ray:2.46.0-py312-gpu

WORKDIR /home/ray/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir scipy>=1.12.0

COPY src/ ./src/

# Pre-download ResNet-152 weights into the image to avoid runtime network issues
RUN python -c "from torchvision.models import resnet152, ResNet152_Weights; resnet152(weights=ResNet152_Weights.DEFAULT)"

ENV PYTHONPATH=/home/ray/app
