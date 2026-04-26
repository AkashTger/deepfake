FROM python:3.10-slim

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# Set the working directory
WORKDIR $HOME/app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY --chown=user backend/requirements.txt $HOME/app/backend/requirements.txt
RUN pip install --no-cache-dir -r $HOME/app/backend/requirements.txt

# Copy the rest of the application
COPY --chown=user . $HOME/app

# Switch to the non-root user
USER user

# Expose the port that Hugging Face Spaces uses
EXPOSE 7860

# Run the FastAPI server
CMD ["python", "backend/server.py"]
