FROM python:3.10-slim

# Set up user with non-root permissions for HF Spaces
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY --chown=user . .

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Launch Gradio app
CMD ["python", "app.py"]