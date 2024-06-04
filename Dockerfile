# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary files into the container
COPY requirements.txt .

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY demand_predictor ./demand_predictor
COPY raw_data ./raw_data
COPY training_outputs ./training_outputs

# Expose the port that the FastAPI application runs on
EXPOSE 8080

# Command to run the FastAPI application
CMD ["uvicorn", "demand_predictor.api.fast:app", "--host", "0.0.0.0", "--port", "8080"]
