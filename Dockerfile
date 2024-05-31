# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire demand_predictor folder into the container
COPY demand_predictor demand_predictor

# Copy other necessary files (e.g., setup.py if needed)
COPY setup.py setup.py

# Install the demand_predictor package
RUN pip install .

# Expose the port FastAPI will run on
EXPOSE 8001

# Command to run the FastAPI application
CMD ["uvicorn", "demand_predictor.api.fast:app", "--host", "0.0.0.0", "--port", "8001"]
