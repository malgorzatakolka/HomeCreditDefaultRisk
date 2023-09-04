# Start from a base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY ./requirements.txt /app/requirements.txt

# Install the required packages
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy the application code
COPY ./app.py /app/

# Copy the model
COPY ./model.joblib /app/

# Expose the app port
EXPOSE 8080

# Run command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]