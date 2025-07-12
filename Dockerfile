# 1. Use an official Python base image
FROM python:3.10

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy local files into the container
COPY . .

# 4. Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install requests

# 5. Expose the API port
EXPOSE 8000

# 6. Run the API using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
