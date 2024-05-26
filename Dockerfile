# 
FROM python:3.12.2

# 
WORKDIR /code

# 
COPY requirement.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./app /code/app
ENV PORT 8080
# 
CMD ["fastapi", "run", "app/main.py", "--port", "8080"]