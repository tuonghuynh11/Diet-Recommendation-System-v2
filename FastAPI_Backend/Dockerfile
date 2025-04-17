# backend/Dockerfile

FROM python:3.10.8

RUN mkdir -p app/backend

COPY FastAPI_Backend/requirements.txt app/backend/requirements.txt
COPY Data app/Data

COPY FastAPI_Backend app/backend

WORKDIR /app/backend

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 7071

CMD ["uvicorn","main:app","--host","0.0.0.0","--port","7071"]



# docker build --progress=plain -t diet-recommendation-system -f FastAPI_Backend/Dockerfile .
# docker run -d -p 7071:7071 --name diet-recommendation-service diet-recommendation-system
