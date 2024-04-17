FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    texlive \
    texlive-latex-extra \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT [ "streamlit" ]

CMD [ "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]


