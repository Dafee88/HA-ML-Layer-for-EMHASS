FROM python:3.12-slim-bookworm

WORKDIR /app

COPY requirements.txt .
RUN echo "cache-bust-2026-09-24" && pip install --no-cache-dir -r requirements.txt

COPY app/ .
COPY run.sh /run.sh
RUN chmod a+x /run.sh

CMD ["/run.sh"]