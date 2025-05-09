FROM python:3.12-slim
COPY server.py /server.py
WORKDIR /
ENTRYPOINT ["python", "/server.py"]
CMD ["--ip", "0.0.0.0", "--port", "13338"]
