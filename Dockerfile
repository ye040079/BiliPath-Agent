FROM python:3.12-slim

WORKDIR /app

# 先装依赖，利用 Docker 层缓存
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 再拷贝源码
COPY . .

EXPOSE 7860

# 运行时通过环境变量注入 DEEPSEEK_API_KEY（见 .env.example）
# docker run -e DEEPSEEK_API_KEY=sk-xxx -p 7860:7860 study-agent
CMD ["python", "main.py"]
