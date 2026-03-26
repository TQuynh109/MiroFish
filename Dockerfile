FROM python:3.11

# Cài đặt Node.js (đáp ứng >=18) và các công cụ cần thiết
RUN apt-get update \
  && apt-get install -y --no-install-recommends nodejs npm \
  && rm -rf /var/lib/apt/lists/*

# Sao chép uv từ image chính thức của uv
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

WORKDIR /app

# Sao chép trước file mô tả dependency để tận dụng cache
COPY package.json package-lock.json ./
COPY frontend/package.json frontend/package-lock.json ./frontend/
COPY backend/pyproject.toml backend/uv.lock ./backend/

# Cài đặt dependency (Node + Python)
RUN npm ci \
  && npm ci --prefix frontend \
  && cd backend && uv sync --frozen

# Sao chép mã nguồn dự án
COPY . .

EXPOSE 3000 5001

# Khởi chạy đồng thời frontend và backend (chế độ phát triển)
CMD ["npm", "run", "dev"]