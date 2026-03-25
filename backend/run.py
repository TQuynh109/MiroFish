"""
MiroFish Backend entrypoint
"""

import os
import sys

# Khắc phục lỗi hiển thị tiếng Trung trên console Windows: đặt UTF-8 trước mọi import
if sys.platform == 'win32':
    # Đặt biến môi trường để đảm bảo Python dùng UTF-8
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    # Cấu hình lại stdout/stderr sang UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Thêm thư mục gốc của project vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.config import Config


def main():
    """Hàm chính"""
    # Kiểm tra cấu hình
    errors = Config.validate()
    if errors:
        print("Configuration errors:")
        for err in errors:
            print(f"  - {err}")
        print("\nPlease check configuration in the .env file")
        sys.exit(1)
    
    # Tạo ứng dụng
    app = create_app()
    
    # Lấy cấu hình chạy
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5001))
    debug = Config.DEBUG
    
    # Khởi động dịch vụ
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    main()

