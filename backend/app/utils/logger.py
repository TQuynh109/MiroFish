"""
Mô-đun cấu hình logging
Cung cấp quản lý log thống nhất, đồng thời ghi ra console và file
"""

import os
import sys
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler


def _ensure_utf8_stdout():
    """
    Đảm bảo stdout/stderr sử dụng mã hóa UTF-8
    Khắc phục lỗi vỡ font tiếng Trung trên console Windows
    """
    if sys.platform == 'win32':
        # Trên Windows, cấu hình lại stdout/stderr sang UTF-8
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# Thư mục log
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')


def setup_logger(name: str = 'mirofish', level: int = logging.DEBUG) -> logging.Logger:
    """
    Thiết lập logger

    Args:
        name: Tên logger
        level: Mức độ log

    Returns:
        Logger đã được cấu hình
    """
    # Đảm bảo thư mục log tồn tại
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Tạo logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Chặn log propagate lên root logger để tránh in trùng lặp
    logger.propagate = False
    
    # Nếu đã có handler thì không thêm lại
    if logger.handlers:
        return logger
    
    # Định dạng log
    detailed_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # 1. File handler - log chi tiết (đặt tên theo ngày, có rolling)
    log_filename = datetime.now().strftime('%Y-%m-%d') + '.log'
    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, log_filename),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # 2. Console handler - log gọn (INFO trở lên)
    # Đảm bảo Windows dùng UTF-8 để tránh lỗi ký tự
    _ensure_utf8_stdout()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Gắn handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = 'mirofish') -> logging.Logger:
    """
    Lấy logger (nếu chưa có sẽ tạo mới)

    Args:
        name: Tên logger

    Returns:
        Instance logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


# Tạo logger mặc định
logger = setup_logger()


# Hàm gọi nhanh
def debug(msg, *args, **kwargs):
    logger.debug(msg, *args, **kwargs)

def info(msg, *args, **kwargs):
    logger.info(msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
    logger.warning(msg, *args, **kwargs)

def error(msg, *args, **kwargs):
    logger.error(msg, *args, **kwargs)

def critical(msg, *args, **kwargs):
    logger.critical(msg, *args, **kwargs)

