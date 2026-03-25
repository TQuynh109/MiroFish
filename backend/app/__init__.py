"""
MiroFish Backend - Flask application factory
"""

import os
import warnings

# Ẩn cảnh báo từ multiprocessing resource_tracker (đến từ thư viện bên thứ ba như transformers)
# Cần đặt trước mọi import khác
warnings.filterwarnings("ignore", message=".*resource_tracker.*")

from flask import Flask, request
from flask_cors import CORS

from .config import Config
from .utils.logger import setup_logger, get_logger


def create_app(config_class=Config):
    """Hàm factory để khởi tạo Flask app"""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Thiết lập JSON encoding: đảm bảo tiếng Trung hiển thị trực tiếp (không ở dạng \uXXXX)
    # Flask >= 2.3 dùng app.json.ensure_ascii, phiên bản cũ dùng JSON_AS_ASCII
    if hasattr(app, 'json') and hasattr(app.json, 'ensure_ascii'):
        app.json.ensure_ascii = False
    
    # Thiết lập logger
    logger = setup_logger('mirofish')
    
    # Chỉ in log khởi động ở tiến trình con của reloader (tránh in 2 lần khi debug)
    is_reloader_process = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
    debug_mode = app.config.get('DEBUG', False)
    should_log_startup = not debug_mode or is_reloader_process
    
    if should_log_startup:
        logger.info("=" * 50)
        logger.info("MiroFish Backend is starting...")
        logger.info("=" * 50)
    
    # Bật CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Đăng ký hàm dọn tiến trình mô phỏng (đảm bảo dừng toàn bộ khi server tắt)
    from .services.simulation_runner import SimulationRunner
    SimulationRunner.register_cleanup()
    if should_log_startup:
        logger.info("Simulation process cleanup hook registered")
    
    # Middleware log request
    @app.before_request
    def log_request():
        logger = get_logger('mirofish.request')
        logger.debug(f"Request: {request.method} {request.path}")
        if request.content_type and 'json' in request.content_type:
            logger.debug(f"Request body: {request.get_json(silent=True)}")
    
    @app.after_request
    def log_response(response):
        logger = get_logger('mirofish.request')
        logger.debug(f"Response: {response.status_code}")
        return response
    
    # Đăng ký blueprint
    from .api import graph_bp, simulation_bp, report_bp
    app.register_blueprint(graph_bp, url_prefix='/api/graph')
    app.register_blueprint(simulation_bp, url_prefix='/api/simulation')
    app.register_blueprint(report_bp, url_prefix='/api/report')
    
    # Health check
    @app.route('/health')
    def health():
        return {'status': 'ok', 'service': 'MiroFish Backend'}
    
    if should_log_startup:
        logger.info("MiroFish Backend started successfully")
    
    return app

