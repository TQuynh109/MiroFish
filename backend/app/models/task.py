"""
Quản lý trạng thái Task (tác vụ)
Được sử dụng để theo dõi các tác vụ chạy ngầm mất nhiều thời gian (ví dụ: xây dựng Knowledge Graph)
"""

import uuid
import threading
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


class TaskStatus(str, Enum):
    """Định nghĩa các trạng thái (Enum) mà một Task có thể có"""
    PENDING = "pending"          # Đang chờ (Task mới được tạo, chưa được xử lý)
    PROCESSING = "processing"    # Đang xử lý (Hệ thống đang chạy ngầm Task này)
    COMPLETED = "completed"      # Đã hoàn thành thành công
    FAILED = "failed"            # Xảy ra lỗi và thất bại


@dataclass
class Task:
    """Lớp dữ liệu lưu trữ thông tin của một Task cụ thể"""
    task_id: str
    task_type: str
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    progress: int = 0              # Phần trăm tiến độ quá trình chạy (0-100)
    message: str = ""              # Thông báo trạng thái hiện tại để hiển thị cho người dùng
    result: Optional[Dict] = None  # Kết quả trả về sau khi Task chạy xong
    error: Optional[str] = None    # Thông tin chi tiết mỗi khi Task bị lỗi
    metadata: Dict = field(default_factory=dict)  # Siêu dữ liệu bổ sung kèm theo (ví dụ: project_id)
    progress_detail: Dict = field(default_factory=dict)  # Nội dung thông tin chi tiết về các bước trong tiến trình
    
    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi Class thành Dictionary để map vào JSON API Response"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "progress": self.progress,
            "message": self.message,
            "progress_detail": self.progress_detail,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
        }


class TaskManager:
    """
    Trình quản lý Task
    Đảm bảo quản lý trạng thái của các tác vụ được đồng bộ tốt trên nhiều luồng chạy (Thread-safe)
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Kế thừa Singleton Pattern (Chỉ khởi tạo 1 instance duy nhất trên toàn ứng dụng)"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._tasks: Dict[str, Task] = {}
                    cls._instance._task_lock = threading.Lock()
        return cls._instance
    
    def create_task(self, task_type: str, metadata: Optional[Dict] = None) -> str:
        """
        Tạo mới một Task và cho vào hàng đợi (quản lý state)
        
        Args:
            task_type: Loại tác vụ (vd: 'build_graph', 'generate_report', ...)
            metadata: Dữ liệu đính kèm (vd: project_id liên quan để cập nhật dữ liệu sau này)
            
        Returns:
            Chuỗi Mã định danh ngẫu nhiên (UUID) của Task
        """
        task_id = str(uuid.uuid4())
        now = datetime.now()
        
        task = Task(
            task_id=task_id,
            task_type=task_type,
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
            metadata=metadata or {}
        )
        
        with self._task_lock:
            self._tasks[task_id] = task
        
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Lấy thông tin một Task đang chạy/kết thúc dựa theo Task UUID"""
        with self._task_lock:
            return self._tasks.get(task_id)
    
    def update_task(
        self,
        task_id: str,
        status: Optional[TaskStatus] = None,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        result: Optional[Dict] = None,
        error: Optional[str] = None,
        progress_detail: Optional[Dict] = None
    ):
        """
        Cập nhật tiến trình của Task
        
        Args:
            task_id: Mã ID của Task đang chạy
            status: Trạng thái cập nhật (Pending, Processing, Completed, Failed)
            progress: % Tiến độ hiện tại
            message: Tin nhắn mô tả ngắn gọn hiện trạng làm gì
            result: Trả về kết quả đầu ra khi thành công
            error: Lời nhắn/Exception khi thất bại
            progress_detail: Các sub-tiến trình chi tiết
        """
        with self._task_lock:
            task = self._tasks.get(task_id)
            if task:
                task.updated_at = datetime.now()
                if status is not None:
                    task.status = status
                if progress is not None:
                    task.progress = progress
                if message is not None:
                    task.message = message
                if result is not None:
                    task.result = result
                if error is not None:
                    task.error = error
                if progress_detail is not None:
                    task.progress_detail = progress_detail
    
    def complete_task(self, task_id: str, result: Dict):
        """Hành động đánh dấu tác vụ đã kết thúc Thành Công và gán 100% cho progress"""
        self.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            progress=100,
            message="The task has been completed!",
            result=result
        )
    
    def fail_task(self, task_id: str, error: str):
        """Hành động đánh dấu tác vụ đã Thất Bại do lỗi"""
        self.update_task(
            task_id,
            status=TaskStatus.FAILED,
            message="The task has an error!?",
            error=error
        )
    
    def list_tasks(self, task_type: Optional[str] = None) -> list:
        """Liệt kê danh sách tất cả các Task (Hoặc filter theo type của task)"""
        with self._task_lock:
            tasks = list(self._tasks.values())
            if task_type:
                tasks = [t for t in tasks if t.task_type == task_type]
            return [t.to_dict() for t in sorted(tasks, key=lambda x: x.created_at, reverse=True)]
    
    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Dọn dẹp/xoá khỏi bộ nhớ các Task đã cũ (Đã hoàn thành hoặc lỗi sau N giờ) để tránh rò rỉ hoặc xài tốn RAM"""
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        
        with self._task_lock:
            old_ids = [
                tid for tid, task in self._tasks.items()
                if task.created_at < cutoff and task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
            ]
            for tid in old_ids:
                del self._tasks[tid]

