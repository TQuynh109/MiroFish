"""
OASIS模拟运行器
在后台运行模拟并记录每个Agent的动作，支持实时状态监控
"""

import os
import sys
import json
import time
import asyncio
import threading
import subprocess
import signal
import atexit
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import Queue

from ..config import Config
from ..utils.logger import get_logger
from .zep_graph_memory_updater import ZepGraphMemoryManager
from .simulation_ipc import SimulationIPCClient, CommandType, IPCResponse

logger = get_logger('mirofish.simulation_runner')

# Cờ đánh dấu đã đăng ký hàm dọn dẹp hay chưa
_cleanup_registered = False

# Kiểm tra hệ điều hành
IS_WINDOWS = sys.platform == 'win32'


class RunnerStatus(str, Enum):
    """Trạng thái của bộ chạy tiến trình mô phỏng"""
    IDLE = "idle"         # Rảnh rỗi, chưa chạy
    STARTING = "starting" # Đang khởi động
    RUNNING = "running"   # Đang chạy
    PAUSED = "paused"     # Đã tạm dừng
    STOPPING = "stopping" # Đang dừng lại
    STOPPED = "stopped"   # Đã dừng
    COMPLETED = "completed" # Đã hoàn thành
    FAILED = "failed"     # Bị lỗi


@dataclass
class AgentAction:
    """Bản ghi hành động của Agent"""
    round_num: int        # Số thứ tự của vòng (round) mô phỏng
    timestamp: str        # Dấu thời gian
    platform: str         # Nền tảng thực hiện: twitter / reddit
    agent_id: int         # ID của agent
    agent_name: str       # Tên của agent
    action_type: str      # Loại hành động: CREATE_POST, LIKE_POST, v.v.
    action_args: Dict[str, Any] = field(default_factory=dict) # Tham số của hành động
    result: Optional[str] = None # Kết quả thực thi
    success: bool = True  # Hành động có thành công hay không
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_num": self.round_num,
            "timestamp": self.timestamp,
            "platform": self.platform,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "action_type": self.action_type,
            "action_args": self.action_args,
            "result": self.result,
            "success": self.success,
        }


@dataclass
class RoundSummary:
    """Tóm tắt thông tin của mỗi vòng (round)"""
    round_num: int        # Số thứ tự vòng
    start_time: str       # Thời gian bắt đầu
    end_time: Optional[str] = None # Thời gian kết thúc
    simulated_hour: int = 0 # Số giờ đã mô phỏng trong vòng này
    twitter_actions: int = 0 # Số hành động trên Twitter
    reddit_actions: int = 0 # Số hành động trên Reddit
    active_agents: List[int] = field(default_factory=list) # Danh sách ID các agent đang hoạt động
    actions: List[AgentAction] = field(default_factory=list) # Danh sách các hành động
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_num": self.round_num,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "simulated_hour": self.simulated_hour,
            "twitter_actions": self.twitter_actions,
            "reddit_actions": self.reddit_actions,
            "active_agents": self.active_agents,
            "actions_count": len(self.actions),
            "actions": [a.to_dict() for a in self.actions],
        }


@dataclass
class SimulationRunState:
    """Trạng thái đang thực thi của tiến trình mô phỏng (cập nhật theo thời gian thực)"""
    simulation_id: str
    runner_status: RunnerStatus = RunnerStatus.IDLE
    
    # Thông tin tiến độ
    current_round: int = 0
    total_rounds: int = 0
    simulated_hours: int = 0
    total_simulation_hours: int = 0
    
    # Các vòng lặp và thời gian độc lập cho từng nền tảng (sử dụng để hiển thị song song hai nền tảng)
    twitter_current_round: int = 0
    reddit_current_round: int = 0
    twitter_simulated_hours: int = 0
    reddit_simulated_hours: int = 0
    
    # Trạng thái nền tảng đang chạy
    twitter_running: bool = False
    reddit_running: bool = False
    twitter_actions_count: int = 0
    reddit_actions_count: int = 0
    
    # Trạng thái hoàn thành chung của nền tảng (phát hiện qua sự kiện simulation_end trong actions.jsonl)
    twitter_completed: bool = False
    reddit_completed: bool = False
    
    # Tóm tắt lại ở mỗi vòng
    rounds: List[RoundSummary] = field(default_factory=list)
    
    # Các hành động gần nhất (để hiển thị theo thời gian thực (real-time) trên frontend)
    recent_actions: List[AgentAction] = field(default_factory=list)
    max_recent_actions: int = 50
    
    # Dấu thời gian
    started_at: Optional[str] = None
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    
    # Thông tin lỗi
    error: Optional[str] = None
    
    # ID tiến trình (PID) (để dừng/hủy tiến trình)
    process_pid: Optional[int] = None
    
    def add_action(self, action: AgentAction):
        """Thêm một hành động vào danh sách các hành động gần nhất"""
        self.recent_actions.insert(0, action)
        if len(self.recent_actions) > self.max_recent_actions:
            self.recent_actions = self.recent_actions[:self.max_recent_actions]
        
        if action.platform == "twitter":
            self.twitter_actions_count += 1
        else:
            self.reddit_actions_count += 1
        
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "simulation_id": self.simulation_id,
            "runner_status": self.runner_status.value,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "simulated_hours": self.simulated_hours,
            "total_simulation_hours": self.total_simulation_hours,
            "progress_percent": round(self.current_round / max(self.total_rounds, 1) * 100, 1),
            # Vòng lặp và thời gian độc lập cho mỗi nền tảng
            "twitter_current_round": self.twitter_current_round,
            "reddit_current_round": self.reddit_current_round,
            "twitter_simulated_hours": self.twitter_simulated_hours,
            "reddit_simulated_hours": self.reddit_simulated_hours,
            "twitter_running": self.twitter_running,
            "reddit_running": self.reddit_running,
            "twitter_completed": self.twitter_completed,
            "reddit_completed": self.reddit_completed,
            "twitter_actions_count": self.twitter_actions_count,
            "reddit_actions_count": self.reddit_actions_count,
            "total_actions_count": self.twitter_actions_count + self.reddit_actions_count,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "process_pid": self.process_pid,
        }
    
    def to_detail_dict(self) -> Dict[str, Any]:
        """Chi tiết thông tin bao gồm các hành động gần nhất"""
        result = self.to_dict()
        result["recent_actions"] = [a.to_dict() for a in self.recent_actions]
        result["rounds_count"] = len(self.rounds)
        return result


class SimulationRunner:
    """
    Trình chạy mô phỏng
    
    Quy trách nhiệm:
    1. Chạy mô phỏng OASIS trong tiến trình nền (background process)
    2. Phân tích nhật ký chạy (log), ghi lại hành động của mỗi Agent
    3. Cung cấp API truy vấn trạng thái thời gian thực
    4. Hỗ trợ thao tác tạm dừng (pause)/dừng (stop)/tiếp tục (resume)
    """
    
    # Thư mục lưu trữ trạng thái chạy
    RUN_STATE_DIR = os.path.join(
        os.path.dirname(__file__),
        '../../uploads/simulations'
    )
    
    # Thư mục chứa các script con (script chạy ứng dụng)
    SCRIPTS_DIR = os.path.join(
        os.path.dirname(__file__),
        '../../scripts'
    )
    
    # Trạng thái chạy trong bộ nhớ Memory (RAM)
    _run_states: Dict[str, SimulationRunState] = {}
    _processes: Dict[str, subprocess.Popen] = {}
    _action_queues: Dict[str, Queue] = {}
    _monitor_threads: Dict[str, threading.Thread] = {}
    _stdout_files: Dict[str, Any] = {}  # Lưu trữ tay cầm file đầu ra chuẩn (stdout)
    _stderr_files: Dict[str, Any] = {}  # Lưu trữ tay cầm file lỗi chuẩn (stderr)
    
    # Cấu hình cập nhật bộ nhớ Đồ thị (Graph Memory)
    _graph_memory_enabled: Dict[str, bool] = {}  # simulation_id -> enabled (Bật/tắt)
    
    @classmethod
    def get_run_state(cls, simulation_id: str) -> Optional[SimulationRunState]:
        """Lấy trạng thái chạy hiện tại"""
        if simulation_id in cls._run_states:
            return cls._run_states[simulation_id]
        
        # Thử tải từ file nếu không có trong memory
        state = cls._load_run_state(simulation_id)
        if state:
            cls._run_states[simulation_id] = state
        return state
    
    @classmethod
    def _load_run_state(cls, simulation_id: str) -> Optional[SimulationRunState]:
        """Tải trạng thái chạy từ tệp tin (run_state.json)"""
        state_file = os.path.join(cls.RUN_STATE_DIR, simulation_id, "run_state.json")
        if not os.path.exists(state_file):
            return None
        
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            state = SimulationRunState(
                simulation_id=simulation_id,
                runner_status=RunnerStatus(data.get("runner_status", "idle")),
                current_round=data.get("current_round", 0),
                total_rounds=data.get("total_rounds", 0),
                simulated_hours=data.get("simulated_hours", 0),
                total_simulation_hours=data.get("total_simulation_hours", 0),
                # Các vòng lặp và thời gian độc lập cho mỗi nền tảng
                twitter_current_round=data.get("twitter_current_round", 0),
                reddit_current_round=data.get("reddit_current_round", 0),
                twitter_simulated_hours=data.get("twitter_simulated_hours", 0),
                reddit_simulated_hours=data.get("reddit_simulated_hours", 0),
                twitter_running=data.get("twitter_running", False),
                reddit_running=data.get("reddit_running", False),
                twitter_completed=data.get("twitter_completed", False),
                reddit_completed=data.get("reddit_completed", False),
                twitter_actions_count=data.get("twitter_actions_count", 0),
                reddit_actions_count=data.get("reddit_actions_count", 0),
                started_at=data.get("started_at"),
                updated_at=data.get("updated_at", datetime.now().isoformat()),
                completed_at=data.get("completed_at"),
                error=data.get("error"),
                process_pid=data.get("process_pid"),
            )
            
            # Tải danh sách các hành động gần đây
            actions_data = data.get("recent_actions", [])
            for a in actions_data:
                state.recent_actions.append(AgentAction(
                    round_num=a.get("round_num", 0),
                    timestamp=a.get("timestamp", ""),
                    platform=a.get("platform", ""),
                    agent_id=a.get("agent_id", 0),
                    agent_name=a.get("agent_name", ""),
                    action_type=a.get("action_type", ""),
                    action_args=a.get("action_args", {}),
                    result=a.get("result"),
                    success=a.get("success", True),
                ))
            
            return state
        except Exception as e:
            logger.error(f"Failed to load run state: {str(e)}")
            return None
    
    @classmethod
    def _save_run_state(cls, state: SimulationRunState):
        """Lưu trạng thái chạy vào file"""
        sim_dir = os.path.join(cls.RUN_STATE_DIR, state.simulation_id)
        os.makedirs(sim_dir, exist_ok=True)
        state_file = os.path.join(sim_dir, "run_state.json")
        
        data = state.to_detail_dict()
        
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        cls._run_states[state.simulation_id] = state
    
    @classmethod
    def start_simulation(
        cls,
        simulation_id: str,
        platform: str = "parallel",  # twitter / reddit / parallel
        max_rounds: int = None,  # Số vòng mô phỏng tối đa (tùy chọn, dùng để cắt ngắn các mô phỏng quá dài)
        enable_graph_memory_update: bool = False,  # Có liên tục cập nhật hoạt động của Agent vào Zep graph hay không
        graph_id: str = None  # ID của Zep graph (Bắt buộc nếu bật tính năng cập nhật sơ đồ (graph))
    ) -> SimulationRunState:
        """
        Bắt đầu mô phỏng
        
        Args:
            simulation_id: ID mô phỏng
            platform: Nền tảng chạy (twitter/reddit/parallel)
            max_rounds: Số vòng chạy tối đa (để cắt bớt)
            enable_graph_memory_update: Có cập nhật hành vi Agent vào Zep Graph hay không
            graph_id: Zep Graph ID
            
        Returns:
            SimulationRunState (Trạng thái sau khi cấu hình)
        """
        # Kiểm tra xem có tiến trình nào đang chạy không
        existing = cls.get_run_state(simulation_id)
        if existing and existing.runner_status in [RunnerStatus.RUNNING, RunnerStatus.STARTING]:
            raise ValueError(f"Simulation is already running: {simulation_id}")
        
        # Tải cấu hình mô phỏng
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        config_path = os.path.join(sim_dir, "simulation_config.json")
        
        if not os.path.exists(config_path):
            raise ValueError(f"Simulation configuration not found, please call the /prepare API first")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Khởi tạo trạng thái chạy
        time_config = config.get("time_config", {})
        total_hours = time_config.get("total_simulation_hours", 72)
        minutes_per_round = time_config.get("minutes_per_round", 30)
        total_rounds = int(total_hours * 60 / minutes_per_round)
        
        # Nếu chỉ định maximum rounds, tiến hành việc cắt bớt
        if max_rounds is not None and max_rounds > 0:
            original_rounds = total_rounds
            total_rounds = min(total_rounds, max_rounds)
            if total_rounds < original_rounds:
                logger.info(f"Rounds truncated: {original_rounds} -> {total_rounds} (max_rounds={max_rounds})")
        
        state = SimulationRunState(
            simulation_id=simulation_id,
            runner_status=RunnerStatus.STARTING,
            total_rounds=total_rounds,
            total_simulation_hours=total_hours,
            started_at=datetime.now().isoformat(),
        )
        
        cls._save_run_state(state)
        
        # Nếu tính năng cập nhật bộ nhớ graph được bật, tạo một updater
        if enable_graph_memory_update:
            if not graph_id:
                raise ValueError("graph_id is required to enable graph memory updates")
            
            try:
                ZepGraphMemoryManager.create_updater(simulation_id, graph_id)
                cls._graph_memory_enabled[simulation_id] = True
                logger.info(f"Graph memory update enabled: simulation_id={simulation_id}, graph_id={graph_id}")
            except Exception as e:
                logger.error(f"Failed to create graph memory updater: {e}")
                cls._graph_memory_enabled[simulation_id] = False
        else:
            cls._graph_memory_enabled[simulation_id] = False
        
        # Xác định script nào sẽ chạy (các script nằm trong thư mục backend/scripts/)
        if platform == "twitter":
            script_name = "run_twitter_simulation.py"
            state.twitter_running = True
        elif platform == "reddit":
            script_name = "run_reddit_simulation.py"
            state.reddit_running = True
        else:
            script_name = "run_parallel_simulation.py"
            state.twitter_running = True
            state.reddit_running = True
        
        script_path = os.path.join(cls.SCRIPTS_DIR, script_name)
        
        if not os.path.exists(script_path):
            raise ValueError(f"Script no longer exists: {script_path}")
        
        # Tạo hàng đợi các hành động (Queue)
        action_queue = Queue()
        cls._action_queues[simulation_id] = action_queue
        
        # Bắt đầu chạy tiến trình mô phỏng
        try:
            # Xây dựng lệnh chạy, sử dụng full path
            # Cấu trúc log mới:
            #   twitter/actions.jsonl - Log cho các hành động trên Twitter
            #   reddit/actions.jsonl  - Log cho các hành động trên Reddit
            #   simulation.log        - Log cho tiến trình chính
            
            cmd = [
                sys.executable,  # Python Interpreter
                script_path,
                "--config", config_path,  # Use full path to config
            ]
            
            # Nếu có thiết lập giới hạn vòng tối đa, hãy truyền nó qua dòng lệnh (command line args)
            if max_rounds is not None and max_rounds > 0:
                cmd.extend(["--max-rounds", str(max_rounds)])
            
            # Tạo tệp log chính để tránh bộ đệm ống dẫn (pipe buffer) stdout/stderr của tiến trình đầy
            main_log_path = os.path.join(sim_dir, "simulation.log")
            main_log_file = open(main_log_path, 'w', encoding='utf-8')
            
            # Đặt môi trường cho quy trình con để đảm bảo trên Windows được mã hóa thành UTF-8
            # Điều này sửa lỗi thư viện của bên thứ 3 khi họ gọi file hệ thống nếu không chỉ định rõ encode.
            env = os.environ.copy()
            env['PYTHONUTF8'] = '1'  # Python 3.7+ hỗ trợ điều này, giúp mọi hàm open() mặc định theo UTF-8
            env['PYTHONIOENCODING'] = 'utf-8'  # Đảm bảo đầu ra có stdout/stderr dưới dạng UTF-8
            
            # Đặt thư mục làm việc (CWD - Current Working Directory) thành thư mục nơi mô phỏng
            # Thiết lập start_new_session=True sẽ tạo ra nhóm các tiến trình con mới, vì thế thông qua os.killpg có thể hủy toàn bộ những cái đó
            process = subprocess.Popen(
                cmd,
                cwd=sim_dir,
                stdout=main_log_file,
                stderr=subprocess.STDOUT,  # Đẩy luồng stderr cũng vào file đó
                text=True,
                encoding='utf-8',  # Explicitly specify encoding
                bufsize=1,
                env=env,  # Đi kèm bộ setting Environment có set UTF-8
                start_new_session=True,  # Bắt đầu tạo 1 luồng xử lý mới (New process group)
            )
            
            # Ghi lại file để cho bước đóng (close) được thực hiện dễ dàng
            cls._stdout_files[simulation_id] = main_log_file
            cls._stderr_files[simulation_id] = None  # Không cần lưu file stderr độc lập nữa
            
            state.process_pid = process.pid
            state.runner_status = RunnerStatus.RUNNING
            cls._processes[simulation_id] = process
            cls._save_run_state(state)
            
            # Khởi động tiểu trình giám sát (Monitor thread)
            monitor_thread = threading.Thread(
                target=cls._monitor_simulation,
                args=(simulation_id,),
                daemon=True
            )
            monitor_thread.start()
            cls._monitor_threads[simulation_id] = monitor_thread
            
            logger.info(f"Simulation started successfully: {simulation_id}, pid={process.pid}, platform={platform}")
            
        except Exception as e:
            state.runner_status = RunnerStatus.FAILED
            state.error = str(e)
            cls._save_run_state(state)
            raise
        
        return state
    
    @classmethod
    def _monitor_simulation(cls, simulation_id: str):
        """Giám sát (Monitor) phân tích nhật ký ghi lại các hành động"""
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        
        # 新的日志结构：分平台的动作日志
        twitter_actions_log = os.path.join(sim_dir, "twitter", "actions.jsonl")
        reddit_actions_log = os.path.join(sim_dir, "reddit", "actions.jsonl")
        
        process = cls._processes.get(simulation_id)
        state = cls.get_run_state(simulation_id)
        
        if not process or not state:
            return
        
        twitter_position = 0
        reddit_position = 0
        
        try:
            while process.poll() is None:  # 进程仍在运行
                # 读取 Twitter 动作日志
                if os.path.exists(twitter_actions_log):
                    twitter_position = cls._read_action_log(
                        twitter_actions_log, twitter_position, state, "twitter"
                    )
                
                # 读取 Reddit 动作日志
                if os.path.exists(reddit_actions_log):
                    reddit_position = cls._read_action_log(
                        reddit_actions_log, reddit_position, state, "reddit"
                    )
                
                # 更新状态
                cls._save_run_state(state)
                time.sleep(2)
            
            # 进程结束后，最后读取一次日志
            if os.path.exists(twitter_actions_log):
                cls._read_action_log(twitter_actions_log, twitter_position, state, "twitter")
            if os.path.exists(reddit_actions_log):
                cls._read_action_log(reddit_actions_log, reddit_position, state, "reddit")
            
            # 进程结束
            exit_code = process.returncode
            
            if exit_code == 0:
                state.runner_status = RunnerStatus.COMPLETED
                state.completed_at = datetime.now().isoformat()
                logger.info(f"模拟完成: {simulation_id}")
            else:
                state.runner_status = RunnerStatus.FAILED
                # 从主日志文件读取错误信息
                main_log_path = os.path.join(sim_dir, "simulation.log")
                error_info = ""
                try:
                    if os.path.exists(main_log_path):
                        with open(main_log_path, 'r', encoding='utf-8') as f:
                            error_info = f.read()[-2000:]  # 取最后2000字符
                except Exception:
                    pass
                state.error = f"进程退出码: {exit_code}, 错误: {error_info}"
                logger.error(f"模拟失败: {simulation_id}, error={state.error}")
            
            state.twitter_running = False
            state.reddit_running = False
            cls._save_run_state(state)
            
        except Exception as e:
            logger.error(f"监控线程异常: {simulation_id}, error={str(e)}")
            state.runner_status = RunnerStatus.FAILED
            state.error = str(e)
            cls._save_run_state(state)
        
        finally:
            # 停止图谱记忆更新器
            if cls._graph_memory_enabled.get(simulation_id, False):
                try:
                    ZepGraphMemoryManager.stop_updater(simulation_id)
                    logger.info(f"已停止图谱记忆更新: simulation_id={simulation_id}")
                except Exception as e:
                    logger.error(f"停止图谱记忆更新器失败: {e}")
                cls._graph_memory_enabled.pop(simulation_id, None)
            
            # 清理进程资源
            cls._processes.pop(simulation_id, None)
            cls._action_queues.pop(simulation_id, None)
            
            # 关闭日志文件句柄
            if simulation_id in cls._stdout_files:
                try:
                    cls._stdout_files[simulation_id].close()
                except Exception:
                    pass
                cls._stdout_files.pop(simulation_id, None)
            if simulation_id in cls._stderr_files and cls._stderr_files[simulation_id]:
                try:
                    cls._stderr_files[simulation_id].close()
                except Exception:
                    pass
                cls._stderr_files.pop(simulation_id, None)
    
    @classmethod
    def _read_action_log(
        cls, 
        log_path: str, 
        position: int, 
        state: SimulationRunState,
        platform: str
    ) -> int:
        """
        Đọc tệp tin nhật ký (log) của hệ thống
        
        Args:
            log_path: Đường dẫn tệp nhật ký
            position: Vị trí đọc trước đó
            state: Đối tượng trạng thái đang chạy
            platform: Nền tảng (twitter/reddit)
            
        Returns:
            Vị trí đọc mới
        """
        # Kiểm tra xem có bật tính năng cập nhật bộ nhớ graph hay không
        graph_memory_enabled = cls._graph_memory_enabled.get(state.simulation_id, False)
        graph_updater = None
        if graph_memory_enabled:
            graph_updater = ZepGraphMemoryManager.get_updater(state.simulation_id)
        
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                f.seek(position)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            action_data = json.loads(line)
                            
                            # Xử lý các mục của loại sự kiện
                            if "event_type" in action_data:
                                event_type = action_data.get("event_type")
                                
                                # Phát hiện sự kiện simulation_end và đánh dấu nền tảng đã hoàn thành
                                if event_type == "simulation_end":
                                    if platform == "twitter":
                                        state.twitter_completed = True
                                        state.twitter_running = False
                                        logger.info(f"Twitter simulation completed: {state.simulation_id}, total_rounds={action_data.get('total_rounds')}, total_actions={action_data.get('total_actions')}")
                                    elif platform == "reddit":
                                        state.reddit_completed = True
                                        state.reddit_running = False
                                        logger.info(f"Reddit simulation completed: {state.simulation_id}, total_rounds={action_data.get('total_rounds')}, total_actions={action_data.get('total_actions')}")
                                    
                                    # Kiểm tra xem có phải tất cả các nền tảng được bật đều đã hoàn thành hay không
                                    # Nếu chỉ một nền tảng đang chạy, hãy chỉ kiểm tra nền tảng đó
                                    # Nếu 2 nền tảng đang chạy thì yêu cầu phải hoàn thành cả 2 nền tảng
                                    all_completed = cls._check_all_platforms_completed(state)
                                    if all_completed:
                                        state.runner_status = RunnerStatus.COMPLETED
                                        state.completed_at = datetime.now().isoformat()
                                        logger.info(f"Simulation completed for all platforms: {state.simulation_id}")
                                
                                # Cập nhật thông tin vòng (round_num) (từ sự kiện round_end)
                                elif event_type == "round_end":
                                    round_num = action_data.get("round", 0)
                                    simulated_hours = action_data.get("simulated_hours", 0)
                                    
                                    # Cập nhật thời gian và vòng thứ tự độc lập cho nền tảng
                                    if platform == "twitter":
                                        if round_num > state.twitter_current_round:
                                            state.twitter_current_round = round_num
                                        state.twitter_simulated_hours = simulated_hours
                                    elif platform == "reddit":
                                        if round_num > state.reddit_current_round:
                                            state.reddit_current_round = round_num
                                        state.reddit_simulated_hours = simulated_hours
                                    
                                    # Số vòng chung sẽ là số lớn nhất của hai nền tảng
                                    if round_num > state.current_round:
                                        state.current_round = round_num
                                    # Thời gian chung sẽ là số lớn nhất của hai nền tảng
                                    state.simulated_hours = max(state.twitter_simulated_hours, state.reddit_simulated_hours)
                                
                                continue
                            
                            action = AgentAction(
                                round_num=action_data.get("round", 0),
                                timestamp=action_data.get("timestamp", datetime.now().isoformat()),
                                platform=platform,
                                agent_id=action_data.get("agent_id", 0),
                                agent_name=action_data.get("agent_name", ""),
                                action_type=action_data.get("action_type", ""),
                                action_args=action_data.get("action_args", {}),
                                result=action_data.get("result"),
                                success=action_data.get("success", True),
                            )
                            state.add_action(action)
                            
                            # Cập nhật thông tin (vòng) round
                            if action.round_num and action.round_num > state.current_round:
                                state.current_round = action.round_num
                            
                            # Nếu cập nhật bộ nhớ graph được bật, thêm hoạt động vào Zep graph
                            if graph_updater:
                                graph_updater.add_activity_from_dict(action_data, platform)
                            
                        except json.JSONDecodeError:
                            pass
                return f.tell()
        except Exception as e:
            logger.warning(f"Failed to read action logs: {log_path}, error={e}")
            return position
    
    @classmethod
    def _check_all_platforms_completed(cls, state: SimulationRunState) -> bool:
        """
        Kiểm tra xem tất cả các nền tảng có hoàn thành quá trình mô phỏng hay chưa?
        
        Kiểm tra xem nền tảng có được kích hoạt (hay enable) hay không bằng cách xem tệp tin actions.jsonl có tồn tại hay không
        
        Returns:
            True Nếu tất cả các nền tảng được bật đều đã hoàn thành
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, state.simulation_id)
        twitter_log = os.path.join(sim_dir, "twitter", "actions.jsonl")
        reddit_log = os.path.join(sim_dir, "reddit", "actions.jsonl")
        
        # Kiểm tra xem mình có đang bật nền tảng nào không (Sử dụng cách kiểm tra tệp tin có tồn tại (exist) hay không)
        twitter_enabled = os.path.exists(twitter_log)
        reddit_enabled = os.path.exists(reddit_log)
        
        # Nền tảng nào chưa xong thì trả về false
        if twitter_enabled and not state.twitter_completed:
            return False
        if reddit_enabled and not state.reddit_completed:
            return False
        
        # Phải có ít nhất 1 nền tảng chạy xong thì mới là true. (nếu 1 nền tảng không chạy => False. False and False = False. True and False = True)
        return twitter_enabled or reddit_enabled
    
    @classmethod
    def _terminate_process(cls, process: subprocess.Popen, simulation_id: str, timeout: int = 10):
        """
        Khả năng tương thích nền tảng, dừng một quá trình và các quá trình con (nhánh)
        
        Args:
            process: Quy trình để chấm dứt (kill)
            simulation_id: Ghi log ID
            timeout: Thời gian chờ cho phép tiến trình kết thúc tính bằng giây (seconds)
        """
        if IS_WINDOWS:
            # Windows: Sử dụng câu lệnh taskkill để xóa cả tiến trình theo cấu trúc branch tree
            # /F = Force termination (Xoa bằng mọi giá), /T = Terminate tree (xóa nhánh tiến trình) bao gồm các sub-process
            logger.info(f"Đang dừng quá trình (Windows): simulation={simulation_id}, pid={process.pid}")
            try:
                # Trước hết hãy cố găng dừng mềm
                subprocess.run(
                    ['taskkill', '/PID', str(process.pid), '/T'],
                    capture_output=True,
                    timeout=5
                )
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    # Nếu không thể dùng dừng mềm (sau timeout), xóa cứng bằng /F
                    logger.warning(f"Process unresponsive, force terminating: {simulation_id}")
                    subprocess.run(
                        ['taskkill', '/F', '/PID', str(process.pid), '/T'],
                        capture_output=True,
                        timeout=5
                    )
                    process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"taskkill failed, attempting terminate: {e}")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        else:
            # Unix: Sử dụng process group chấm dứt
            # Dùng start_new_session=True, giá trị pgid sẽ bằng đúng với PID gốc của tiến trình
            pgid = os.getpgid(process.pid)
            logger.info(f"Đang dừng nhóm tiến trình (Unix): simulation={simulation_id}, pgid={pgid}")
            
            # Gửi SIGTERM tới toàn bộ process group
            os.killpg(pgid, signal.SIGTERM)
            
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Nếu xảy ra hiện tượng chưa tự hủy sau timeout, xóa cưỡng bức bằng SIGKILL
                logger.warning(f"Process group unresponsive to SIGTERM, force terminating: {simulation_id}")
                os.killpg(pgid, signal.SIGKILL)
                process.wait(timeout=5)
    
    @classmethod
    def stop_simulation(cls, simulation_id: str) -> SimulationRunState:
        """Dừng lại tiến trình mô phỏng"""
        state = cls.get_run_state(simulation_id)
        if not state:
            raise ValueError(f"Simulation not found: {simulation_id}")
        
        if state.runner_status not in [RunnerStatus.RUNNING, RunnerStatus.PAUSED]:
            raise ValueError(f"Simulation is not running: {simulation_id}, status={state.runner_status}")
        
        state.runner_status = RunnerStatus.STOPPING
        cls._save_run_state(state)
        
        # Kết thúc tiến trình con (child process)
        process = cls._processes.get(simulation_id)
        if process and process.poll() is None:
            try:
                cls._terminate_process(process, simulation_id)
            except ProcessLookupError:
                # Quá trình không còn ở đây nữa (Đã thoát hoặc bị đóng)
                pass
            except Exception as e:
                logger.error(f"Failed to terminate process group: {simulation_id}, error={e}")
                # Thử thêm một cách nữa để chăc chắn hủy tiến trình
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except Exception:
                    process.kill()
        
        state.runner_status = RunnerStatus.STOPPED
        state.twitter_running = False
        state.reddit_running = False
        state.completed_at = datetime.now().isoformat()
        cls._save_run_state(state)
        
        # Dừng quá trình graph memory updater
        if cls._graph_memory_enabled.get(simulation_id, False):
            try:
                ZepGraphMemoryManager.stop_updater(simulation_id)
                logger.info(f"Graph memory update stopped: simulation_id={simulation_id}")
            except Exception as e:
                logger.error(f"Failed to stop graph memory updater: {e}")
            cls._graph_memory_enabled.pop(simulation_id, None)
        
        logger.info(f"Simulation stopped: {simulation_id}")
        return state
    
    @classmethod
    def _read_actions_from_file(
        cls,
        file_path: str,
        default_platform: Optional[str] = None,
        platform_filter: Optional[str] = None,
        agent_id: Optional[int] = None,
        round_num: Optional[int] = None
    ) -> List[AgentAction]:
        """
        Đọc các hoạt động từ một tệp duy nhất
        
        Args:
            file_path: Đường dẫn tệp log của hành động đó
            default_platform: Nền tảng mặc định (nếu trong nhật ký không có platform)
            platform_filter: Lọc nền tảng (chỉ định platform cần đọc log)
            agent_id: Lọc ID của Agent cụ thể
            round_num: Lọc số vòng của Agent
        """
        if not os.path.exists(file_path):
            return []
        
        actions = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Bỏ qua các bản ghi không phải hành động (chẳng hạn như là sự kiện về hệ thống: simulation_start, round_start, round_end v.v.)
                    if "event_type" in data:
                        continue
                    
                    # Bỏ lỡ các sự kiện không phải do agent tạo ra (Không có ID đặc trưng của Agent)
                    if "agent_id" not in data:
                        continue
                    
                    # Lấy nền tảng (Platform): Ưu tiên lấy từ bản ghi nếu có `platform`, nếu không thì dùng `default_platform`
                    record_platform = data.get("platform") or default_platform or ""
                    
                    # Bộ lọc (Filtering)
                    if platform_filter and record_platform != platform_filter:
                        continue
                    if agent_id is not None and data.get("agent_id") != agent_id:
                        continue
                    if round_num is not None and data.get("round") != round_num:
                        continue
                    
                    actions.append(AgentAction(
                        round_num=data.get("round", 0),
                        timestamp=data.get("timestamp", ""),
                        platform=record_platform,
                        agent_id=data.get("agent_id", 0),
                        agent_name=data.get("agent_name", ""),
                        action_type=data.get("action_type", ""),
                        action_args=data.get("action_args", {}),
                        result=data.get("result"),
                        success=data.get("success", True),
                    ))
                    
                except json.JSONDecodeError:
                    continue
        
        return actions
    
    @classmethod
    def get_all_actions(
        cls,
        simulation_id: str,
        platform: Optional[str] = None,
        agent_id: Optional[int] = None,
        round_num: Optional[int] = None
    ) -> List[AgentAction]:
        """
        Lấy thông tin tất cả lịch sử hoạt động của các nền tảng (không giới hạn phân trang)
        
        Args:
            simulation_id: ID mô phỏng
            platform: Bộ lọc nền tảng hoạt động (twitter/reddit)
            agent_id: Lọc Agent
            round_num: Lọc số vòng
            
        Returns:
            Danh sách đầy đủ các actions (sắp xếp theo thời gian mới nhất lên trước)
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        actions = []
        
        # Đọc tệp tin Actions của Twitter (Khai báo tự động điền twitter theo cấu trúc tệp tin log)
        twitter_actions_log = os.path.join(sim_dir, "twitter", "actions.jsonl")
        if not platform or platform == "twitter":
            actions.extend(cls._read_actions_from_file(
                twitter_actions_log,
                default_platform="twitter",  # Điền dữ liệu tự động cho record `platform`
                platform_filter=platform,
                agent_id=agent_id, 
                round_num=round_num
            ))
        
        # Đọc tệp tin Actions của Reddit (Tự động điền phần 'reddit' căn cứ thư mục chứa tệp tin)
        reddit_actions_log = os.path.join(sim_dir, "reddit", "actions.jsonl")
        if not platform or platform == "reddit":
            actions.extend(cls._read_actions_from_file(
                reddit_actions_log,
                default_platform="reddit",  # Automatically fill the platform field
                platform_filter=platform,
                agent_id=agent_id,
                round_num=round_num
            ))
        
        # Nếu thư mục chạy các nền tảng chạy parallel này (twitter / reddit) không có ở đó. Hãy thử với các tệp định dạng cũ
        if not actions:
            actions_log = os.path.join(sim_dir, "actions.jsonl")
            actions = cls._read_actions_from_file(
                actions_log,
                default_platform=None,  # Các file json log định dạng cũ đã có sẵn record về platform nên không điền default
                platform_filter=platform,
                agent_id=agent_id,
                round_num=round_num
            )
        
        # Sắp xếp lại log theo thời gian timestamp giảm dần (từ mới hơn lên trước)
        actions.sort(key=lambda x: x.timestamp, reverse=True)
        
        return actions
    
    @classmethod
    def get_actions(
        cls,
        simulation_id: str,
        limit: int = 100,
        offset: int = 0,
        platform: Optional[str] = None,
        agent_id: Optional[int] = None,
        round_num: Optional[int] = None
    ) -> List[AgentAction]:
        """
        Lấy thông tin lịch sử diễn ra (Hỗ trợ phân trang bằng offset và limit)
        
        Args:
            simulation_id: Simulation ID
            limit: Record Count returns limit
            offset: Offset
            platform: Filter Platform
            agent_id: Filter Agent by ID
            round_num: Filter round loop
            
        Returns:
            Actions list
        """
        actions = cls.get_all_actions(
            simulation_id=simulation_id,
            platform=platform,
            agent_id=agent_id,
            round_num=round_num
        )
        
        # Phân trang
        return actions[offset:offset + limit]
    
    @classmethod
    def get_timeline(
        cls,
        simulation_id: str,
        start_round: int = 0,
        end_round: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Lấy thời gian (timeline) mô phỏng diễn ra (tóm tắt theo vòng được khai báo)
        
        Args:
            simulation_id: ID Mô phỏng
            start_round:  Bắt đầu từ một vòng lặp nhất định (First round Number)
            end_round: Kết thúc từ vòng ở đó (End round Number)
            
        Returns:
            Cung cấp đầy đủ thông tin về các round bị gói gọn
        """
        actions = cls.get_actions(simulation_id, limit=10000)
        
        # Nhóm tiến trình lại vào trong Vòng (round grouping)
        rounds: Dict[int, Dict[str, Any]] = {}
        
        for action in actions:
            round_num = action.round_num
            
            if round_num < start_round:
                continue
            if end_round is not None and round_num > end_round:
                continue
            
            if round_num not in rounds:
                rounds[round_num] = {
                    "round_num": round_num,
                    "twitter_actions": 0,
                    "reddit_actions": 0,
                    "active_agents": set(),
                    "action_types": {},
                    "first_action_time": action.timestamp,
                    "last_action_time": action.timestamp,
                }
            
            r = rounds[round_num]
            
            if action.platform == "twitter":
                r["twitter_actions"] += 1
            else:
                r["reddit_actions"] += 1
            
            r["active_agents"].add(action.agent_id)
            r["action_types"][action.action_type] = r["action_types"].get(action.action_type, 0) + 1
            r["last_action_time"] = action.timestamp
        
        # Chuyển đổi trạng thái về Lists Arrays Data type
        result = []
        for round_num in sorted(rounds.keys()):
            r = rounds[round_num]
            result.append({
                "round_num": round_num,
                "twitter_actions": r["twitter_actions"],
                "reddit_actions": r["reddit_actions"],
                "total_actions": r["twitter_actions"] + r["reddit_actions"],
                "active_agents_count": len(r["active_agents"]),
                "active_agents": list(r["active_agents"]),
                "action_types": r["action_types"],
                "first_action_time": r["first_action_time"],
                "last_action_time": r["last_action_time"],
            })
        
        return result
    
    @classmethod
    def get_agent_stats(cls, simulation_id: str) -> List[Dict[str, Any]]:
        """
        Lấy thông kê của mọi agent
        
        Returns:
            Danh sách thống kê Agent
        """
        actions = cls.get_actions(simulation_id, limit=10000)
        
        agent_stats: Dict[int, Dict[str, Any]] = {}
        
        for action in actions:
            agent_id = action.agent_id
            
            if agent_id not in agent_stats:
                agent_stats[agent_id] = {
                    "agent_id": agent_id,
                    "agent_name": action.agent_name,
                    "total_actions": 0,
                    "twitter_actions": 0,
                    "reddit_actions": 0,
                    "action_types": {},
                    "first_action_time": action.timestamp,
                    "last_action_time": action.timestamp,
                }
            
            stats = agent_stats[agent_id]
            stats["total_actions"] += 1
            
            if action.platform == "twitter":
                stats["twitter_actions"] += 1
            else:
                stats["reddit_actions"] += 1
            
            stats["action_types"][action.action_type] = stats["action_types"].get(action.action_type, 0) + 1
            stats["last_action_time"] = action.timestamp
        
        # Sắp xếp theo tổng số hành động giảm dần (reverse = true)
        result = sorted(agent_stats.values(), key=lambda x: x["total_actions"], reverse=True)
        
        return result
    
    @classmethod
    def cleanup_simulation_logs(cls, simulation_id: str) -> Dict[str, Any]:
        """
        Xóa tệp log chạy để buộc mô phỏng được khởi động lại
        
        Xóa sạch các tệp tin này bao gồm:
        - run_state.json
        - twitter/actions.jsonl
        - reddit/actions.jsonl
        - simulation.log
        - stdout.log / stderr.log
        - twitter_simulation.db（Dữ liệu nền tảng twitter)
        - reddit_simulation.db（Dữ liệu nền tảng reddit）
        - env_status.json（Trạng thái file Environment status）
        
        Chú ý: Các file liên kết đến cấu hình mô phỏng hay config thiết lập (như là simulation_config.json) hay Profile đều sẽ KHÔNG bị xóa đi.
        
        Args:
            simulation_id: Simulation ID
            
        Returns:
            Kết quả của lệnh xóa sạch (clean up)
        """
        import shutil
        
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        
        if not os.path.exists(sim_dir):
            return {"success": True, "message": "Simulation directory does not exist, no need to clean."}
        
        cleaned_files = []
        errors = []
        
        # Các tệp tin cần bị loại bỏ bao gồm log, database...
        files_to_delete = [
            "run_state.json",
            "simulation.log",
            "stdout.log",
            "stderr.log",
            "twitter_simulation.db",  # Twitter Database
            "reddit_simulation.db",   # Reddit Database
            "env_status.json",        # Env state status file
        ]
        
        # Nhưng tệp tin có cấp quyền cần xóa (có liên quan nhật ký hoạt động actions.jsonl)
        dirs_to_clean = ["twitter", "reddit"]
        
        # Loại bỏ các tệp không cần tới (Delete them)
        for filename in files_to_delete:
            file_path = os.path.join(sim_dir, filename)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    cleaned_files.append(filename)
                except Exception as e:
                    errors.append(f"Failed to delete {filename}: {str(e)}")
        
        # Kiểm tra lại các file theo thư mục chứa action history action.jsonl files
        for dir_name in dirs_to_clean:
            dir_path = os.path.join(sim_dir, dir_name)
            if os.path.exists(dir_path):
                actions_file = os.path.join(dir_path, "actions.jsonl")
                if os.path.exists(actions_file):
                    try:
                        os.remove(actions_file)
                        cleaned_files.append(f"{dir_name}/actions.jsonl")
                    except Exception as e:
                        errors.append(f"Failed to delete {dir_name}/actions.jsonl: {str(e)}")
        
        # Xóa (clear) cache nhớ run state
        if simulation_id in cls._run_states:
            del cls._run_states[simulation_id]
        
        logger.info(f"Clean up complete for simulation: {simulation_id}, Deleted files: {cleaned_files}")
        
        return {
            "success": len(errors) == 0,
            "cleaned_files": cleaned_files,
            "errors": errors if errors else None
        }
    
    # Flags Ngăn việc phải làm quá nhiều việc cho một action cleanup đã làm từ đầu hay khi gọi lần tới lệnh giống
    _cleanup_done = False
    
    @classmethod
    def cleanup_all_simulations(cls):
        """
        Dọn dẹp tất cả các tiến trình mô phỏng đang chạy
        
        Được gọi (call) khi đóng máy chủ, nhằm vào việc muốn các máy chủ con (child processes) bị tắt theo
        """
        # Nếu đã clean (dọn dẹp) thì không làm nữa
        if cls._cleanup_done:
            return
        cls._cleanup_done = True
        
        # Kiểm tra xem có gì để dọn dẹp không (tránh log rỗng khi không có tiến trình chạy)
        has_processes = bool(cls._processes)
        has_updaters = bool(cls._graph_memory_enabled)
        
        if not has_processes and not has_updaters:
            return  # Không có gì để dọn, kết thúc
        
        logger.info("Cleaning up all simulation processes...")
        
        # Dừng tất cả cái update đồ thị nhớ (Bộ nhớ Graph) (Stop_all ghi nhận ở bên trong)
        try:
            ZepGraphMemoryManager.stop_all()
        except Exception as e:
            logger.error(f"Failed to stop Zep Graph update daemon: {e}")
        cls._graph_memory_enabled.clear()
        
        # Tạo bản sao Dictionary dict() từ cls._processes.items để không bị hỏng List lúc lặp (iterating)
        processes = list(cls._processes.items())
        
        for simulation_id, process in processes:
            try:
                if process.poll() is None:  # Process (Tiến trình) Vẫn đang chạy
                    logger.info(f"Terminating simulation process: {simulation_id}, pid={process.pid}")
                    
                    try:
                        # Áp dụng giải pháp dừng liên nền tảng (Cross-platform termination method)
                        cls._terminate_process(process, simulation_id, timeout=5)
                    except (ProcessLookupError, OSError):
                        # Trong trường hợp có thể các process này đã biến mất ở đâu đó rồi, xóa một cách bắt buộc
                        try:
                            process.terminate()
                            process.wait(timeout=3)
                        except Exception:
                            process.kill()
                    
                    # Cập nhật run_state.json
                    state = cls.get_run_state(simulation_id)
                    if state:
                        state.runner_status = RunnerStatus.STOPPED
                        state.twitter_running = False
                        state.reddit_running = False
                        state.completed_at = datetime.now().isoformat()
                        state.error = "Server shut down, simulation terminated."
                        cls._save_run_state(state)
                    
                    # Đồng thời cập nhật trạng thái `stopped` cho tệp (file) state.json
                    try:
                        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
                        state_file = os.path.join(sim_dir, "state.json")
                        logger.info(f"Attempting to update state.json: {state_file}")
                        if os.path.exists(state_file):
                            with open(state_file, 'r', encoding='utf-8') as f:
                                state_data = json.load(f)
                            state_data['status'] = 'stopped'
                            state_data['updated_at'] = datetime.now().isoformat()
                            with open(state_file, 'w', encoding='utf-8') as f:
                                json.dump(state_data, f, indent=2, ensure_ascii=False)
                            logger.info(f"Updated state.json status to 'stopped': {simulation_id}")
                        else:
                            logger.warning(f"state.json not found: {state_file}")
                    except Exception as state_err:
                        logger.warning(f"Failed to update state.json: {simulation_id}, error={state_err}")
                        
            except Exception as e:
                logger.error(f"Failed to clean up process: {simulation_id}, error={e}")
        
        # Đóng tất cả tệp xử lý file handles (Log file, Errors File)
        for simulation_id, file_handle in list(cls._stdout_files.items()):
            try:
                if file_handle:
                    file_handle.close()
            except Exception:
                pass
        cls._stdout_files.clear()
        
        for simulation_id, file_handle in list(cls._stderr_files.items()):
            try:
                if file_handle:
                    file_handle.close()
            except Exception:
                pass
        cls._stderr_files.clear()
        
        # Dọn dẹp trạng thái ở trong Ram Memory
        cls._processes.clear()
        cls._action_queues.clear()
        
        logger.info("Simulation process clean up completed.")
    
    @classmethod
    def register_cleanup(cls):
        """
        Đăng ký một lệnh Dọn dẹp (Cleanup command)
        
        Trong lúc chuẩn bị khởi tạo App Flask, mình sẽ thiết lập nó sao cho gọi là máy chủ kết thúc (tắt) mọi quá trình (Simulation Process)
        """
        global _cleanup_registered
        
        if _cleanup_registered:
            return
        
        # Flask ở trong cơ chế gỡ rối `debug`, lúc này chỉ đăng ký ứng dụng để cho thằng app.run làm (Werkzeug) 
        # WERKZEUG_RUN_MAIN=true Đại diện quá trình tiến trình máy chủ được nạp lại
        # Nhưng nó sẽ ko apply điều này ở Production nếu app không debug
        is_reloader_process = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
        is_debug_mode = os.environ.get('FLASK_DEBUG') == '1' or os.environ.get('WERKZEUG_RUN_MAIN') is not None
        
        # Trong DebugMode, chúng ta chỉ cho phép re-loader Child-process chạy. Production vẫn luôn phải chạy process này
        if is_debug_mode and not is_reloader_process:
            _cleanup_registered = True  # Check list đã lưu lại. Hủy quyền yêu cầu thêm
            return
        
        # Lưu các tín hiệu để trả về Signal handling sau khi dừng Process hoàn tất 
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)
        # SIGHUP Chỉ xuất hiện trong Unix (Mac/Linux), Windows không có cái này
        original_sighup = None
        has_sighup = hasattr(signal, 'SIGHUP')
        if has_sighup:
            original_sighup = signal.getsignal(signal.SIGHUP)
        
        def cleanup_handler(signum=None, frame=None):
            """Xử lý điều hướng tín hiệu (Signal Routing): Bắt đầu dọn tiến trình xong gởi lệnh báo (Original Processing Router)"""
            # Chỉ báo nhật kí (log) nếu có tiến trình (Process) cần xử lý
            if cls._processes or cls._graph_memory_enabled:
                logger.info(f"Received signal {signum}, starting clean up...")
            cls.cleanup_all_simulations()
            
            # Gửi tín hiệu gọi hàm báo (handling functions) lúc đấy của Flask => App được tự do ngắt điện
            if signum == signal.SIGINT and callable(original_sigint):
                original_sigint(signum, frame)
            elif signum == signal.SIGTERM and callable(original_sigterm):
                original_sigterm(signum, frame)
            elif has_sighup and signum == signal.SIGHUP:
                # SIGHUP: Được trả về khi máy chủ bị dừng (Terminal Closed)
                if callable(original_sighup):
                    original_sighup(signum, frame)
                else:
                    # Mặc định hành vi: Đóng bình thường => sys.exit(0) "Tạm biệt các hành khách"
                    sys.exit(0)
            else:
                # Hành động ở cơ sở gốc (Root) không gọi được (SIG_DFL) => Hãy phát cảnh báo
                raise KeyboardInterrupt
        
        # Một phương án khác nếu tín hiệu đăng ký xử lý gặp khó (Fallback Option) 
        atexit.register(cls.cleanup_all_simulations)
        
        # Đăng ký quản lý báo tín hiệu (Chỉ riêng trong chủ / main thread có cái luồng)
        try:
            # SIGTERM:  Tín hiệu gốc của Kill Server (Linux/Mac)
            signal.signal(signal.SIGTERM, cleanup_handler)
            # SIGINT: Bấm lệnh Control + C / Ctrl+C
            signal.signal(signal.SIGINT, cleanup_handler)
            # SIGHUP: Máy bị đóng (Unix OS)
            if has_sighup:
                signal.signal(signal.SIGHUP, cleanup_handler)
        except ValueError:
            # Không ở MainThread => Sử dụng được duy nhất fallback Atexit
            logger.warning("Failed to register signal handlers (not in main thread). Falling back to atexit.")
        
        _cleanup_registered = True
    
    @classmethod
    def get_running_simulations(cls) -> List[str]:
        """
        Lấy danh sách tất cả các ID của các phiên mô phỏng đang hoạt động
        """
        running = []
        for sim_id, process in cls._processes.items():
            if process.poll() is None:
                running.append(sim_id)
        return running
    
    # ============== Tính năng Phỏng vấn (Interview) ==============
    
    @classmethod
    def check_env_alive(cls, simulation_id: str) -> bool:
        """
        Kiểm tra xem environment còn sống không (có thể nhận lệnh Interview)

        Args:
            simulation_id: Simulation ID

        Returns:
            True Nếu environment còn sống, False nghĩa là đã đóng
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        if not os.path.exists(sim_dir):
            return False

        ipc_client = SimulationIPCClient(sim_dir)
        return ipc_client.check_env_alive()

    @classmethod
    def get_env_status_detail(cls, simulation_id: str) -> Dict[str, Any]:
        """
        Lấy thông tin chi tiết về trạng thái của environment

        Args:
            simulation_id: Mô phỏng ID

        Returns:
            Bảng trạng thái chi tiết (Dictionary) bao gồm: status, twitter_available, reddit_available, timestamp
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        status_file = os.path.join(sim_dir, "env_status.json")
        
        default_status = {
            "status": "stopped",
            "twitter_available": False,
            "reddit_available": False,
            "timestamp": None
        }
        
        if not os.path.exists(status_file):
            return default_status
        
        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                status = json.load(f)
            return {
                "status": status.get("status", "stopped"),
                "twitter_available": status.get("twitter_available", False),
                "reddit_available": status.get("reddit_available", False),
                "timestamp": status.get("timestamp")
            }
        except (json.JSONDecodeError, OSError):
            return default_status

    @classmethod
    def interview_agent(
        cls,
        simulation_id: str,
        agent_id: int,
        prompt: str,
        platform: str = None,
        timeout: float = 60.0
    ) -> Dict[str, Any]:
        """
        Phỏng vấn trên 1 Agent

        Args:
            simulation_id: ID mô phỏng
            agent_id: Agent ID
            prompt: Câu hỏi phỏng vấn
            platform: Chỉ định nền tảng (Tùy chọn/Optional)
                - "twitter":  Chỉ PV trên account Twitter
                - "reddit": Chỉ PV trên account Reddit
                - None: Phỏng vấn chéo trên cả hai nền tảng, trả về kết quả hợp lại (Nếu chạy mô phỏng nền tảng kép)
            timeout: Thời gian chờ tối đa (giây)

        Returns:
            Từ điền chứa kết quả PV

        Raises:
            ValueError: Không có mô phỏng hoặc environment ko chạy
            TimeoutError: Đang chờ phản hồi bị timeout
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        if not os.path.exists(sim_dir):
            raise ValueError(f"Simulation does not exist: {simulation_id}")

        ipc_client = SimulationIPCClient(sim_dir)

        if not ipc_client.check_env_alive():
            raise ValueError(f"Simulation environment is not running or closed, cannot perform Interview: {simulation_id}")

        logger.info(f"Sending Interview Command: simulation_id={simulation_id}, agent_id={agent_id}, platform={platform}")

        response = ipc_client.send_interview(
            agent_id=agent_id,
            prompt=prompt,
            platform=platform,
            timeout=timeout
        )

        if response.status.value == "completed":
            return {
                "success": True,
                "agent_id": agent_id,
                "prompt": prompt,
                "result": response.result,
                "timestamp": response.timestamp
            }
        else:
            return {
                "success": False,
                "agent_id": agent_id,
                "prompt": prompt,
                "error": response.error,
                "timestamp": response.timestamp
            }
    
    @classmethod
    def interview_agents_batch(
        cls,
        simulation_id: str,
        interviews: List[Dict[str, Any]],
        platform: str = None,
        timeout: float = 120.0
    ) -> Dict[str, Any]:
        """
        Phỏng vấn hàng loạt nhiều Agent

        Args:
            simulation_id: ID mô phỏng
            interviews:  Danh sách nội dung phỏng vấn, mỗi phần tử (element) chứa {"agent_id": int, "prompt": str, "platform": str(tùy chọn)}
            platform: Nền tảng mặc định (Nếu không chọn riêng cho từng phần tử)
                - "twitter": Mặc định chỉ dùng mạng Twitter
                - "reddit": Mặc định chỉ dùng mạng Reddit
                - None: Phỏng vấn gộp trên cả hai nền tảng với mỗi Agent
            timeout: Hết thời gian chờ (ms) (seconds)

        Returns:
            Dict từ điển với các kết quả phỏng vấn hàng loạt

        Raises:
            ValueError: Chưa có tiến trình chạy mô phỏng
            TimeoutError: Phỏng vấn lâu quá (Timeout timeout timeout)
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        if not os.path.exists(sim_dir):
            raise ValueError(f"Simulation does not exist: {simulation_id}")

        ipc_client = SimulationIPCClient(sim_dir)

        if not ipc_client.check_env_alive():
            raise ValueError(f"Simulation environment is not running or closed, cannot perform Interview: {simulation_id}")

        logger.info(f"Sending batch Interview command: simulation_id={simulation_id}, count={len(interviews)}, platform={platform}")

        response = ipc_client.send_batch_interview(
            interviews=interviews,
            platform=platform,
            timeout=timeout
        )

        if response.status.value == "completed":
            return {
                "success": True,
                "interviews_count": len(interviews),
                "result": response.result,
                "timestamp": response.timestamp
            }
        else:
            return {
                "success": False,
                "interviews_count": len(interviews),
                "error": response.error,
                "timestamp": response.timestamp
            }
    
    @classmethod
    def interview_all_agents(
        cls,
        simulation_id: str,
        prompt: str,
        platform: str = None,
        timeout: float = 180.0
    ) -> Dict[str, Any]:
        """
        Phỏng vấn TOÀN BỘ Agent (Phỏng vấn tổng quan - Global interview)

        Hỏi một câu hỏi với toàn bộ Agent đang có trong phiên mô phỏng hiện tại

        Args:
            simulation_id: ID mô phỏng 
            prompt: Câu hỏi (Cho tất cả các Agent)
            platform: Quyết định nền tảng (Platform decision)
                - "twitter": Trỏ tới Twitter Platform
                - "reddit": Trỏ tới Reddit Platform
                - None: Interview kết hợp trên cả nền tảng của từng agent
            timeout:  Timeout

        Returns:
            Kết quả của toàn thể hội đồng Agents (Lớp/Nhóm)
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        if not os.path.exists(sim_dir):
            raise ValueError(f"Simulation does not exist: {simulation_id}")

        # Fetch All Agents profile (Lấy thông tin agents từ phần thiết lập)
        config_path = os.path.join(sim_dir, "simulation_config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"Simulation config not found: {simulation_id}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        agent_configs = config.get("agent_configs", [])
        if not agent_configs:
            raise ValueError(f"No Agent defined under simulation configs: {simulation_id}")

        # Tập hợp danh sách phỏng vấn tất cả
        interviews = []
        for agent_config in agent_configs:
            agent_id = agent_config.get("agent_id")
            if agent_id is not None:
                interviews.append({
                    "agent_id": agent_id,
                    "prompt": prompt
                })

        logger.info(f"Sending GLOBAL Interview command: simulation_id={simulation_id}, agent_count={len(interviews)}, platform={platform}")

        return cls.interview_agents_batch(
            simulation_id=simulation_id,
            interviews=interviews,
            platform=platform,
            timeout=timeout
        )
    
    @classmethod
    def close_simulation_env(
        cls,
        simulation_id: str,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Đóng Environment giả lập (Không phải dừng process tắt hẳn nó đi)
        
        Gửi lệnh ra hiệu cho Simulation ngắt bỏ tiến trình để các processes thoát ra an toàn và êm đẹp về trạng thái đang chờ nhận lệnh
        
        Args:
            simulation_id:  ID Simulation
            timeout: Timeout chờ kết nối
            
        Returns:
            Kiểu từ điển: Quá trình (Process Status execution)
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        if not os.path.exists(sim_dir):
            raise ValueError(f"Simulation does not exist: {simulation_id}")
        
        ipc_client = SimulationIPCClient(sim_dir)
        
        if not ipc_client.check_env_alive():
            return {
                "success": True,
                "message": "Environment is already closed"
            }
        
        logger.info(f"Sending command to close Environment: simulation_id={simulation_id}")
        
        try:
            response = ipc_client.send_close_env(timeout=timeout)
            
            return {
                "success": response.status.value == "completed",
                "message": "Environment close command sent",
                "result": response.result,
                "timestamp": response.timestamp
            }
        except TimeoutError:
            # Hết thời gian chờ nguyên nhân lớn nhất là vì Simulation environment đang đóng giữa chừng.
            return {
                "success": True,
                "message": "Environment close command sent (timeout waiting for response, env might be closing)"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to send close env command: {str(e)}"
            }
    
    @classmethod
    def _get_interview_history_from_db(
        cls,
        db_path: str,
        platform_name: str,
        agent_id: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Lấy lịch sử phỏng vấn từ Local Database của nền tảng"""
        import sqlite3
        
        if not os.path.exists(db_path):
            return []
        
        results = []
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            if agent_id is not None:
                cursor.execute("""
                    SELECT user_id, info, created_at
                    FROM trace
                    WHERE action = 'interview' AND user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (agent_id, limit))
            else:
                cursor.execute("""
                    SELECT user_id, info, created_at
                    FROM trace
                    WHERE action = 'interview'
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))
            
            for user_id, info_json, created_at in cursor.fetchall():
                try:
                    info = json.loads(info_json) if info_json else {}
                except json.JSONDecodeError:
                    info = {"raw": info_json}
                
                results.append({
                    "agent_id": user_id,
                    "response": info.get("response", info),
                    "prompt": info.get("prompt", ""),
                    "timestamp": created_at,
                    "platform": platform_name
                })
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to load Interview history ({platform_name}): {e}")
        
        return results

    @classmethod
    def get_interview_history(
        cls,
        simulation_id: str,
        platform: str = None,
        agent_id: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Lịch sử lấy danh sách câu trả lời của các câu hỏi với agents (Đọc từ DataBase db)
        
        Args:
            simulation_id: Nhận dạng ID cho mỗi Simulation
            platform: Chị định Nền tảng (reddit/twitter/None)
                - "reddit": Chỉ trên reddit 
                - "twitter": Chỉ lấy records ghi được trên mạng xã hội twitter giả lập
                - None: Kết hợp lấy logs của cả hai social network
            agent_id: Cung cấp tùy chọn cho loại Agent qua ID
            limit: Lượng dữ liệu load tối đa cho 1 request get query trên 1 nền tảng
            
        Returns:
            Danh sách lưu vết record lịch sử phỏng vấn của các agents
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        
        results = []
        
        # Xác nhận nền tảng cung cấp cho truy vấn
        if platform in ("reddit", "twitter"):
            platforms = [platform]
        else:
            # Nếu người dùng để trống, có nghĩa là gọi tất cả kết quả 
            platforms = ["twitter", "reddit"]
        
        for p in platforms:
            db_path = os.path.join(sim_dir, f"{p}_simulation.db")
            platform_results = cls._get_interview_history_from_db(
                db_path=db_path,
                platform_name=p,
                agent_id=agent_id,
                limit=limit
            )
            results.extend(platform_results)
        
        # Sắp xếp chúng lại bằng thời gian gần nhất lên trước
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Cho trường hợp query kết hợp nhiều nền tảng, phải tiến hành gọt lấy đúng 1 giới hạn nhất định
        if len(platforms) > 1 and len(results) > limit:
            results = results[:limit]
        
        return results

