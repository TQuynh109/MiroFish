"""
Trình tạo tạo ra Profile Agent (Hồ sơ Nhân vật) cho Agent bằng Framework OASIS
Chuyển đổi dữ liệu Thực thể được Query từ Zep ra chuẩn định dạng của các Agent tham gia vào mạng

Nâng cấp cải thiện:
1. Kết hợp dùng chức năng Search trên Zep để lấy Profile giàu sắc thái
2. Gen các cấu hình về tính cách một cách sắc xảo và cực sâu cho Prompt
3. Nhận dạng rạch ròi người với Group/Công ty/Phe phái trên social network
"""

import json
import random
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from openai import OpenAI
from zep_cloud.client import Zep

from ..config import Config
from ..utils.logger import get_logger
from ..utils.llm_cost import create_tracked_chat_completion
from .zep_entity_reader import EntityNode, ZepEntityReader

logger = get_logger('mirofish.oasis_profile')


@dataclass
class OasisAgentProfile:
    """Cấu trúc của Dataclass Profile Agent qua quy định của OASIS"""
    # Các Field thông dụng (Common data)
    user_id: int
    user_name: str
    name: str
    bio: str
    persona: str
    
    # Chọn bổ sung (Tùy chọn) - Thông số nền tảng Reddit (Karma)
    karma: int = 1000
    
    # Chọn bổ sung (Tùy chọn) - Thông số nền tảng Twitter 
    friend_count: int = 100
    follower_count: int = 150
    statuses_count: int = 500
    
    # Một số Thông tin Data cá nhân bổ sung (Bóp để tăng tính Thực tế nếu LLM sinh ra)
    age: Optional[int] = None
    gender: Optional[str] = None
    mbti: Optional[str] = None
    country: Optional[str] = None
    profession: Optional[str] = None
    interested_topics: List[str] = field(default_factory=list)
    
    # Lịch sử thông tin entity gốc được lấy 
    source_entity_uuid: Optional[str] = None
    source_entity_type: Optional[str] = None
    
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    
    def to_reddit_format(self) -> Dict[str, Any]:
        """Convert trả ra cho định dạng Agent reddit"""
        profile = {
            "user_id": self.user_id,
            "username": self.user_name,  # Source mã của OASIS Library yêu cầu không có dấu "_" cho param username
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "karma": self.karma,
            "created_at": self.created_at,
        }
        
        # Merge Thông tin Data Profile Cá nhân (Nếu CÓ)
        if self.age:
            profile["age"] = self.age
        if self.gender:
            profile["gender"] = self.gender
        if self.mbti:
            profile["mbti"] = self.mbti
        if self.country:
            profile["country"] = self.country
        if self.profession:
            profile["profession"] = self.profession
        if self.interested_topics:
            profile["interested_topics"] = self.interested_topics
        
        return profile
    
    def to_twitter_format(self) -> Dict[str, Any]:
        """Convert trả ra cho định dạng Agent Twitter"""
        profile = {
            "user_id": self.user_id,
            "username": self.user_name,  # Tương tự như trên
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "friend_count": self.friend_count,
            "follower_count": self.follower_count,
            "statuses_count": self.statuses_count,
            "created_at": self.created_at,
        }
        
        # Merge Thông tin Data Profile Cả nhân
        if self.age:
            profile["age"] = self.age
        if self.gender:
            profile["gender"] = self.gender
        if self.mbti:
            profile["mbti"] = self.mbti
        if self.country:
            profile["country"] = self.country
        if self.profession:
            profile["profession"] = self.profession
        if self.interested_topics:
            profile["interested_topics"] = self.interested_topics
        
        return profile
    
    def to_dict(self) -> Dict[str, Any]:
        """Quy đổi thành toàn bộ Dictionary Cấu Trúc Khép Kín """
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "karma": self.karma,
            "friend_count": self.friend_count,
            "follower_count": self.follower_count,
            "statuses_count": self.statuses_count,
            "age": self.age,
            "gender": self.gender,
            "mbti": self.mbti,
            "country": self.country,
            "profession": self.profession,
            "interested_topics": self.interested_topics,
            "source_entity_uuid": self.source_entity_uuid,
            "source_entity_type": self.source_entity_type,
            "created_at": self.created_at,
        }


class OasisProfileGenerator:
    """
    Trình Gen Profile cho Simulation (Hệ OASIS)
    
    Sử dụng các Node Entity lấy được từ ZEP -> OASIS Mocks cho Simulation Agent
    
    Các Option Cải Tiến Tối Ưu Tích Hợp:
    1. Có liên kết với Server Zep API cho bước Query Dữ liệu từ Vector Database
    2. Tập trung Mô tả Tiểu sửa (Background) (Nhấn mạnh Nghề nghiệp/Tính cách/StatusMXH/vvv)
    3. Ngắt rời loại Person ra bên ngoài để nhận thức Phân Cấp Group
    """
    
    # 16 tính cách của Con người (Quy Chuẩn)
    MBTI_TYPES = [
        "INTJ", "INTP", "ENTJ", "ENTP",
        "INFJ", "INFP", "ENFJ", "ENFP",
        "ISTJ", "ISFJ", "ESTJ", "ESFJ",
        "ISTP", "ISFP", "ESTP", "ESFP"
    ]
    
    # List mảng quốc tịch Cơ Bản
    COUNTRIES = [
        "China", "US", "UK", "Japan", "Germany", "France", 
        "Canada", "Australia", "Brazil", "India", "South Korea"
    ]
    
    # Thực thể nhận biết là người 1 mình (Độc Lập, 1 Person)
    INDIVIDUAL_ENTITY_TYPES = [
        "student", "alumni", "professor", "person", "publicfigure", 
        "expert", "faculty", "official", "journalist", "activist"
    ]
    
    # Thực Thể được xem là một tổ chức
    GROUP_ENTITY_TYPES = [
        "university", "governmentagency", "organization", "ngo", 
        "mediaoutlet", "company", "institution", "group", "community"
    ]
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        zep_api_key: Optional[str] = None,
        graph_id: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model_name = model_name or Config.LLM_MODEL_NAME
        
        if not self.api_key:
            raise ValueError("Không tìm thấy LLM_API_KEY")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self._runtime_metadata: Dict[str, Any] = {
            "component": "oasis_profile_generator",
            "phase": "generate_profiles",
        }
        
        # Kết nối lên trên ZEP Database Search Context
        self.zep_api_key = zep_api_key or Config.ZEP_API_KEY
        self.zep_client = None
        self.graph_id = graph_id
        
        if self.zep_api_key:
            try:
                self.zep_client = Zep(api_key=self.zep_api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Zep client: {e}")
    
    def generate_profile_from_entity(
        self, 
        entity: EntityNode, 
        user_id: int,
        use_llm: bool = True
    ) -> OasisAgentProfile:
        """
        Bắt đầu Gen Profile từ Entity được móc từ data từ zep
        
        Args:
            entity: Thực thể Zep
            user_id: Số ID để map (sử dụng trên OASIS)
            use_llm: Chọn bật tắt xem tạo Profile có dùng gen nhân vật bằng LLM
            
        Returns:
            OasisAgentProfile
        """
        entity_type = entity.get_entity_type() or "Entity"
        
        # Mức cơ bản thông tin
        name = entity.name
        user_name = self._generate_username(name)
        
        # Build các Context thông tin liên quan lại 
        context = self._build_entity_context(entity)
        
        if use_llm:
            # Gửi Prompt lên LLM
            profile_data = self._generate_profile_with_llm(
                entity_name=name,
                entity_type=entity_type,
                entity_summary=entity.summary,
                entity_attributes=entity.attributes,
                context=context
            )
        else:
            # Chạy hàm Auto rule nếu LLm tắt
            profile_data = self._generate_profile_rule_based(
                entity_name=name,
                entity_type=entity_type,
                entity_summary=entity.summary,
                entity_attributes=entity.attributes
            )
        
        return OasisAgentProfile(
            user_id=user_id,
            user_name=user_name,
            name=name,
            bio=profile_data.get("bio", f"{entity_type}: {name}"),
            persona=profile_data.get("persona", entity.summary or f"A {entity_type} named {name}."),
            karma=profile_data.get("karma", random.randint(500, 5000)),
            friend_count=profile_data.get("friend_count", random.randint(50, 500)),
            follower_count=profile_data.get("follower_count", random.randint(100, 1000)),
            statuses_count=profile_data.get("statuses_count", random.randint(100, 2000)),
            age=profile_data.get("age"),
            gender=profile_data.get("gender"),
            mbti=profile_data.get("mbti"),
            country=profile_data.get("country"),
            profession=profile_data.get("profession"),
            interested_topics=profile_data.get("interested_topics", []),
            source_entity_uuid=entity.uuid,
            source_entity_type=entity_type,
        )
    
    def _generate_username(self, name: str) -> str:
        """Thêm chức năng Generate Username username ngẫu nhiên"""
        # Hút bỏ khoảng trống và dấu đặc biệt
        username = name.lower().replace(" ", "_")
        username = ''.join(c for c in username if c.isalnum() or c == '_')
        
        # Chèn thêm hậu tố cho bớt đụng hàng
        suffix = random.randint(100, 999)
        return f"{username}_{suffix}"
    
    def _search_zep_for_entity(self, entity: EntityNode) -> Dict[str, Any]:
        """
        Dùng hỗn hợp lệnh Query DB Vector qua Zep để lấy các fact/sự kiện liên quan về 1 thực thể.
        
        Vì Zep chưa hỗ trợ hỗn hợp cả 2 cùng một lúc, nên cần tìm song song từ Edge và Node sau đó gộp kết quả.
        
        Args:
            entity: Đầu cắm Thực Thể Node
            
        Returns:
            Dictionary gồm facts, node_summaries, context
        """
        import concurrent.futures
        
        if not self.zep_client:
            return {"facts": [], "node_summaries": [], "context": ""}
        
        entity_name = entity.name
        
        results = {
            "facts": [],
            "node_summaries": [],
            "context": ""
        }
        
        # Yêu cầu graph_id mới truy vấn được
        if not self.graph_id:
            logger.debug(f"Skipping Zep search: graph_id not set")
            return results
        
        comprehensive_query = f"Provide all facts, activities, relationships, and context about: {entity_name}"
        
        def search_edges():
            """Lookup cạnh relations - Kết hợp cơ chế retry"""
            max_retries = 3
            last_exception = None
            delay = 2.0
            
            for attempt in range(max_retries):
                try:
                    return self.zep_client.graph.search(
                        query=comprehensive_query,
                        graph_id=self.graph_id,
                        limit=30,
                        scope="edges",
                        reranker="rrf"
                    )
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.debug(f"Zep Edge search failed on attempt {attempt + 1}: {str(e)[:80]}, retrying...")
                        time.sleep(delay)
                        delay *= 2
                    else:
                        logger.debug(f"Zep Edge search entirely failed after {max_retries} attempts: {e}")
            return None
        
        def search_nodes():
            """Lookup mảng Node entity tóm tắt - Kết hợp cơ chế retry"""
            max_retries = 3
            last_exception = None
            delay = 2.0
            
            for attempt in range(max_retries):
                try:
                    return self.zep_client.graph.search(
                        query=comprehensive_query,
                        graph_id=self.graph_id,
                        limit=20,
                        scope="nodes",
                        reranker="rrf"
                    )
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.debug(f"Zep Node search failed on attempt {attempt + 1}: {str(e)[:80]}, retrying...")
                        time.sleep(delay)
                        delay *= 2
                    else:
                        logger.debug(f"Zep Node search entirely failed after {max_retries} attempts: {e}")
            return None
        
        try:
            # Cho chạy cả task Cạnh và Node song song
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                edge_future = executor.submit(search_edges)
                node_future = executor.submit(search_nodes)
                
                # Fetch result trả về
                edge_result = edge_future.result(timeout=30)
                node_result = node_future.result(timeout=30)
            
            # Quản lý kết quả fact của các cạnh Edge
            all_facts = set()
            if edge_result and hasattr(edge_result, 'edges') and edge_result.edges:
                for edge in edge_result.edges:
                    if hasattr(edge, 'fact') and edge.fact:
                        all_facts.add(edge.fact)
            results["facts"] = list(all_facts)
            
            # Quản lý kết quả tên thực thể và summary của quá trình search Node
            all_summaries = set()
            if node_result and hasattr(node_result, 'nodes') and node_result.nodes:
                for node in node_result.nodes:
                    if hasattr(node, 'summary') and node.summary:
                        all_summaries.add(node.summary)
                    if hasattr(node, 'name') and node.name and node.name != entity_name:
                        all_summaries.add(f"Related Entities: {node.name}")
            results["node_summaries"] = list(all_summaries)
            
            # Tổng hợp ra 1 chuỗi Context bao quanh
            context_parts = []
            if results["facts"]:
                context_parts.append("Facts & Infomation:\n" + "\n".join(f"- {f}" for f in results["facts"][:20]))
            if results["node_summaries"]:
                context_parts.append("Related Entities:\n" + "\n".join(f"- {s}" for s in results["node_summaries"][:10]))
            results["context"] = "\n\n".join(context_parts)
            
            logger.info(f"Zep unified search completed: {entity_name}, fetched {len(results['facts'])} facts, {len(results['node_summaries'])} related nodes")
            
        except concurrent.futures.TimeoutError:
            logger.warning(f"Zep Retrieval Time-Out ({entity_name})")
        except Exception as e:
            logger.warning(f"Zep Retrieval Failed ({entity_name}): {e}")
        
        return results
    
    def _build_entity_context(self, entity: EntityNode) -> str:
        """
        Nối tất cả info thu được liên quan thành 1 chuỗi Context bao quanh hoàn chỉnh cho Entity
        
        Nó sẽ lấy:
        1. Context từ các cạnh hiện tại đã gắn Entity (Dữ liệu về relation/fact)
        2. Mô tả sơ lược thêm của các Node dính liền
        3. Cuối cùng nhồi thêm những thứ moi được từ quá trình chạy Zep search hỗn hợp bên trên
        """
        context_parts = []
        
        # 1. Thu thập Attributes/Properties của node nếu có
        if entity.attributes:
            attrs = []
            for key, value in entity.attributes.items():
                if value and str(value).strip():
                    attrs.append(f"- {key}: {value}")
            if attrs:
                context_parts.append("### Entity Attributes\n" + "\n".join(attrs))
        
        # 2. Add các facts và mô phỏng cạnh (Relationship/Facts)
        existing_facts = set()
        if entity.related_edges:
            relationships = []
            for edge in entity.related_edges:  # Khum bị giới hạn SL
                fact = edge.get("fact", "")
                edge_name = edge.get("edge_name", "")
                direction = edge.get("direction", "")
                
                if fact:
                    relationships.append(f"- {fact}")
                    existing_facts.add(fact)
                elif edge_name:
                    if direction == "outgoing":
                        relationships.append(f"- {entity.name} --[{edge_name}]--> (Related Entity)")
                    else:
                        relationships.append(f"- (Related Entity) --[{edge_name}]--> {entity.name}")
            
            if relationships:
                context_parts.append("### Facts & Relationships\n" + "\n".join(relationships))
        
        # 3. Kẹp chi tiết miêu tả về node anh em cạnh bên
        if entity.related_nodes:
            related_info = []
            for node in entity.related_nodes:  # Không block giới hạn số lượng
                node_name = node.get("name", "")
                node_labels = node.get("labels", [])
                node_summary = node.get("summary", "")
                
                # Bỏ nhãn mặc định khỏi string xuất ra
                custom_labels = [l for l in node_labels if l not in ["Entity", "Node"]]
                label_str = f" ({', '.join(custom_labels)})" if custom_labels else ""
                
                if node_summary:
                    related_info.append(f"- **{node_name}**{label_str}: {node_summary}")
                else:
                    related_info.append(f"- **{node_name}**{label_str}")
            
            if related_info:
                context_parts.append("### Related Entity Info\n" + "\n".join(related_info))
        
        # 4. Sử dụng kết quả Query Search từ hàm zep
        zep_results = self._search_zep_for_entity(entity)
        
        if zep_results.get("facts"):
            # Lọc bớt cặn trùng lắp: không add những Fact đã có ở mục số 2
            new_facts = [f for f in zep_results["facts"] if f not in existing_facts]
            if new_facts:
                context_parts.append("### Facts retrieved via ZEP\n" + "\n".join(f"- {f}" for f in new_facts[:15]))
        
        if zep_results.get("node_summaries"):
            context_parts.append("### Entity nodes retrieved via Zep\n" + "\n".join(f"- {s}" for s in zep_results["node_summaries"][:10]))
        
        return "\n\n".join(context_parts)
    
    def _is_individual_entity(self, entity_type: str) -> bool:
        """KTra và True cho các dạng người Single Person"""
        return entity_type.lower() in self.INDIVIDUAL_ENTITY_TYPES
    
    def _is_group_entity(self, entity_type: str) -> bool:
        """KTra xem thực thể hiện tại là Group/Media/Company ..."""
        return entity_type.lower() in self.GROUP_ENTITY_TYPES
    
    def _generate_profile_with_llm(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str
    ) -> Dict[str, Any]:
        """
        Dùng LLM cấp lại Profile mô phỏng tính cách cụ thể và rõ nét nhất
        
        Kiểm tra đầu vào Entity type để chia nhánh:
        - Cá nhân: Tạo setting miêu tả cá nhân, công việc riêng
        - Tập Thể/Cơ quan/Tổ chức: Tạo Profile cho 1 tài khoản đại điện tổ chức đó
        """
        
        is_individual = self._is_individual_entity(entity_type)
        
        if is_individual:
            prompt = self._build_individual_persona_prompt(
                entity_name, entity_type, entity_summary, entity_attributes, context
            )
        else:
            prompt = self._build_group_persona_prompt(
                entity_name, entity_type, entity_summary, entity_attributes, context
            )

        # Retry liên tục vòng lặp nếu LLM timeout hoặc fail
        max_attempts = 3
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                response = create_tracked_chat_completion(
                    client=self.client,
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt(is_individual)},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7 - (attempt * 0.1),  # Giảm tính sáng tạo ngẫu nhiên đi một chút mỗi khi fail để tăng khả năng thành công ở vòng tiếp theo
                    metadata=self._runtime_metadata,
                )
                
                content = response.choices[0].message.content
                
                # Check LLM trả về vì sao bị kẹt lại/Dừng lại (Finish_Reason khác "stop")
                finish_reason = response.choices[0].finish_reason
                if finish_reason == 'length':
                    logger.warning(f"LLM output truncated (attempt {attempt+1}), attempting to fix...")
                    content = self._fix_truncated_json(content)
                
                # Parse chép vào JSON
                try:
                    result = json.loads(content)
                    
                    # Xác minh tham số được Bot gen thành công chưa
                    if "bio" not in result or not result["bio"]:
                        result["bio"] = entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}"
                    if "persona" not in result or not result["persona"]:
                        result["persona"] = entity_summary or f"{entity_name} is a {entity_type}."
                    
                    return result
                    
                except json.JSONDecodeError as je:
                    logger.warning(f"JSON Parsing Failed (attempt {attempt+1}): {str(je)[:80]}")
                    
                    # Tool sửa lỗi JSON syntax tự chế
                    result = self._try_fix_json(content, entity_name, entity_type, entity_summary)
                    if result.get("_fixed"):
                        del result["_fixed"]
                        return result
                    
                    last_error = je
                    
            except Exception as e:
                logger.warning(f"LLM call Failed (attempt {attempt+1}): {str(e)[:80]}")
                last_error = e
                import time
                time.sleep(1 * (attempt + 1))  # Exponential backoff
        
        logger.warning(f"Generating profile through LLM failed totally after {max_attempts} attempts: {last_error}, switching to basic hard-code rules configs")
        return self._generate_profile_rule_based(
            entity_name, entity_type, entity_summary, entity_attributes
        )
    
    def _fix_truncated_json(self, content: str) -> str:
        """Fix Output JSON bị Max_tokens đè cắt gãy"""
        import re
        
        # Bọc ngoài
        content = content.strip()
        
        # Điểm kiểm chứng xem dấu ngoặc được đầy đủ hay chưa
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')
        
        # Check string xem đủ không
        # Nếu phần tử cuối cùng k phải dấu câu đóng block, tự chèn vào
        if content and content[-1] not in '",}]':
            # Ngoặc cho đít chuỗi
            content += '"'
        
        # Ngoặc block
        content += ']' * open_brackets
        content += '}' * open_braces
        
        return content
    
    def _try_fix_json(self, content: str, entity_name: str, entity_type: str, entity_summary: str = "") -> Dict[str, Any]:
        """Thử Fix nội dung JSON"""
        import re
        
        # 1. Bọc json trước
        content = self._fix_truncated_json(content)
        
        # 2. Extract block lớn
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group()
            
            # 3. Clean lại các chuỗi xuống dòng
            # Regex lôi code ra ngoài
            def fix_string_newlines(match):
                s = match.group(0)
                # Escape code line chuyển cho space cho an toàn
                s = s.replace('\n', ' ').replace('\r', ' ')
                # Chém các khoảng trống còn dư quá gắt
                s = re.sub(r'\s+', ' ', s)
                return s
            
            # Khớp lại nội dung
            json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string_newlines, json_str)
            
            # 4. Bắt đầu json parse
            try:
                result = json.loads(json_str)
                result["_fixed"] = True
                return result
            except json.JSONDecodeError as e:
                # 5. Phá lấu clean xóa nếu còn bị lỗi Control char ẩn (0x00 đến 0x1f ...)
                try:
                    # Chém control character
                    json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
                    # Gọt lại khoảng trống dư
                    json_str = re.sub(r'\s+', ' ', json_str)
                    result = json.loads(json_str)
                    result["_fixed"] = True
                    return result
                except:
                    pass
        
        # 6. Rescue lấy các property còn lại mót ra từ đóng hỗn độn
        bio_match = re.search(r'"bio"\s*:\s*"([^"]*)"', content)
        persona_match = re.search(r'"persona"\s*:\s*"([^"]*)', content)  # Bị cắt khúc thì ráng chịu
        
        bio = bio_match.group(1) if bio_match else (entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}")
        persona = persona_match.group(1) if persona_match else (entity_summary or f"{entity_name} is a {entity_type}.")
        
        # Lụm mót được data xịn thì mark là fix thành công
        if bio_match or persona_match:
            logger.info(f"Successfully extracted partial info from corrupted JSON")
            return {
                "bio": bio,
                "persona": persona,
                "_fixed": True
            }
        
        # 7. Failed sạch, quăng cái khung mặc định ra
        logger.warning(f"Failed to fix JSON, returning basic structured data")
        return {
            "bio": entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}",
            "persona": entity_summary or f"{entity_name} is a {entity_type}."
        }
    
    def _get_system_prompt(self, is_individual: bool) -> str:
        """Lấy prompt cho hệ thống"""

        # base_prompt = "You are an expert in generating social media user personas. Generate detailed and realistic personas for public opinion simulation to recreate existing real-world conditions to the greatest extent possible. You must return a valid JSON format; all string values must not contain unescaped line breaks. Use Chinese."

        base_prompt = "Bạn là chuyên gia tạo hồ sơ người dùng mạng xã hội. Hãy tạo các nhân vật chi tiết và chân thực phục vụ cho việc mô phỏng dư luận, nhằm tái hiện tối đa các tình huống thực tế hiện có. Phải trả về định dạng JSON hợp lệ; tất cả các giá trị chuỗi không được chứa ký tự xuống dòng chưa được xử lý (unescaped). Sử dụng tiếng Việt."
        return base_prompt
    
    def _build_individual_persona_prompt(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str
    ) -> str:
        """Tạo prompt nhân vật chi tiết cho thực thể cá nhân"""
        
        attrs_str = json.dumps(entity_attributes, ensure_ascii=False) if entity_attributes else "Không có"
        context_str = context[:3000] if context else "Không có ngữ cảnh bổ sung"
        
#         return f"""Generate a detailed social media user persona for the entity, recreating existing real-world conditions to the greatest extent possible.
 
# Entity Name: {entity_name}
# Entity Type: {entity_type}
# Entity Summary: {entity_summary}
# Entity Attributes: {attrs_str}
 
# Context Information:
# {context_str}
 
# Please generate a JSON containing the following fields:
 
# 1. bio: Social media biography, 200 characters.
# 2. persona: Detailed persona description (2000 words of plain text), which must include:
#    - Basic information (age, occupation, educational background, location)
#    - Background (significant experiences, connection to the event, social relationships)
#    - Personality traits (MBTI type, core personality, emotional expression style)
#    - Social media behavior (posting frequency, content preferences, interaction style, linguistic characteristics)
#    - Stance and views (attitude toward the topic, content that might provoke or move them)
#    - Unique features (catchphrases, special experiences, personal hobbies)
#    - Personal memory (a vital part of the persona, describing the individual's connection to the event and their existing actions/reactions)
# 3. age: Age as a number (must be an integer)
# 4. gender: Gender, must be in English: "male" or "female"
# 5. mbti: MBTI type (e.g., INTJ, ENFP, etc.)
# 6. country: Country (use Chinese, e.g., "中国")
# 7. profession: Occupation
# 8. interested_topics: An array of interested topics
 
# IMPORTANT:
# - All field values must be strings or numbers; do not use line breaks.
# - The 'persona' must be a coherent block of text description.
# - Use Chinese (except for the 'gender' field, which must be English male/female).
# - Content must remain consistent with the entity information.
# - 'age' must be a valid integer; 'gender' must be "male" or "female"."""

        return f"""Tạo hồ sơ người dùng mạng xã hội chi tiết cho thực thể, tái hiện tối đa các tình huống thực tế hiện có.

Tên thực thể: {entity_name}
Loại thực thể: {entity_type}
Tóm tắt thực thể: {entity_summary}
Thuộc tính thực thể: {attrs_str}

Thông tin ngữ cảnh:
{context_str}

Vui lòng tạo JSON bao gồm các trường sau:
 
1. bio: Tiểu sử mạng xã hội, 200 ký tự.
2. persona: Mô tả nhân vật chi tiết (văn bản thuần túy khoảng 2000 từ), cần bao gồm:
   - Thông tin cơ bản (tuổi, nghề nghiệp, trình độ học vấn, nơi ở)
   - Nền tảng nhân vật (trải nghiệm quan trọng, mối liên hệ với sự kiện, quan hệ xã hội)
   - Đặc điểm tính cách (loại MBTI, tính cách cốt lõi, cách biểu đạt cảm xúc)
   - Hành vi mạng xã hội (tần suất đăng bài, sở thích nội dung, phong cách tương tác, đặc điểm ngôn ngữ)
   - Lập trường quan điểm (thái độ đối với chủ đề, nội dung dễ gây kích động hoặc gây xúc động)
   - Đặc điểm độc đáo (câu cửa miệng, trải nghiệm đặc biệt, sở thích cá nhân)
   - Ký ức cá nhân (phần quan trọng của nhân vật, giới thiệu mối liên hệ của cá nhân này với sự kiện, cũng như các hành động và phản ứng đã có của họ trong sự kiện)
3. age: Con số tuổi (phải là số nguyên)
4. gender: Giới tính, phải là tiếng Anh: "male" hoặc "female"
5. mbti: Loại MBTI (như INTJ, ENFP, v.v.)
6. country: Quốc gia (sử dụng tiếng Trung, ví dụ: "中国")
7. profession: Nghề nghiệp
8. interested_topics: Mảng các chủ đề quan tâm
 
QUAN TRỌNG:
- Tất cả giá trị các trường phải là chuỗi hoặc số, không sử dụng ký tự xuống dòng.
- 'persona' phải là một đoạn mô tả văn bản mạch lạc.
- Sử dụng tiếng Trung (ngoại trừ trường 'gender' phải dùng tiếng Anh male/female).
- Nội dung phải nhất quán với thông tin thực thể.
- 'age' phải là số nguyên hợp lệ, 'gender' phải là "male" hoặc "female".
"""

    def _build_group_persona_prompt(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str
    ) -> str:
        """Tạo prompt chi tiết cho tài khoản đại diện tổ chức/nhóm"""
        
        attrs_str = json.dumps(entity_attributes, ensure_ascii=False) if entity_attributes else "None"
        context_str = context[:3000] if context else "No additional context"

#         return f"""Generate a detailed social media account persona for an organization/group entity, recreating existing real-world conditions to the greatest extent possible.
 
# Entity Name: {entity_name}
# Entity Type: {entity_type}
# Entity Summary: {entity_summary}
# Entity Attributes: {attrs_str}
 
# Context Information:
# {context_str}
 
# Please generate a JSON containing the following fields:
 
# 1. bio: Official account biography, 200 characters, professional and appropriate.
# 2. persona: Detailed account setting description (2000 words of plain text), which must include:
#    - Basic information (formal name, nature of the organization, establishment background, primary functions)
#    - Account positioning (account type, target audience, core functions)
#    - Communication style (linguistic characteristics, common expressions, taboo topics)
#    - Content characteristics (content types, posting frequency, active time periods)
#    - Stance and attitude (official stance on core topics, handling of controversies)
#    - Special notes (persona of the group represented, operational habits)
#    - Organizational memory (a vital part of the persona, describing the organization's connection to the event and its existing actions/reactions)
# 3. age: Fixed at 30 (virtual age for an organizational account)
# 4. gender: Fixed as "other" (representing non-individual accounts)
# 5. mbti: MBTI type used to describe the account's style (e.g., ISTJ for rigorous/conservative)
# 6. country: Country (use Chinese, e.g., "中国")
# 7. profession: Description of organizational functions
# 8. interested_topics: An array of focused fields/areas of interest
 
# IMPORTANT:
# - All field values must be strings or numbers; null values are not allowed.
# - 'persona' must be a coherent block of text description; do not use line breaks.
# - Use Chinese (except for the 'gender' field, which must be the English string "other").
# - 'age' must be the integer 30; 'gender' must be the string "other".
# - The account's tone and discourse must strictly align with its institutional identity and positioning.
# """
        
        return f"""Tạo thiết lập tài khoản mạng xã hội chi tiết cho thực thể tổ chức/nhóm, tái hiện tối đa các tình huống thực tế hiện có.

Tên thực thể: {entity_name}
Loại thực thể: {entity_type}
Tóm tắt thực thể: {entity_summary}
Thuộc tính thực thể: {attrs_str}

Thông tin ngữ cảnh:
{context_str}

Vui lòng tạo JSON bao gồm các trường sau:
 
1. bio: Tiểu sử tài khoản chính thức, 200 ký tự, chuyên nghiệp và chuẩn mực.
2. persona: Mô tả chi tiết thiết lập tài khoản (văn bản thuần túy khoảng 2000 từ), cần bao gồm:
   - Thông tin cơ bản về tổ chức (tên chính thức, tính chất tổ chức, bối cảnh thành lập, chức năng chính)
   - Định vị tài khoản (loại tài khoản, đối tượng mục tiêu, chức năng cốt lõi)
   - Phong cách phát ngôn (đặc điểm ngôn ngữ, biểu đạt thường dùng, các chủ đề cấm kỵ)
   - Đặc điểm nội dung đăng tải (loại nội dung, tần suất đăng, khung giờ hoạt động)
   - Lập trường thái độ (quan điểm chính thức về các chủ đề cốt lõi, cách xử lý tranh cãi)
   - Ghi chú đặc biệt (hồ sơ của nhóm mà tổ chức đại diện, thói quen vận hành)
   - Ký ức tổ chức (phần quan trọng của hồ sơ, giới thiệu mối liên hệ của tổ chức này với sự kiện, cũng như các hành động và phản ứng đã có của tổ chức trong sự kiện)
3. age: Cố định là 30 (tuổi ảo cho tài khoản tổ chức)
4. gender: Cố định là "other" (biểu thị tài khoản tổ chức, không phải cá nhân)
5. mbti: Loại MBTI dùng để mô tả phong cách tài khoản (ví dụ: ISTJ đại diện cho sự nghiêm túc, bảo thủ)
6. country: Quốc gia (sử dụng tiếng Trung, ví dụ: "中国")
7. profession: Mô tả chức năng của tổ chức
8. interested_topics: Mảng các lĩnh vực quan tâm
 
QUAN TRỌNG:
- Tất cả giá trị các trường phải là chuỗi hoặc số, không cho phép giá trị null.
- 'persona' phải là một đoạn mô tả văn bản mạch lạc, không sử dụng ký tự xuống dòng.
- Sử dụng tiếng Trung (ngoại trừ trường 'gender' phải dùng chuỗi tiếng Anh "other").
- 'age' phải là số nguyên 30, 'gender' phải là chuỗi "other".
- Phát ngôn và giọng điệu của tài khoản phải phù hợp tuyệt đối với định vị danh tính và đặc thù của tổ chức.
"""
    
    def _generate_profile_rule_based(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sử dụng rule để tạo Profile cơ bản khi dự phòng"""
        
        # Phân nhánh theo loại thực thể để tạo Profile thủ công
        entity_type_lower = entity_type.lower()
        
        if entity_type_lower in ["student", "alumni"]:
            return {
                "bio": f"{entity_type} with interests in academics and social issues.",
                "persona": f"{entity_name} is a {entity_type.lower()} who is actively engaged in academic and social discussions. They enjoy sharing perspectives and connecting with peers.",
                "age": random.randint(18, 30),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(self.MBTI_TYPES),
                "country": random.choice(self.COUNTRIES),
                "profession": "Student",
                "interested_topics": ["Education", "Social Issues", "Technology"],
            }
        
        elif entity_type_lower in ["publicfigure", "expert", "faculty"]:
            return {
                "bio": f"Expert and thought leader in their field.",
                "persona": f"{entity_name} is a recognized {entity_type.lower()} who shares insights and opinions on important matters. They are known for their expertise and influence in public discourse.",
                "age": random.randint(35, 60),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(["ENTJ", "INTJ", "ENTP", "INTP"]),
                "country": random.choice(self.COUNTRIES),
                "profession": entity_attributes.get("occupation", "Expert"),
                "interested_topics": ["Politics", "Economics", "Culture & Society"],
            }
        
        elif entity_type_lower in ["mediaoutlet", "socialmediaplatform"]:
            return {
                "bio": f"Official account for {entity_name}. News and updates.",
                "persona": f"{entity_name} is a media entity that reports news and facilitates public discourse. The account shares timely updates and engages with the audience on current events.",
                "age": 30,  # Tuổi ảo của cơ quan/tổ chức
                "gender": "other",  # Cơ quan dùng "other"
                "mbti": "ISTJ",  # Phong cách tổ chức: nghiêm túc bảo thủ
                "country": "Việt Nam",
                "profession": "Media",
                "interested_topics": ["General News", "Current Events", "Public Affairs"],
            }
        
        elif entity_type_lower in ["university", "governmentagency", "ngo", "organization"]:
            return {
                "bio": f"Official account of {entity_name}.",
                "persona": f"{entity_name} is an institutional entity that communicates official positions, announcements, and engages with stakeholders on relevant matters.",
                "age": 30,  # Tuổi ảo của cơ quan/tổ chức
                "gender": "other",  # Cơ quan dùng "other"
                "mbti": "ISTJ",  # Phong cách tổ chức: nghiêm túc bảo thủ
                "country": "Việt Nam",
                "profession": entity_type,
                "interested_topics": ["Public Policy", "Community", "Official Announcements"],
            }
        
        else:
            # Profile mặc định (Fallback default)
            return {
                "bio": entity_summary[:150] if entity_summary else f"{entity_type}: {entity_name}",
                "persona": entity_summary or f"{entity_name} is a {entity_type.lower()} participating in social discussions.",
                "age": random.randint(25, 50),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(self.MBTI_TYPES),
                "country": random.choice(self.COUNTRIES),
                "profession": entity_type,
                "interested_topics": ["General", "Social Issues"],
            }
    
    def set_graph_id(self, graph_id: str):
        """Lưu lại Graph ID để dùng cho việc tra cứu Zep"""
        self.graph_id = graph_id
    
    def generate_profiles_from_entities(
        self,
        entities: List[EntityNode],
        use_llm: bool = True,
        progress_callback: Optional[callable] = None,
        graph_id: Optional[str] = None,
        parallel_count: int = 5,
        realtime_output_path: Optional[str] = None,
        output_platform: str = "reddit",
        simulation_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> List[OasisAgentProfile]:
        """
        Khởi tạo hàng loạt các Agent Profile từ các thực thể (Hỗ trợ Gen đa luồng song song)
        
        Args:
            entities: Danh sách thực thể
            use_llm: Có sử dụng LLM để tạo tính cách chi tiết hay không
            progress_callback: Hàm CallBack báo tiến độ (current, total, message)
            graph_id: Đưa Graph ID vào để Zep retrieval thêm nhiều ngữ cảnh phong phú
            parallel_count: Số luồng song song, mặc định 5
            realtime_output_path: Đường dẫn lưu file realtime (Gen ra đứa nào auto save đứa đó luôn)
            output_platform: Format lưu trữ output ("reddit" hoạc "twitter")
            
        Returns:
            Danh sách Profile Agent
        """
        import concurrent.futures
        from threading import Lock
        
        # Lưu Graph ID lại cho Zep xử lý search
        if graph_id:
            self.graph_id = graph_id

        self._runtime_metadata = {
            "component": "oasis_profile_generator",
            "phase": "generate_profiles",
            "simulation_id": simulation_id,
            "project_id": project_id,
            "platform": output_platform,
        }
        
        total = len(entities)
        profiles = [None] * total  # Cấp trước 1 mảng để giữ đúng thứ tự Index
        completed_count = [0]  # Phải dùng List để closure của các Sub Thread update được
        lock = Lock()
        
        # Hàm con hỗ trợ việc ghi realtime file trong Thread
        def save_profiles_realtime():
            """Lưu file json ngay lập tức khi profile được tạo mới thành công"""
            if not realtime_output_path:
                return
            
            with lock:
                # Lọc ra những profile đã làm xong
                existing_profiles = [p for p in profiles if p is not None]
                if not existing_profiles:
                    return
                
                try:
                    if output_platform == "reddit":
                        # Cấu trúc dành cho định dạng Reddit
                        profiles_data = [p.to_reddit_format() for p in existing_profiles]
                        with open(realtime_output_path, 'w', encoding='utf-8') as f:
                            json.dump(profiles_data, f, ensure_ascii=False, indent=2)
                    else:
                        # Cấu trúc dành cho định dạng Twitter (CSV)
                        import csv
                        profiles_data = [p.to_twitter_format() for p in existing_profiles]
                        if profiles_data:
                            fieldnames = list(profiles_data[0].keys())
                            with open(realtime_output_path, 'w', encoding='utf-8', newline='') as f:
                                writer = csv.DictWriter(f, fieldnames=fieldnames)
                                writer.writeheader()
                                writer.writerows(profiles_data)
                except Exception as e:
                    logger.warning(f"Failed to save profile in realtime: {e}")
        
        def generate_single_profile(idx: int, entity: EntityNode) -> tuple:
            """Hàm Worker gen từng profile riêng lẻ"""
            entity_type = entity.get_entity_type() or "Entity"
            
            try:
                profile = self.generate_profile_from_entity(
                    entity=entity,
                    user_id=idx,
                    use_llm=use_llm
                )
                
                # Print output để nhìn trực tiếp Log terminal
                self._print_generated_profile(entity.name, entity_type, profile)
                
                return idx, profile, None
                
            except Exception as e:
                logger.error(f"Failed to generate profile for entity {entity.name}: {str(e)}")
                # Rơi vào tạo Profile dự phòng (Fallback)
                fallback_profile = OasisAgentProfile(
                    user_id=idx,
                    user_name=self._generate_username(entity.name),
                    name=entity.name,
                    bio=f"{entity_type}: {entity.name}",
                    persona=entity.summary or f"A participant in social discussions.",
                    source_entity_uuid=entity.uuid,
                    source_entity_type=entity_type,
                )
                return idx, fallback_profile, str(e)
        
        logger.info(f"Start parallel profile generation for {total} entities (Concurrency: {parallel_count})...")
        print(f"\n{'='*60}")
        print(f"Starting Agent Profile Generation - Total {total} entities, concurrency: {parallel_count}")
        print(f"{'='*60}\n")
        
        # Chạy đa luồng thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_count) as executor:
            # Giao Task
            future_to_entity = {
                executor.submit(generate_single_profile, idx, entity): (idx, entity)
                for idx, entity in enumerate(entities)
            }
            
            # Thu gom kết quả
            for future in concurrent.futures.as_completed(future_to_entity):
                idx, entity = future_to_entity[future]
                entity_type = entity.get_entity_type() or "Entity"
                
                try:
                    result_idx, profile, error = future.result()
                    profiles[result_idx] = profile
                    
                    with lock:
                        completed_count[0] += 1
                        current = completed_count[0]
                    
                    # Ghi file Realtime
                    save_profiles_realtime()
                    
                    if progress_callback:
                        progress_callback(
                            current, 
                            total, 
                            f"Completed {current}/{total}: {entity.name} ({entity_type})"
                        )
                    
                    if error:
                        logger.warning(f"[{current}/{total}] Entity {entity.name} applied fallback profile due to error: {error}")
                    else:
                        logger.info(f"[{current}/{total}] Automatically generated profile for: {entity.name} ({entity_type})")
                        
                except Exception as e:
                    logger.error(f"Error handling profile for entity {entity.name}: {str(e)}")
                    with lock:
                        completed_count[0] += 1
                    profiles[idx] = OasisAgentProfile(
                        user_id=idx,
                        user_name=self._generate_username(entity.name),
                        name=entity.name,
                        bio=f"{entity_type}: {entity.name}",
                        persona=entity.summary or "A participant in social discussions.",
                        source_entity_uuid=entity.uuid,
                        source_entity_type=entity_type,
                    )
                    # Ghi file Realtime file (Dù là profile xài fallback)
                    save_profiles_realtime()
        
        print(f"\n{'='*60}")
        print(f"Profile generation complete! Successfully created {len([p for p in profiles if p])} Agents")
        print(f"{'='*60}\n")
        
        return profiles
    
    def _print_generated_profile(self, entity_name: str, entity_type: str, profile: OasisAgentProfile):
        """Xuất thông tin Profile vưa gen ra Terminal để review dễ dàng (Kéo dài không bị gãy log)"""
        separator = "-" * 70
        
        # Xây cấu trúc Log
        topics_str = ', '.join(profile.interested_topics) if profile.interested_topics else 'Không có'
        
        output_lines = [
            f"\n{separator}",
            f"[Generated] {entity_name} ({entity_type})",
            f"{separator}",
            f"Tên tài khoản (Username): {profile.user_name}",
            f"",
            f"【Tiểu sử / Bio】",
            f"{profile.bio}",
            f"",
            f"【Nhân cách cụ thể / Persona】",
            f"{profile.persona}",
            f"",
            f"【Thuộc tính cơ bản / Attributes】",
            f"Tuổi: {profile.age} | Giới tính: {profile.gender} | MBTI: {profile.mbti}",
            f"Nghề nghiệp: {profile.profession} | Quốc gia: {profile.country}",
            f"Chủ đề quan tâm: {topics_str}",
            separator
        ]
        
        output = "\n".join(output_lines)
        
        # Chỉ in ra Console bằng lệnh print (Logger sẽ làm rối và có thể bị truncate)
        print(output)
    
    def save_profiles(
        self,
        profiles: List[OasisAgentProfile],
        file_path: str,
        platform: str = "reddit"
    ):
        """
        Ghi file Profile xuống thư mục (Cấu trúc file tuỳ thuộc vào nền tảng)
        
        Định dạng mặc định của Framework OASIS yêu cầu:
        - Twitter: Định dạng file CSV
        - Reddit: Định dạng file JSON
        
        Args:
            profiles: Danh sách Profile
            file_path: Đường dẫn lưu file
            platform: Tên nền tảng ("reddit" hoặc "twitter")
        """
        if platform == "twitter":
            self._save_twitter_csv(profiles, file_path)
        else:
            self._save_reddit_json(profiles, file_path)
    
    def _save_twitter_csv(self, profiles: List[OasisAgentProfile], file_path: str):
        """
        Lưu Profile hệ Twitter ở định dạng CSV (Bám vào yêu cầu kỹ thuật do OASIS ban hành)
        
        Các trường bắt buộc để tương thích OASIS Twitter File CSV:
        - user_id: Mã định danh ID (Từ 0 theo Index mảng)
        - name: Tên thật của Agent đó
        - username: Tên Alias/Tài khoản xài trong hệ thống
        - user_char: Bản nháp Setting cụ thể truyền vào System Prompt LLM, định hình mọi ý nghĩ/phát ngôn
        - description: Bản Bio gắn ngoài hiển thị cho các User khác thấy (Ngắn gọn)
        
        Sự khác biệt user_char và description:
        - user_char: Data nội bộ chỉ Gen AI thấy (Giống prompt điều khiển não)
        - description: Public Info đưa lên trang cá nhân
        """
        import csv
        
        # Check đuôi file có nhầm thành json không
        if not file_path.endswith('.csv'):
            file_path = file_path.replace('.json', '.csv')
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Khởi tạo Header theo chuẩn file OASIS
            headers = ['user_id', 'name', 'username', 'user_char', 'description']
            writer.writerow(headers)
            
            # Xuất từng dòng Dữ liệu Profile
            for idx, profile in enumerate(profiles):
                # user_char: Nhân cách tổng (bio + persona) - Thả cho Prompt System LLM
                user_char = profile.bio
                if profile.persona and profile.persona != profile.bio:
                    user_char = f"{profile.bio} {profile.persona}"
                # Làm sạch dấu newline để nhét vào dòng CSV
                user_char = user_char.replace('\n', ' ').replace('\r', ' ')
                
                # description: Thông tin Bio hiển thị công khai mạng xã hội
                description = profile.bio.replace('\n', ' ').replace('\r', ' ')
                
                row = [
                    idx,                    # user_id: ID bắt đầu từ 0
                    profile.name,           # name: Tên thực
                    profile.user_name,      # username: Tên định danh
                    user_char,              # user_char: Mô tả ẩn của Bot
                    description             # description: Bảng mô tả Công khai
                ]
                writer.writerow(row)
        
        logger.info(f"Saved {len(profiles)} Twitter Profiles to {file_path} (OASIS CSV Format)")
    
    def _normalize_gender(self, gender: Optional[str]) -> str:
        """
        Biên dịch, chuẩn hóa cột Gender về đúng dạng mà OASIS engine chấp nhận
        
        OASIS quy định buộc xài enum: male, female, other
        """
        if not gender:
            return "other"
        
        gender_lower = gender.lower().strip()
        
        # Mapping các Keyword
        gender_map = {
            "男": "male",
            "女": "female",
            "机构": "other",
            "其他": "other",
            "nam": "male",
            "nữ": "female",
            "tổ chức": "other",
            # Giữ nguyên Tiếng Anh Default
            "male": "male",
            "female": "female",
            "other": "other",
        }
        
        return gender_map.get(gender_lower, "other")
    
    def _save_reddit_json(self, profiles: List[OasisAgentProfile], file_path: str):
        """
        Lưu Profile hệ Reddit bằng JSON (Bám vào yêu cầu kỹ thuật do OASIS ban hành)
        
        Format cấu trúc dựa tương đồng với hàm to_reddit_format().
        Luôn luôn phải có thuộc tính user_id, KEY QUAN TRỌNG ĐỂ HỖ TRỢ HÀM agent_graph.get_agent() MAP CÁC PROFILE !!!
        
        Các field bắt buộc:
        - user_id: User ID dạng Int
        - username: ID Account
        - name: Tên hiển thị
        - bio: Thông tin hiển thị Bio cá nhân
        - persona: Prompt Settings điều khiển Bot nội bộ
        - age: Tuổi (Int)
        - gender: "male", "female", hoặc "other"
        - mbti: Kiểu loại nhóm MBTI
        - country: Quốc gia Country
        """
        data = []
        for idx, profile in enumerate(profiles):
            # Parse Format chung với hàm class to_reddit_format()
            item = {
                "user_id": profile.user_id if profile.user_id is not None else idx,  # Quan trọng: Bắt buộc kèm "user_id"
                "username": profile.user_name,
                "name": profile.name,
                "bio": profile.bio[:150] if profile.bio else f"{profile.name}",
                "persona": profile.persona or f"{profile.name} is a participant in social discussions.",
                "karma": profile.karma if profile.karma else 1000,
                "created_at": profile.created_at,
                # Fix bù tham số ảo cho các properties bị trống
                "age": profile.age if profile.age else 30,
                "gender": self._normalize_gender(profile.gender),
                "mbti": profile.mbti if profile.mbti else "ISTJ",
                "country": profile.country if profile.country else "Việt Nam",
            }
            
            # Cột Tuỳ chọn
            if profile.profession:
                item["profession"] = profile.profession
            if profile.interested_topics:
                item["interested_topics"] = profile.interested_topics
            
            data.append(item)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(profiles)} Reddit Profiles to {file_path} (JSON Config File - with user_id mapped)")
    
    # Giữ lại Function name cũ để hệ thống vẫn tương thích backward.
    def save_profiles_to_json(
        self,
        profiles: List[OasisAgentProfile],
        file_path: str,
        platform: str = "reddit"
    ):
        """[Deprecated - Hết hạn dùng] KHUYÊN DÙNG LỆNH save_profiles() THAY VÌ PHƯƠNG THỨC NÀY"""
        logger.warning("save_profiles_to_json is Deprecated. Use save_profiles method instead!")
        self.save_profiles(profiles, file_path, platform)

