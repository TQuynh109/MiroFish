"""
Dịch vụ đọc và lọc thực thể Zep
Đọc các node từ đồ thị Zep, lọc ra các node phù hợp với các loại thực thể đã được định nghĩa trước
"""

import time
from typing import Dict, Any, List, Optional, Set, Callable, TypeVar
from dataclasses import dataclass, field

from zep_cloud.client import Zep

from ..config import Config
from ..utils.logger import get_logger
from ..utils.zep_paging import fetch_all_nodes, fetch_all_edges

logger = get_logger('mirofish.zep_entity_reader')

# Dùng cho các kiểu trả về generic
T = TypeVar('T')


@dataclass
class EntityNode:
    """Cấu trúc dữ liệu của node thực thể"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    # Thông tin edge liên quan
    related_edges: List[Dict[str, Any]] = field(default_factory=list)
    # Thông tin các node khác liên quan
    related_nodes: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
            "related_edges": self.related_edges,
            "related_nodes": self.related_nodes,
        }
    
    def get_entity_type(self) -> Optional[str]:
        """Lấy loại thực thể (loại trừ nhãn Entity mặc định)"""
        for label in self.labels:
            if label not in ["Entity", "Node"]:
                return label
        return None


@dataclass
class FilteredEntities:
    """Tập hợp các thực thể sau khi lọc"""
    entities: List[EntityNode]
    entity_types: Set[str]
    total_count: int
    filtered_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "entity_types": list(self.entity_types),
            "total_count": self.total_count,
            "filtered_count": self.filtered_count,
        }


class ZepEntityReader:
    """
    Dịch vụ đọc và lọc thực thể Zep
    
    Chức năng chính:
    1. Đọc toàn bộ các node từ đồ thị Zep
    2. Lọc ra các node phù hợp với các loại thực thể đã được định nghĩa (Các node có Labels không chỉ là Entity)
    3. Lấy ra thông tin edge cũng như các node liên quan đối với từng thực thể
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or Config.ZEP_API_KEY
        if not self.api_key:
            raise ValueError("ZEP_API_KEY is not configured")
        
        self.client = Zep(api_key=self.api_key)
    
    def _call_with_retry(
        self, 
        func: Callable[[], T], 
        operation_name: str,
        max_retries: int = 3,
        initial_delay: float = 2.0
    ) -> T:
        """
        Gọi hàm Zep API có cơ chế thử lại (retry)
        
        Args:
            func: Hàm cần thực thi (lambda không tham số hoặc callable)
            operation_name: Tên thao tác, dùng cho log
            max_retries: Số lần thử lại tối đa (mặc định 3 lần, tức là thử tối đa 3 lần)
            initial_delay: Số giây trì hoãn ban đầu
            
        Returns:
            Kết quả của lệnh gọi API
        """
        last_exception = None
        delay = initial_delay
        
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Zep {operation_name} attempt {attempt + 1} failed: {str(e)[:100]}, "
                        f"retrying in {delay:.1f} seconds..."
                    )
                    time.sleep(delay)
                    delay *= 2  # Lùi bước nhịp mũ (Exponential backoff)
                else:
                    logger.error(f"Zep {operation_name} failed after {max_retries} attempts: {str(e)}")
        
        raise last_exception
    
    def get_all_nodes(self, graph_id: str) -> List[Dict[str, Any]]:
        """
        Lấy toàn bộ các node của đồ thị (có phân trang)

        Args:
            graph_id: ID của đồ thị

        Returns:
            Danh sách node
        """
        logger.info(f"Fetching all nodes for graph {graph_id}...")

        nodes = fetch_all_nodes(self.client, graph_id)

        nodes_data = []
        for node in nodes:
            nodes_data.append({
                "uuid": getattr(node, 'uuid_', None) or getattr(node, 'uuid', ''),
                "name": node.name or "",
                "labels": node.labels or [],
                "summary": node.summary or "",
                "attributes": node.attributes or {},
            })

        logger.info(f"Total {len(nodes_data)} nodes fetched")
        return nodes_data

    def get_all_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        """
        Lấy toàn bộ các edge của đồ thị (có phân trang)

        Args:
            graph_id: ID của đồ thị

        Returns:
            Danh sách edge
        """
        logger.info(f"Fetching all edges for graph {graph_id}...")

        edges = fetch_all_edges(self.client, graph_id)

        edges_data = []
        for edge in edges:
            edges_data.append({
                "uuid": getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', ''),
                "name": edge.name or "",
                "fact": edge.fact or "",
                "source_node_uuid": edge.source_node_uuid,
                "target_node_uuid": edge.target_node_uuid,
                "attributes": edge.attributes or {},
            })

        logger.info(f"Total {len(edges_data)} edges fetched")
        return edges_data
    
    def get_node_edges(self, node_uuid: str) -> List[Dict[str, Any]]:
        """
        Lấy tất cả các edge liên quan của node được chỉ định (có cơ chế thử lại)
        
        Args:
            node_uuid: UUID của node
            
        Returns:
            Danh sách edge
        """
        try:
            # Sử dụng cơ chế thử lại để gọi Zep API
            edges = self._call_with_retry(
                func=lambda: self.client.graph.node.get_entity_edges(node_uuid=node_uuid),
                operation_name=f"Fetch node edges(node={node_uuid[:8]}...)"
            )
            
            edges_data = []
            for edge in edges:
                edges_data.append({
                    "uuid": getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', ''),
                    "name": edge.name or "",
                    "fact": edge.fact or "",
                    "source_node_uuid": edge.source_node_uuid,
                    "target_node_uuid": edge.target_node_uuid,
                    "attributes": edge.attributes or {},
                })
            
            return edges_data
        except Exception as e:
            logger.warning(f"Failed to fetch edges for node {node_uuid}: {str(e)}")
            return []
    
    def filter_defined_entities(
        self, 
        graph_id: str,
        defined_entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True
    ) -> FilteredEntities:
        """
        Lọc ra các node phù hợp với các loại thực thể đã được định nghĩa
        
        Logic lọc:
        - Nếu Labels của node chỉ có một nhãn là "Entity", tức là thực thể này không hợp với loại chúng ta định nghĩa, tiến hành bỏ qua
        - Nếu Labels của node chứa các nhãn khác ngoài "Entity" và "Node", tức là hợp lệ, tiến hành giữ lại
        
        Args:
            graph_id: ID của đồ thị
            defined_entity_types: Danh sách các loại thực thể định nghĩa trước (không bắt buộc, nếu có thì chỉ giữ lại các loại đó)
            enrich_with_edges: Có lấy thông tin edge liên quan của từng thực thể hay không
            
        Returns:
            FilteredEntities: Tập hợp các thực thể sau khi lọc
        """
        logger.info(f"Start filtering entities for graph {graph_id}...")
        
        # Lấy toàn bộ các node
        all_nodes = self.get_all_nodes(graph_id)
        total_count = len(all_nodes)
        
        # Lấy toàn bộ các edge (để lấy liên kết sau này)
        all_edges = self.get_all_edges(graph_id) if enrich_with_edges else []
        
        # Xây dựng map ánh xạ từ UUID của node sang dữ liệu node
        node_map = {n["uuid"]: n for n in all_nodes}
        
        # Lọc các thực thể đáp ứng điều kiện
        filtered_entities = []
        entity_types_found = set()
        
        for node in all_nodes:
            labels = node.get("labels", [])
            
            # Logic lọc: Labels bắt buộc phải chứa các nhãn khác "Entity" và "Node"
            custom_labels = [l for l in labels if l not in ["Entity", "Node"]]
            
            if not custom_labels:
                # Chỉ có nhãn mặc định, bỏ qua
                continue
            
            # Nếu đã chỉ định loại thực thể cho trước, kiểm tra xem có khớp hay không
            if defined_entity_types:
                matching_labels = [l for l in custom_labels if l in defined_entity_types]
                if not matching_labels:
                    continue
                entity_type = matching_labels[0]
            else:
                entity_type = custom_labels[0]
            
            entity_types_found.add(entity_type)
            
            # Tạo object cho node thực thể
            entity = EntityNode(
                uuid=node["uuid"],
                name=node["name"],
                labels=labels,
                summary=node["summary"],
                attributes=node["attributes"],
            )
            
            # Lấy các edge và node liên quan
            if enrich_with_edges:
                related_edges = []
                related_node_uuids = set()
                
                for edge in all_edges:
                    if edge["source_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "outgoing",
                            "edge_name": edge["name"],
                            "fact": edge["fact"],
                            "target_node_uuid": edge["target_node_uuid"],
                        })
                        related_node_uuids.add(edge["target_node_uuid"])
                    elif edge["target_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "incoming",
                            "edge_name": edge["name"],
                            "fact": edge["fact"],
                            "source_node_uuid": edge["source_node_uuid"],
                        })
                        related_node_uuids.add(edge["source_node_uuid"])
                
                entity.related_edges = related_edges
                
                # Lấy thông tin cơ bản của các node được liên kết
                related_nodes = []
                for related_uuid in related_node_uuids:
                    if related_uuid in node_map:
                        related_node = node_map[related_uuid]
                        related_nodes.append({
                            "uuid": related_node["uuid"],
                            "name": related_node["name"],
                            "labels": related_node["labels"],
                            "summary": related_node.get("summary", ""),
                        })
                
                entity.related_nodes = related_nodes
            
            filtered_entities.append(entity)
        
        logger.info(f"Filtering completed: Total nodes {total_count}, Matched {len(filtered_entities)}, "
                   f"Entity types: {entity_types_found}")
        
        return FilteredEntities(
            entities=filtered_entities,
            entity_types=entity_types_found,
            total_count=total_count,
            filtered_count=len(filtered_entities),
        )
    
    def get_entity_with_context(
        self, 
        graph_id: str, 
        entity_uuid: str
    ) -> Optional[EntityNode]:
        """
        Lấy thông tin của một thực thể cụ thể và ngữ cảnh đầy đủ của nó (edge và node liên kết, với cơ chế thử lại)
        
        Args:
            graph_id: ID của đồ thị
            entity_uuid: UUID của thực thể
            
        Returns:
            EntityNode hoặc None
        """
        try:
            # Sử dụng cơ chế thử lại để lấy thông tin node
            node = self._call_with_retry(
                func=lambda: self.client.graph.node.get(uuid_=entity_uuid),
                operation_name=f"Fetch node detail(uuid={entity_uuid[:8]}...)"
            )
            
            if not node:
                return None
            
            # Lấy các edge của node
            edges = self.get_node_edges(entity_uuid)
            
            # Lấy tất cả các node để tìm liên kết
            all_nodes = self.get_all_nodes(graph_id)
            node_map = {n["uuid"]: n for n in all_nodes}
            
            # Xử lý các edge và node liên quan
            related_edges = []
            related_node_uuids = set()
            
            for edge in edges:
                if edge["source_node_uuid"] == entity_uuid:
                    related_edges.append({
                        "direction": "outgoing",
                        "edge_name": edge["name"],
                        "fact": edge["fact"],
                        "target_node_uuid": edge["target_node_uuid"],
                    })
                    related_node_uuids.add(edge["target_node_uuid"])
                else:
                    related_edges.append({
                        "direction": "incoming",
                        "edge_name": edge["name"],
                        "fact": edge["fact"],
                        "source_node_uuid": edge["source_node_uuid"],
                    })
                    related_node_uuids.add(edge["source_node_uuid"])
            
            # Lấy thông tin về node được liên kết
            related_nodes = []
            for related_uuid in related_node_uuids:
                if related_uuid in node_map:
                    related_node = node_map[related_uuid]
                    related_nodes.append({
                        "uuid": related_node["uuid"],
                        "name": related_node["name"],
                        "labels": related_node["labels"],
                        "summary": related_node.get("summary", ""),
                    })
            
            return EntityNode(
                uuid=getattr(node, 'uuid_', None) or getattr(node, 'uuid', ''),
                name=node.name or "",
                labels=node.labels or [],
                summary=node.summary or "",
                attributes=node.attributes or {},
                related_edges=related_edges,
                related_nodes=related_nodes,
            )
            
        except Exception as e:
            logger.error(f"Failed to fetch entity {entity_uuid}: {str(e)}")
            return None
    
    def get_entities_by_type(
        self, 
        graph_id: str, 
        entity_type: str,
        enrich_with_edges: bool = True
    ) -> List[EntityNode]:
        """
        Lấy tất cả các thực thể dựa theo loại cụ thể
        
        Args:
            graph_id: ID của đồ thị
            entity_type: Loại thực thể (ví dụ: "Student", "PublicFigure", v.v..)
            enrich_with_edges: Có lấy thông tin edge liên quan hay không
            
        Returns:
            Danh sách thực thể
        """
        result = self.filter_defined_entities(
            graph_id=graph_id,
            defined_entity_types=[entity_type],
            enrich_with_edges=enrich_with_edges
        )
        return result.entities


