"""
Dịch vụ tạo Ontology (Hệ thực thể / Quan hệ)
API 1: Phân tích nội dung văn bản, khởi tạo các định nghĩa về loại thực thể và quan hệ phù hợp cho việc mô phỏng mạng xã hội
"""

import json
from typing import Dict, Any, List, Optional
from ..utils.llm_client import LLMClient


# System prompt dùng cho việc tự động sinh Ontology
ONTOLOGY_SYSTEM_PROMPT = """Bạn là một chuyên gia thiết kế bản thể học (Ontology) cho Tri thức đồ thị (Knowledge Graph). Nhiệm vụ của bạn là phân tích nội dung văn bản được cung cấp và nhu cầu để thiết kế các loại thực thể (Entity) và loại mối quan hệ (Relationship) thiết kế phù hợp cho **Mô phỏng dư luận trên mạng xã hội**.

**QUAN TRỌNG: Bạn BẮT BUỘC phải đầu ra một cấu trúc định dạng JSON hợp lệ, KHÔNG ĐƯỢC xuất thêm bất kỳ văn bản nào khác.**

## Bối cảnh nhiệm vụ cốt lõi

Chúng tôi đang xây dựng một **hệ thống mô phỏng tin đồn và dư luận mạng xã hội**. Trong hệ thống này:
- Mỗi thực thể là một "tài khoản" hoặc "chủ thể" có thể lên tiếng, tương tác và lan truyền thông tin trên mạng xã hội.
- Các thực thể có thể gây ảnh hưởng, chuyển tiếp (retweet), bình luận hoặc phản hồi lẫn nhau.
- Chúng tôi cần mô phỏng phản ứng của các bên và đường truyền thông tin trong các sự kiện dư luận.

Do đó, **thực thể phải là các chủ thể có thật trong thế giới thực, có khả năng lên tiếng và tương tác trên mạng xã hội**:

**CÓ THỂ LÀ**:
- Cá nhân cụ thể (nhân vật của công chúng, các bên liên quan, KOL, chuyên gia / học giả, người bình thường)
- Công ty, doanh nghiệp (bao gồm cả tài khoản chính thức của họ)
- Tổ chức (trường đại học, hiệp hội, tổ chức phi chính phủ (NGO), công đoàn, v.v.)
- Các cơ quan chính phủ, cơ quan quản lý
- Tổ chức báo chí / truyền thông (báo đài, đài truyền hình, tự do truyền thông, trang web)
- Bản thân nền tảng mạng xã hội
- Đại diện nhóm cụ thể (như hội cựu sinh viên, fan group, nhóm bảo vệ quyền lợi, v.v.)

**KHÔNG ĐƯỢC LÀ**:
- Khái niệm trừu tượng (như "dư luận", "cảm xúc", "xu hướng")
- Chủ đề / đề tài (như "tính toàn vẹn học thuật", "cải cách giáo dục")
- Quan điểm / thái độ (như "phe ủng hộ", "bên phản đối")

## Định dạng đầu ra

Hãy trả về dưới định dạng JSON, bao gồm cấu trúc sau:

```json
{
    "entity_types": [
        {
            "name": "Tên loại thực thể (Tiếng Anh, PascalCase)",
            "description": "Mô tả ngắn gọn (Tiếng Anh, tối đa 100 ký tự)",
            "attributes": [
                {
                    "name": "Tên thuộc tính (Tiếng Anh, snake_case)",
                    "type": "text",
                    "description": "Mô tả của thuộc tính"
                }
            ],
            "examples": ["Ví dụ thực thể 1", "Ví dụ thực thể 2"]
        }
    ],
    "edge_types": [
        {
            "name": "Tên loại quan hệ (Tiếng Anh, UPPER_SNAKE_CASE)",
            "description": "Mô tả ngắn (Tiếng Anh, tối đa 100 ký tự)",
            "source_targets": [
                {"source": "Loại thực thể nguồn", "target": "Loại thực thể đích"}
            ],
            "attributes": []
        }
    ],
    "analysis_summary": "Giải thích ngắn gọn phân tích của bạn về văn bản (Tiếng Việt)"
}
```

## Hướng dẫn Thiết kế (CỰC KỲ QUAN TRỌNG!)

### 1. Thiết kế loại Thực thể (Entity Types) - Phải tuân thủ nghiêm ngặt

**Yêu cầu số lượng: Đúng 10 loại Thực thể.**

**Yêu cầu về cấu trúc phân cấp (Phải có cả Loại cụ thể và Loại bao quát/fallback):**

10 loại thực thể của bạn phải bao gồm cấp độ sau:

A. **Loại bao quát (Fallback Types) (BẮT BUỘC, phải nằm ở 2 vị trí cuối cùng trong mảng)**:
   - `Person`: Là loại bao quát cho MỌI cá nhân tự nhiên. Nếu một người không thuộc các loại cụ thể ở trên, người đó sẽ thuộc `Person`.
   - `Organization`: Là loại bao quát cho MỌI tổ chức. Đặc trưng cho các tổ chức nhỏ hoặc không phù hợp với các loại tổ chức cụ thể khác.

B. **Loại cụ thể (8 loại, phụ thuộc vào nội dung văn bản)**:
   - Thiết kế các loại cụ thể cho các vai chính được nhắc đến nhiều nhất trong văn bản.
   - Ví dụ: Nếu văn bản nói về scandal trường học, có thể có: `Student`, `Professor`, `University`
   - Ví dụ: Nếu văn bản là câu chuyện kinh doanh, có thể có: `Company`, `CEO`, `Employee`

**Tại sao cần các loại Bao quát (Fallback):**
- Văn bản thường chứa các thông tin như "giáo viên tiểu học", "một người qua đường", "một cư dân mạng"
- Nếu không có loại được định nghĩa riêng cho họ, họ nên thuộc về loại `Person`
- Tương tự, tổ chức nhỏ bé hoặc nhóm học tập tạm thời nên thuộc `Organization`

**Nguyên tắc cho các loại Cụ thể:**
- Nhận dạng tần suất xuất hiện và sức ảnh hưởng tới cốt truyện để xây dựng loại thực thể.
- Mỗi loại nên có một ranh giới rõ ràng, không bị chồng chéo.
- Thuộc tính mô tả (description) phải giải thích vì sao loại này tách biệt. 

### 2. Thiết kế Cạnh/Quan hệ (Edge Types)

- Số lượng: Khoảng 6-10 loại quan hệ
- Các mối quan hệ này phải giải thích và gắn kết được hành vi tương tác trên mạng xã hội của các nhân vật.
- Đảm bảo mapping quan hệ hai chiều `source_targets` khớp với các thực thể phía trên.

### 3. Thiết kế Thuộc tính (Attributes)

- Mỗi loại thực thể cần 1-3 thuộc tính chính để làm rõ nhân thân.
- **CHÚ Ý**: Không sử dụng các ID nội bộ làm thuộc tính như `name`, `uuid`, `group_id`, `created_at`, `summary` (chúng là từ khóa hệ thống).
- Khuyên dùng: `full_name`, `title`, `role`, `position`, `location`, `description`,...

## Loại Thực thể tham khảo 

**Loại cá nhân (Cụ thể):**
- Student: Học sinh/Sinh viên
- Professor: Giáo sư/Học giả
- Journalist: Nhà báo/Phóng viên
- Celebrity: Người nổi tiếng/Idol
- Executive: Các giám đốc, CEO, cấp lãnh đạo
- Official: Các vị công chức chính phủ
- Lawyer: Luật sư
- Doctor: Y sĩ/Bác sĩ

**Loại cá nhân (Bao quát):**
- Person: Là loại bao quát cho MỌI cá nhân tự nhiên nào không thuộc chi tiết ở trên.

**Loại tổ chức (Cụ thể):**
- University: Đại học hoặc học viện
- Company: Doanh nghiệp hay Công ty, tập đoàn
- GovernmentAgency: Cơ quan quản lý, các cơ quan ban ngành công quyền
- MediaOutlet: Truyền thông hay Tạp chí, Đài tin tức
- Hospital: Bệnh viện / Trung tâm y tế
- School: Bậc tiểu/trung học
- NGO: Các loại Tổ chức phi chính phủ hoặc từ thiện

**Loại tổ chức (Bao quát):**
- Organization: Là loại bao quát cho MỌI cơ cấu hợp tác không thuôc chi tiết tổ chức ở trên.

## Loại Khái niệm Liên kết (Quan Hệ)

- WORKS_FOR: Làm việc và ăn lương bởi tổ chức
- STUDIES_AT: Đang học tại nhà trường
- AFFILIATED_WITH: Liên quan, Trực thuộc vào đơn vị
- REPRESENTS: Thể hiện tư cách hành động đại diện cho tập thể
- REGULATES: Theo dõi, quản lý, thanh tra chính sách
- REPORTS_ON: Tác nghiệp báo chí, có tin về hiện tượng 
- COMMENTS_ON: Có phản hồi hoặc lên tiếng về tranh cãi
- RESPONDS_TO: Hành động đáp trả
- SUPPORTS: Theo phe ủng hộ điều luật
- OPPOSES: Phản đối chính sách
- COLLABORATES_WITH: Tham gia phối ứng xử lý sự cố. 
- COMPETES_WITH: Quan hệ thù địch.
"""


class OntologyGenerator:
    """
    Trình khởi tạo Ontology
    Phân tích nội dung đoạn văn bản truyền vào, sau đó tự động suy luận ra định nghĩa của các loại Thực thể và Mối quan hệ
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
    
    def generate(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Khởi tạo định nghĩa Ontology
        
        Args:
            document_texts: Danh sách mảng các văn bản nội dung nguồn
            simulation_requirement: Chuỗi mô tả nhu cầu mô phỏng của người dùng
            additional_context: Văn bản cung cấp thêm các ngữ cảnh phụ (nếu có)
            
        Returns:
            Dictionary gồm cấu trúc Ontology (entity_types, edge_types v.v.)
        """
        # Tạo câu lệnh Prompt gửi cho mô hình LLM
        user_message = self._build_user_message(
            document_texts, 
            simulation_requirement,
            additional_context
        )
        
        messages = [
            {"role": "system", "content": ONTOLOGY_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        
        # Gửi request đến LLM
        result = self.llm_client.chat_json(
            messages=messages,
            temperature=0.3,        
        )
        
        # Kiểm tra tính hợp lệ và xử lý tinh chỉnh kết quả đầu ra
        result = self._validate_and_process(result)
        
        return result
    
    # Định mức giới hạn độ dài ký tự tối đa của đoạn văn bản có thể gửi cho LLM (5 vạn chữ)
    MAX_TEXT_LENGTH_FOR_LLM = 80000
    
    def _build_user_message(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str]
    ) -> str:
        """Ghép các thông tin đầu vào thành User Prompt hoàn chỉnh để gửi tới LLM"""
        
        # Gộp tất cả các đoạn văn bản thành một string duy nhất
        combined_text = "\n\n---\n\n".join(document_texts)
        original_length = len(combined_text)
        
        # Nếu vượt quá giới hạn tối đa, thực hiện cắt bớt (Việc này chỉ ảnh hưởng prompt gửi nhận diện Ontology, không ảnh hưởng thư viện Graph building ở sau)
        if len(combined_text) > self.MAX_TEXT_LENGTH_FOR_LLM:
            combined_text = combined_text[:self.MAX_TEXT_LENGTH_FOR_LLM]
            combined_text += f"\n\n...(Văn bản gốc dài {original_length} chữ, đã chủ động cắt lấy {self.MAX_TEXT_LENGTH_FOR_LLM} chữ đầu tiên để phục vụ phân tích Ontology)..."
        
        message = f"""## Nhu cầu mô phỏng

{simulation_requirement}

## Nội dung tài liệu

{combined_text}
"""
        
        if additional_context:
            message += f"""
## Giải thích bổ sung

{additional_context}
"""
        
        message += """
Dựa vào các thông tin trên đây, hãy thiết kế các loại mô hình Thực Thể và Quan Hệ phù hợp để phục vụ việc mô phỏng dư luận trên mạng xã hội.

**Các quy tắc BẮT BUỘC tuân thủ**:
1. Số lượng chính xác: Xuất phải CHUẨN XÁC 10 loại Thực thể
2. 2 vị trí cuối cùng bắt buộc là từ Khoá phụ (Fallback): Person (Cho cá nhân) và Organization (Cho Tổ chức)
3. 8 vị trí đầu tiên phải phân tích và suy luận dựa vào cấu trúc của chính văn bản truyền vào
4. Tất cả các thực thể được liệt kê phải đóng vai trò là Chủ thể (nhân vật có thể lên tiếng ngoài đời thực), KHÔNG ĐƯỢC dùng làm khái niệm trừu tượng.
5. Tên biến thuộc tính KHÔNG ĐƯỢC là name, uuid, group_id hay các biến số bảo lưu của hệ thống khác. Vui lòng chuyển thành full_name, org_name, v.v.
"""
        
        return message
    
    def _validate_and_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Tiền kiểm tra tính trọn vẹn và cấu trúc của dữ liệu phản hồi JSON"""
        
        # Đảm bảo các thuộc tính mảng bắt buộc phải xuất hiện
        if "entity_types" not in result:
            result["entity_types"] = []
        if "edge_types" not in result:
            result["edge_types"] = []
        if "analysis_summary" not in result:
            result["analysis_summary"] = ""
        
        # Tiền xử lý để loại thực thể hợp lệ
        for entity in result["entity_types"]:
            if "attributes" not in entity:
                entity["attributes"] = []
            if "examples" not in entity:
                entity["examples"] = []
            # Cắt ngắn description nếu vượt quá độ dài tối đa 100 character
            if len(entity.get("description", "")) > 100:
                entity["description"] = entity["description"][:97] + "..."
        
        # Tiền xử lý để loại quan hệ hợp lệ
        for edge in result["edge_types"]:
            if "source_targets" not in edge:
                edge["source_targets"] = []
            if "attributes" not in edge:
                edge["attributes"] = []
            if len(edge.get("description", "")) > 100:
                edge["description"] = edge["description"][:97] + "..."
        
        # Ràng buộc số lượng đầu ra của API Zep: Tối đa 10 loại thực thể tự tuỳ chỉnh, và Tối đa 10 loại cạnh quan hệ tùy chỉnh
        MAX_ENTITY_TYPES = 10
        MAX_EDGE_TYPES = 10
        
        # Khai báo định nghĩa về 2 đối tượng bao quát (fallback) mặc định
        person_fallback = {
            "name": "Person",
            "description": "Any individual person not fitting other specific person types.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Full name of the person"},
                {"name": "role", "type": "text", "description": "Role or occupation"}
            ],
            "examples": ["ordinary citizen", "anonymous netizen"]
        }
        
        organization_fallback = {
            "name": "Organization",
            "description": "Any organization not fitting other specific organization types.",
            "attributes": [
                {"name": "org_name", "type": "text", "description": "Name of the organization"},
                {"name": "org_type", "type": "text", "description": "Type of organization"}
            ],
            "examples": ["small business", "community group"]
        }
        
        # Sàng lọc kiểm tra xem kết quả đầu ra đã chứa sẵn các danh mục rỗng (fallback) ở vị trí chuẩn chưa
        entity_names = {e["name"] for e in result["entity_types"]}
        has_person = "Person" in entity_names
        has_organization = "Organization" in entity_names
        
        # Danh sách cần phải gán bù vào
        fallbacks_to_add = []
        if not has_person:
            fallbacks_to_add.append(person_fallback)
        if not has_organization:
            fallbacks_to_add.append(organization_fallback)
        
        if fallbacks_to_add:
            current_count = len(result["entity_types"])
            needed_slots = len(fallbacks_to_add)
            
            # Nếu thêm vào bị quá giới hạn 10 loại, cần phải loại bỏ bớt các loại Entity đằng trước
            if current_count + needed_slots > MAX_ENTITY_TYPES:
                # Tính lượng bị thừa ra so với hạn mức (Để bỏ đi)
                to_remove = current_count + needed_slots - MAX_ENTITY_TYPES
                # Bỏ bớt n vị trí tính từ cuối mảng (Đảm bảo chừa lại nhóm các thực thể cụ thể đã phân tích ở trên)
                result["entity_types"] = result["entity_types"][:-to_remove]
            
            # Nối cụm Fallbacks vừa khởi tạo vào cuối chuỗi
            result["entity_types"].extend(fallbacks_to_add)
        
        # Đảm bảo phòng thủ một lần cuối cùng không có quá 10 Element Array
        if len(result["entity_types"]) > MAX_ENTITY_TYPES:
            result["entity_types"] = result["entity_types"][:MAX_ENTITY_TYPES]
        
        if len(result["edge_types"]) > MAX_EDGE_TYPES:
            result["edge_types"] = result["edge_types"][:MAX_EDGE_TYPES]
        
        return result
    
    def generate_python_code(self, ontology: Dict[str, Any]) -> str:
        """
        Dựng (Generate) file Script với nội dung Python Class tương ứng khai báo dữ liệu Ontology để máy đọc (Tương tự như file ontology.py)
        
        Args:
            ontology: Từ điển định nghĩa Ontology
            
        Returns:
            Chuỗi đoạn code File Python cần tạo để lưu
        """
        code_lines = [
            '"""',
            'Các loại đối tượng (Thực thể) tuỳ chỉnh',
            'Được khởi tạo tự động bởi công cụ MiroFish, ứng dụng vào việc chạy giả lập diễn biến dư luận',
            '"""',
            '',
            'from pydantic import Field',
            'from zep_cloud.external_clients.ontology import EntityModel, EntityText, EdgeModel',
            '',
            '',
            '# ============== Định nghĩa Tên Lớp Các thực thể (Entity) ==============',
            '',
        ]
        
        # Khởi tạo các đoạn mã tương ứng với Định nghĩa thực thể Entity
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            desc = entity.get("description", f"A {name} entity.")
            
            code_lines.append(f'class {name}(EntityModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = entity.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        
        code_lines.append('# ============== Định nghĩa Các Nhóm Quan Hệ/Hành Vi (Edge) ==============')
        code_lines.append('')
        
        # Khởi tạo các đoạn mã tạo lập Relationship (Edges)
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            # Đổi cấu trúc tên format Class theo chuẩn PascalCase của Python
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            desc = edge.get("description", f"A {name} relationship.")
            
            code_lines.append(f'class {class_name}(EdgeModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = edge.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        
        # Tự động kết xuất ra dictionary mapping từ Tên Loại - sang class Object
        code_lines.append('# ============== Các tuỳ chỉnh Map Cấu Hình ==============')
        code_lines.append('')
        code_lines.append('ENTITY_TYPES = {')
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            code_lines.append(f'    "{name}": {name},')
        code_lines.append('}')
        code_lines.append('')
        code_lines.append('EDGE_TYPES = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            code_lines.append(f'    "{name}": {class_name},')
        code_lines.append('}')
        code_lines.append('')
        
        # Cấu hình mảng mapping giới hạn Source->Target cho từng cạnh (Edges source_targets config)
        code_lines.append('EDGE_SOURCE_TARGETS = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            source_targets = edge.get("source_targets", [])
            if source_targets:
                st_list = ', '.join([
                    f'{{"source": "{st.get("source", "Entity")}", "target": "{st.get("target", "Entity")}"}}'
                    for st in source_targets
                ])
                code_lines.append(f'    "{name}": [{st_list}],')
        code_lines.append('}')
        
        return '\n'.join(code_lines)

