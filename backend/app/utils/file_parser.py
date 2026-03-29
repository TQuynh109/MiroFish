"""
Tiện ích phân tích tệp
Hỗ trợ trích xuất văn bản từ tệp PDF, Markdown, TXT
"""

import os
from pathlib import Path
from typing import List, Optional


def _read_text_with_fallback(file_path: str) -> str:
    """
    Đọc tệp văn bản, tự động phát hiện mã hóa nếu UTF-8 thất bại.

    Áp dụng chiến lược fallback nhiều tầng:
    1. Thử giải mã bằng UTF-8 trước
    2. Dùng charset_normalizer để phát hiện mã hóa
    3. Fallback sang chardet de phat hien ma hoa
    4. Cuối cùng dùng UTF-8 + errors='replace' để đảm bảo không vỡ ký tự

    Args:
        file_path: Đường dẫn tệp

    Returns:
        Nội dung văn bản sau khi giải mã
    """
    data = Path(file_path).read_bytes()
    
    # Thử UTF-8 trước
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        pass
    
    # Thử phát hiện mã hóa bằng charset_normalizer
    encoding = None
    try:
        from charset_normalizer import from_bytes
        best = from_bytes(data).best()
        if best and best.encoding:
            encoding = best.encoding
    except Exception:
        pass
    
    # Fallback sang chardet
    if not encoding:
        try:
            import chardet
            result = chardet.detect(data)
            encoding = result.get('encoding') if result else None
        except Exception:
            pass
    
    # Fallback cuối: UTF-8 + replace
    if not encoding:
        encoding = 'utf-8'
    
    return data.decode(encoding, errors='replace')


class FileParser:
    """Bộ phân tích tệp"""
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.md', '.markdown', '.txt'}
    
    @classmethod
    def extract_text(cls, file_path: str) -> str:
        """
        Trích xuất văn bản từ tệp

        Args:
            file_path: Đường dẫn tệp

        Returns:
            Nội dung văn bản đã trích xuất
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")
        
        suffix = path.suffix.lower()
        
        if suffix not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        # Format Support: Leverages the existing extract_text method which supports PDF, Markdown, and TXT formats 
        if suffix == '.pdf':
            return cls._extract_from_pdf(file_path)
        elif suffix in {'.md', '.markdown'}:
            return cls._extract_from_md(file_path)
        elif suffix == '.txt':
            return cls._extract_from_txt(file_path)
        
        raise ValueError(f"Cannot process file format: {suffix}")
    
    @staticmethod
    def _extract_from_pdf(file_path: str) -> str:
        """Trích xuất văn bản từ PDF"""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF is required: pip install PyMuPDF")
        
        text_parts = []
        with fitz.open(file_path) as doc:
            for page in doc:
                text = page.get_text()
                if text.strip():
                    text_parts.append(text)
        
        return "\n\n".join(text_parts)
    
    @staticmethod
    def _extract_from_md(file_path: str) -> str:
        """Trích xuất văn bản từ Markdown, hỗ trợ tự động phát hiện mã hóa"""
        return _read_text_with_fallback(file_path)
    
    @staticmethod
    def _extract_from_txt(file_path: str) -> str:
        """Trích xuất văn bản từ TXT, hỗ trợ tự động phát hiện mã hóa"""
        return _read_text_with_fallback(file_path)
    
    @classmethod
    def extract_from_multiple(cls, file_paths: List[str]) -> str:
        """
        Usage Context
        This function is typically used in the early stages of the GraphRAG pipeline when users upload multiple documents that need to be processed together for entity extraction and relationship mapping. The combined output serves as input for text chunking and subsequent LLM-based analysis in the knowledge graph construction workflow

        Trích xuất văn bản từ nhiều tệp và gộp lại
        Args:
            file_paths: Danh sách đường dẫn tệp

        Returns:
            Văn bản đã gộp

        The extract_from_multiple method is a class method of the FileParser class that processes multiple document files simultaneously. It's designed to aggregate content from various source files (PDF, Markdown, TXT) into a single text string that can be fed into the knowledge graph construction pipeline.
        """
        all_texts = []
        
        # Batch Processing: Takes a list of file paths and processes each one sequentially
        for i, file_path in enumerate(file_paths, 1):
            try:
                text = cls.extract_text(file_path)
                filename = Path(file_path).name
                all_texts.append(f"=== Document {i}: {filename} ===\n{text}")
            # Error Handling: If a file fails to extract, it includes an error message in the output rather than failing completely
            except Exception as e:
                all_texts.append(f"=== Document {i}: {file_path} (extract failed: {str(e)}) ===")
        
        return "\n\n".join(all_texts)


def split_text_into_chunks(
    text: str, 
    chunk_size: int = 500, 
    overlap: int = 50
) -> List[str]:
    """
    Chia văn bản thành các đoạn nhỏ

    Args:
        text: Văn bản gốc
        chunk_size: Số ký tự mỗi đoạn
        overlap: Số ký tự chồng lấp

    Returns:
        Danh sách các đoạn văn bản
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Cố gắng cắt tại ranh giới câu
        if end < len(text):
            # Tìm dấu kết thúc câu gần nhất
            for sep in ['。', '！', '？', '.\n', '!\n', '?\n', '\n\n', '. ', '! ', '? ']:
                last_sep = text[start:end].rfind(sep)
                if last_sep != -1 and last_sep > chunk_size * 0.3:
                    end = start + last_sep + len(sep)
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Đoạn tiếp theo bắt đầu từ vị trí overlap
        start = end - overlap if end < len(text) else len(text)
    
    return chunks

