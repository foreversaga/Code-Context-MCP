from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Parser
from tree_sitter_language_pack import get_language


LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "c_sharp",
}

SEMANTIC_NODES = {
    "python": {"function_definition", "class_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition"},
    "typescript": {"function_declaration", "class_declaration", "method_definition", "interface_declaration"},
    "tsx": {"function_declaration", "class_declaration", "method_definition", "interface_declaration"},
    "java": {"method_declaration", "constructor_declaration", "class_declaration", "interface_declaration"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_item", "struct_item", "trait_item", "impl_item"},
    "c": {"function_definition", "struct_specifier"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier"},
    "c_sharp": {"method_declaration", "constructor_declaration", "class_declaration", "interface_declaration"},
}


@dataclass(frozen=True)
class CodeChunk:
    path: str
    symbol: str | None
    kind: str
    start_line: int
    end_line: int
    content: str


class CodeChunker:
    def __init__(self, fallback_lines: int = 80, overlap: int = 10) -> None:
        self.fallback_lines = fallback_lines
        self.overlap = overlap

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in LANGUAGE_BY_SUFFIX

    def chunk(self, relative_path: str, content: str) -> list[CodeChunk]:
        suffix = Path(relative_path).suffix.lower()
        language_name = LANGUAGE_BY_SUFFIX.get(suffix)
        if not language_name:
            return self._fallback(relative_path, content)

        try:
            parser = Parser(get_language(language_name))
            raw = content.encode("utf-8")
            root = parser.parse(raw).root_node
            chunks: list[CodeChunk] = []

            def visit(node) -> None:
                if node.type in SEMANTIC_NODES.get(language_name, set()):
                    text = raw[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")
                    name_node = node.child_by_field_name("name")
                    symbol = None
                    if name_node is not None:
                        symbol = raw[name_node.start_byte : name_node.end_byte].decode(
                            "utf-8", errors="ignore"
                        )
                    chunks.append(
                        CodeChunk(
                            path=relative_path,
                            symbol=symbol,
                            kind=node.type,
                            start_line=node.start_point.row + 1,
                            end_line=node.end_point.row + 1,
                            content=text,
                        )
                    )
                for child in node.named_children:
                    visit(child)

            visit(root)
            seen: set[tuple[int, int, str]] = set()
            unique: list[CodeChunk] = []
            for chunk in chunks:
                key = (chunk.start_line, chunk.end_line, chunk.kind)
                if key not in seen:
                    seen.add(key)
                    unique.append(chunk)
            return unique or self._fallback(relative_path, content)
        except Exception:
            return self._fallback(relative_path, content)

    def _fallback(self, relative_path: str, content: str) -> list[CodeChunk]:
        lines = content.splitlines()
        if not lines:
            return []
        step = max(1, self.fallback_lines - self.overlap)
        result = []
        for start in range(0, len(lines), step):
            end = min(len(lines), start + self.fallback_lines)
            result.append(
                CodeChunk(
                    path=relative_path,
                    symbol=None,
                    kind="line_chunk",
                    start_line=start + 1,
                    end_line=end,
                    content="\n".join(lines[start:end]),
                )
            )
            if end == len(lines):
                break
        return result
