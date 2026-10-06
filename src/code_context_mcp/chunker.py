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
    "typescript": {
        "function_declaration",
        "class_declaration",
        "method_definition",
        "interface_declaration",
    },
    "tsx": {
        "function_declaration",
        "class_declaration",
        "method_definition",
        "interface_declaration",
    },
    "java": {
        "method_declaration",
        "constructor_declaration",
        "class_declaration",
        "interface_declaration",
    },
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_item", "struct_item", "trait_item", "impl_item"},
    "c": {"function_definition", "struct_specifier"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier"},
    "c_sharp": {
        "method_declaration",
        "constructor_declaration",
        "class_declaration",
        "interface_declaration",
    },
}

CONTAINER_NODES = {
    "class_definition",
    "class_declaration",
    "interface_declaration",
    "type_declaration",
    "struct_item",
    "trait_item",
    "impl_item",
    "class_specifier",
    "struct_specifier",
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
    def __init__(
        self,
        fallback_lines: int = 80,
        overlap: int = 10,
        max_chunk_chars: int = 24_000,
    ) -> None:
        self.fallback_lines = fallback_lines
        self.overlap = overlap
        self.max_chunk_chars = max_chunk_chars

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
                    chunks.extend(
                        self._bounded_chunks(
                            path=relative_path,
                            symbol=symbol,
                            kind=node.type,
                            start_line=node.start_point.row + 1,
                            content=text,
                            is_container=node.type in CONTAINER_NODES,
                        )
                    )
                for child in node.named_children:
                    visit(child)

            visit(root)
            seen: set[tuple[int, int, str, str | None]] = set()
            unique: list[CodeChunk] = []
            for chunk in chunks:
                key = (chunk.start_line, chunk.end_line, chunk.kind, chunk.symbol)
                if key not in seen:
                    seen.add(key)
                    unique.append(chunk)
            return unique or self._fallback(relative_path, content)
        except Exception:
            return self._fallback(relative_path, content)

    def _bounded_chunks(
        self,
        path: str,
        symbol: str | None,
        kind: str,
        start_line: int,
        content: str,
        is_container: bool,
    ) -> list[CodeChunk]:
        if len(content) <= self.max_chunk_chars:
            return [
                CodeChunk(
                    path=path,
                    symbol=symbol,
                    kind=kind,
                    start_line=start_line,
                    end_line=start_line + max(0, content.count("\n")),
                    content=content,
                )
            ]

        # Large classes/interfaces mostly duplicate their child methods. Keep only a bounded
        # declaration/context chunk while child semantic nodes are indexed separately.
        if is_container:
            bounded = content[: self.max_chunk_chars]
            return [
                CodeChunk(
                    path=path,
                    symbol=symbol,
                    kind=kind,
                    start_line=start_line,
                    end_line=start_line + bounded.count("\n"),
                    content=bounded,
                )
            ]

        result: list[CodeChunk] = []
        offset = 0
        while offset < len(content):
            end = min(len(content), offset + self.max_chunk_chars)
            if end < len(content):
                newline = content.rfind("\n", offset, end)
                if newline > offset:
                    end = newline + 1
            part = content[offset:end]
            line_offset = content[:offset].count("\n")
            result.append(
                CodeChunk(
                    path=path,
                    symbol=symbol,
                    kind=f"{kind}_part",
                    start_line=start_line + line_offset,
                    end_line=start_line + line_offset + part.count("\n"),
                    content=part,
                )
            )
            offset = end
        return result

    def _fallback(self, relative_path: str, content: str) -> list[CodeChunk]:
        lines = content.splitlines()
        if not lines:
            return []
        step = max(1, self.fallback_lines - self.overlap)
        result = []
        for start in range(0, len(lines), step):
            end = min(len(lines), start + self.fallback_lines)
            chunk_content = "\n".join(lines[start:end])
            if len(chunk_content) > self.max_chunk_chars:
                chunk_content = chunk_content[: self.max_chunk_chars]
            result.append(
                CodeChunk(
                    path=relative_path,
                    symbol=None,
                    kind="line_chunk",
                    start_line=start + 1,
                    end_line=end,
                    content=chunk_content,
                )
            )
            if end == len(lines):
                break
        return result
