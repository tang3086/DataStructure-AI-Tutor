import os
import numpy as np
from pathlib import Path
from openai import OpenAI
from dataclasses import dataclass

@dataclass
class Chunk:
    content: str
    source: str
    index: int

class RAGLibrary:
    def __init__(
        self, 
        client: OpenAI, 
        embedding_model: str, 
        chunk_size: int = 500, 
        chunk_overlap: int = 100,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
        
        self.client = client
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None

    def load_documents(self, dir_path: str | Path) -> None:
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")
        
        # ====================== 实验手册必做：清空旧数据 ======================
        self.chunks = []
        self.embeddings = None

        # ====================== 实验手册必做：递归遍历所有 txt/md ======================
        for path in dir_path.rglob("*"):
            if path.suffix.lower() not in {".txt", ".md"}:
                continue
            try:
                content = path.read_text(encoding="utf-8")
                for text in self._split_text(content):
                    self.chunks.append(
                        Chunk(
                            content=text,
                            source=path.name,
                            index=len(self.chunks),
                        )
                    )
            except Exception as e:
                print(f"[Warning] Failed to read {path}: {e}")

    def _split_text(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(text), step):
            chunk = text[i:i + self.chunk_size]
            chunks.append(chunk)
        return chunks

    def create_embeddings(self) -> None:
        if not self.chunks:
            raise ValueError("No chunks loaded. Call load_documents() first.")
        
        texts = [chunk.content for chunk in self.chunks]
        vectors = []
        batch_size = 10  # Qwen 限制每次最多10条

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=batch
            )
            vectors.extend([item.embedding for item in response.data])
        
        self.embeddings = np.array(vectors, dtype=np.float32)

    def save_index(self, filepath: str | Path) -> None:
        if self.embeddings is None:
            raise ValueError("No embeddings found. Call create_embeddings() first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            filepath,
            embeddings=self.embeddings,
            chunks_content=np.array([chunk.content for chunk in self.chunks]),
            chunks_source=np.array([chunk.source for chunk in self.chunks]),
            chunks_index=np.array([chunk.index for chunk in self.chunks])
        )

    def load_index(self, filepath: str | Path) -> None:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Index file not found: {filepath}")
        
        data = np.load(filepath)
        self.embeddings = data["embeddings"]
        self.chunks = [
            Chunk(
                content=str(data["chunks_content"][i]),
                source=str(data["chunks_source"][i]),
                index=int(data["chunks_index"][i]),
            )
            for i in range(len(data["chunks_content"]))
        ]

    @staticmethod
    def _cosine_similarity(query: np.ndarray, documents: np.ndarray) -> np.ndarray:
        query_norm = np.linalg.norm(query)
        doc_norms = np.linalg.norm(documents, axis=1)
        dot_products = np.dot(documents, query)
        similarities = dot_products / (doc_norms * query_norm + 1e-8)
        return similarities

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        if self.embeddings is None:
            raise ValueError("No embeddings found. Call create_embeddings() first.")
        if top_k <= 0:
            return []
        
        query_emb = self.client.embeddings.create(
            model=self.embedding_model,
            input=query
        ).data[0].embedding
        query_emb = np.array(query_emb, dtype=np.float32)

        similarities = self._cosine_similarity(query_emb, self.embeddings)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(self.chunks[i], similarities[i]) for i in top_indices]
        return results

    def build_context(self, query: str, top_k: int = 5) -> str:
        results = self.search(query, top_k)
        context = "\n\n".join([chunk.content for chunk, score in results])
        return context

def load_or_build_rag_library(
    client: OpenAI,
    embedding_model: str = "text-embedding-v4",
    docs_dir: str | Path = "library",
    index_path: str | Path = "library/index.npz",
    verbose: bool = True,
) -> RAGLibrary:
    index_path = Path(index_path)
    docs_dir = Path(docs_dir)

    rag = RAGLibrary(client=client, embedding_model=embedding_model)

    if index_path.exists():
        if verbose:
            print(f"[RAG] 加载索引:{index_path}")
        rag.load_index(index_path)
    else:
        if verbose:
            print(f"[RAG] 未找到索引,开始构建:{index_path}")
        rag.load_documents(docs_dir)
        rag.create_embeddings()
        rag.save_index(index_path)
        if verbose:
            print(f"[RAG] 索引构建完成:{index_path}")
    return rag