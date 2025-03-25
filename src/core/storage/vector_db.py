from pathlib import Path
import faiss
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore


class VectorStore:
    def __init__(self, data_dir="documents", index_dir="faiss_index", embedder=None):
        if embedder is None:
            raise ValueError("Embedder must be provided!")

        self.data_dir = Path(data_dir)
        self.index_dir = Path(index_dir)
        self.documents = []
        self.index = None
        self.vector_store = None
        self.index_lock = FileLock(str(self.index_dir / "index.lock"))
        self.embedder = embedder
        self._init_embedding_settings()

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.watcher = DocumentWatcher(
            data_dir=self.data_dir, update_handler=self.handle_document_update
        )
        self.watcher.start()

        self.index_exists = self._check_index_exists()
        self._load_index()

    def _init_embedding_settings(self):
        """Инициализация настроек эмбеддингов"""
        from llama_index.core import Settings

        Settings.embed_model = self._create_embedding_adapter()

    def handle_document_update(self, file_path: Path):
        """Обработчик обновления документов"""
        print(f"🔄 Обнаружено изменение: {file_path.name}")

        try:
            new_docs = self._load_and_process_file(file_path)
            if not new_docs:
                return

            with self.index_lock:
                self._update_index(new_docs)
                self._atomic_save()
                print(f"✅ Индекс успешно обновлен из {file_path.name}")

        except Exception as e:
            print(f"⚠️ Ошибка обработки документа: {str(e)}")
            self._log_error(file_path, str(e))

    def _load_and_process_file(self, path: Path) -> list:
        """Загрузка и обработка документа"""
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
            items = data if isinstance(data, list) else [data]

            valid_docs = []
            for item in items:
                chunks = self._split_document(item)
                valid_docs.extend(chunks)
            return valid_docs

    def _split_document(self, doc: dict) -> list:
        """Разделение документа на чанки"""
        from llama_index.core.node_parser import SentenceSplitter

        splitter = SentenceSplitter(
            chunk_size=1024, chunk_overlap=128, include_metadata=True
        )

        text = doc["text"]
        metadata = doc["metadata"]

        return [
            {
                "text": chunk,
                "metadata": {
                    **metadata,
                    "chunk_id": f"{metadata['doc_id']}_part_{i+1}",
                },
            }
            for i, chunk in enumerate(splitter.split_text(text))
        ]

    def _update_index(self, new_docs: list):
        """Обновление индекса новыми данными"""
        from llama_index.core import Settings
        from llama_index.core.schema import TextNode

        self._init_embedding_settings()

        nodes = [
            TextNode(
                text=doc["text"],
                metadata=doc["metadata"],
                id_=f"{doc['metadata']['doc_id']}_{i}",
                embedding=self.embedder.embed([doc["text"]])[0].tolist(),
            )
            for i, doc in enumerate(new_docs)
        ]

        if self.index:
            self.index.insert_nodes(nodes)
            self.index.storage_context.persist(persist_dir=str(self.index_dir))
        else:
            self.create_index()
        self.index.storage_context.persist(persist_dir=str(self.index_dir))
        print(f"Embedder status: {'OK' if self.embedder else 'NOT INITIALIZED'}")
        print(f"Embedding test: {self.embedder.embed(['test'])[0][:5]}...")

    def _atomic_save(self):
        """Атомарное сохранение индекса"""
        temp_dir = self.index_dir / f"temp_{int(time.time())}"
        temp_dir.mkdir(exist_ok=True)

        try:
            faiss_index = self.index.storage_context.vector_store.client

            self.index.storage_context.persist(persist_dir=str(temp_dir))

            faiss.write_index(faiss_index, str(temp_dir / "faiss.index"))

            for file in temp_dir.glob("*"):
                dest = self.index_dir / file.name
                if dest.exists():
                    dest.unlink()
                file.replace(dest)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _log_error(self, file_path: Path, error: str):
        """Логирование ошибок"""
        log_entry = f"{datetime.now().isoformat()} | {file_path.name} | {error}\n"
        with open("index_errors.log", "a") as f:
            f.write(log_entry)

    def __del__(self):
        if hasattr(self, "watcher") and self.watcher:
            self.watcher.stop()

    def _create_embedding_adapter(self):
        """Создаем адаптер для эмбеддингов"""
        from llama_index.core.embeddings import BaseEmbedding

        embedder = self.embedder

        class CustomEmbeddingAdapter(BaseEmbedding):

            def _get_text_embedding(self, text: str) -> List[float]:
                return embedder.embed([text])[0].tolist()

            def _get_query_embedding(self, query: str) -> List[float]:
                return self._get_text_embedding(query)

            async def _aget_text_embedding(self, text: str) -> List[float]:
                return self._get_text_embedding(text)

            async def _aget_query_embedding(self, query: str) -> List[float]:
                return self._get_query_embedding(query)

            def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
                return [self._get_text_embedding(text) for text in texts]

        CustomEmbeddingAdapter._outer = self
        return CustomEmbeddingAdapter()

    def _check_index_exists(self) -> bool:
        """Проверяет наличие всех необходимых файлов индекса"""
        required_files = {"docstore.json", "vector_store.json", "faiss.index"}
        existing_files = set(os.listdir(self.index_dir))
        return required_files.issubset(existing_files)

    def _load_index(self):
        """Загрузка индекса с улучшенной обработкой ошибок"""
        if not self.index_exists:
            print("🟡 Индекс не найден, будет создан новый")
            return

        try:
            faiss_index = faiss.read_index(str(self.index_dir / "faiss.index"))
            self.vector_store = FaissVectorStore(faiss_index=faiss_index)

            storage_context = StorageContext.from_defaults(
                persist_dir=str(self.index_dir), vector_store=self.vector_store
            )

            self.index = load_index_from_storage(storage_context)
            print(f"✅ Индекс успешно загружен из {self.index_dir}")

        except Exception as e:
            print(f"⚠️ Ошибка загрузки индекса: {str(e)}")
            self._delete_corrupted_index()
            self.index = None
            self.vector_store = None

    def _delete_corrupted_index(self):
        """Удаление поврежденных файлов индекса"""
        print("🟠 Удаление поврежденного индекса...")
        for file in self.index_dir.glob("*"):
            try:
                file.unlink()
            except Exception as e:
                print(f"⚠️ Не удалось удалить {file.name}: {str(e)}")

    def load_documents(self):
        """Загрузка и разделение документов на чанки"""
        from llama_index.core import Document
        from llama_index.core.node_parser import SentenceSplitter

        self.documents = []
        splitter = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=128,
            separator="\n",
            paragraph_separator="\n\n",
            secondary_chunking_regex=r"(?m)^\d+\.",
            include_metadata=True,
        )

        for file in self.data_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
                    items = data if isinstance(data, list) else [data]

                    for item in items:
                        text = item.get("text", "")
                        metadata = item.get("metadata", {})

                        chunks = splitter.split_text(text)

                        for i, chunk in enumerate(chunks):
                            self.documents.append(
                                Document(
                                    text=chunk,
                                    metadata={
                                        **metadata,
                                        "doc_id": f"{file.stem}_chunk_{i+1}",
                                        "original_length": len(text),
                                    },
                                )
                            )

            except Exception as e:
                print(f"Ошибка загрузки {file.name}: {str(e)}")
                continue

        print(f"Загружено чанков: {len(self.documents)}")
        if self.documents:
            print("\nПример загруженного чанка:")
            print(f"Текст: {self.documents[0].text}...")
            print(f"Метаданные: {self.documents[0].metadata}\n")

    def create_index(self):
        """Создание индекса с явным указанием локальных эмбеддингов"""
        from llama_index.core import Settings

        Settings.embed_model = self._create_embedding_adapter()

        test_embed = self.embedder.embed(["test"])
        embedding_dim = test_embed.shape[1]
        print(f"Размерность эмбеддингов: {embedding_dim}")

        faiss_index = faiss.IndexFlatL2(embedding_dim)
        self.vector_store = FaissVectorStore(faiss_index=faiss_index)

        self.load_documents()

        if not self.documents:
            raise ValueError("🚫 Нет документов для индексации")

        self.index = VectorStoreIndex.from_documents(
            self.documents,
            storage_context=StorageContext.from_defaults(
                vector_store=self.vector_store
            ),
            show_progress=True,
        )

        self.index.storage_context.persist(persist_dir=str(self.index_dir))
        faiss.write_index(faiss_index, str(self.index_dir / "faiss.index"))
        print("✅ Индекс успешно создан и сохранен")
        assert (
            self.embedder.embed(["test"]).shape[1] == 384
        ), "Invalid embedding dimension"

    def search(self, query_text: str, top_k: int, min_score: float) -> list:
        """Поиск по векторному индексу"""
        if not self.index:
            return []

        try:
            retriever = self.index.as_retriever(
                similarity_top_k=top_k,
                vector_store_kwargs={"similarity_score_threshold": min_score},
            )

            nodes = retriever.retrieve(query_text)
            return [
                {"text": node.node.get_content(), "score": node.score} for node in nodes
            ]

        except Exception as e:
            print(f"Ошибка поиска: {str(e)}")
            return []
