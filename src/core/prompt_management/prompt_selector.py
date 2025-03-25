import numpy as np


class PromptSelector:
    def __init__(self, embedder, storage_file="prompts_db.json"):
        self.embedder = embedder
        self.storage = PromptStorage(storage_file)
        self.prompts, embeddings = self.storage.load()

        self.embeddings = np.array(embeddings) if embeddings else np.array([])

        if len(embeddings) == 0 and len(self.prompts) > 0:
            print("Пересчитываем эмбеддинги для промптов...")
            self.embeddings = self.embedder.embed(self.prompts)
            self._save_to_storage()

    def add_prompt(self, prompt: str):
        """Добавляет промпт и сохраняет в хранилище"""
        new_embedding = self.embedder.embed([prompt])[0]
        self.prompts.append(prompt)

        if self.embeddings.size == 0:
            self.embeddings = np.array([new_embedding])
        else:
            self.embeddings = np.vstack([self.embeddings, new_embedding])

        self._save_to_storage()

    def _save_to_storage(self):
        """Синхронизация с файлом"""
        self.storage.save(self.prompts, self.embeddings)

    def remove_prompt(self, index: int):
        """Удаление промпта"""
        if 0 <= index < len(self.prompts):
            del self.prompts[index]
            self.embeddings = np.delete(self.embeddings, index, axis=0)
            self._save_to_storage()

    def find_best_prompt(self, query: str) -> str:
        """Находит наиболее подходящий промпт для запроса"""
        if not self.prompts:
            return None

        query_embed = self.embedder.embed([query])[0]

        similarities = np.dot(self.embeddings, query_embed)
        best_idx = np.argmax(similarities)

        return self.prompts[best_idx]

    def get_all_prompts(self) -> List[str]:
        """Получение всех промптов"""
        return self.prompts.copy()
