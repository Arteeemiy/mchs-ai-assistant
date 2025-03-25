import json
import numpy as np
from pathlib import Path


class PromptStorage:
    def __init__(self, storage_file="prompts_db.json"):
        self.storage_file = Path(storage_file)
        self._ensure_storage_exists()

    def _ensure_storage_exists(self):
        if not self.storage_file.exists():
            with open(self.storage_file, "w") as f:
                json.dump({"prompts": [], "embeddings": []}, f)

    def save(self, prompts: list, embeddings: list):
        data = {"prompts": prompts, "embeddings": [emb.tolist() for emb in embeddings]}
        with open(self.storage_file, "w") as f:
            json.dump(data, f, indent=2)

    def load(self):
        try:
            with open(self.storage_file, "r") as f:
                data = json.load(f)
            return data["prompts"], [np.array(emb) for emb in data["embeddings"]]
        except Exception as e:
            print(f"Ошибка загрузки промптов: {str(e)}")
            return [], []
