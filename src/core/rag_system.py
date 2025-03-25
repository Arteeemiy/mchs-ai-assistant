from .embedding.embedder import OptimizedEmbedder
from .llm.mistral_client import MistralAPIClient
from .storage.vector_db import VectorStore
from .validation.response_validator import ResponseValidator
from .prompt_management.prompt_selector import PromptSelector


class RAGSystem:
    def __init__(self, mistral_api_key: str):
        self.embedder = OptimizedEmbedder()
        self.vector_store = VectorStore(
            data_dir="documents", index_dir="faiss_index", embedder=self.embedder
        )
        self.generator = MistralAPIClient(mistral_api_key, max_retries=5)
        self.dialog_history: List[Dict] = []
        self.feedback_examples: List[Dict] = []
        self.validator = ResponseValidator(self.generator)
        self.validation_history = []
        if not self.vector_store.embedder:
            raise ValueError("Embedder not initialized in VectorStore")

        self.prompt_selector = PromptSelector(self.embedder, "mchs_prompts.json")
        if not self.prompt_selector.prompts:
            default_prompt = self._default_prompt_template()
            self.prompt_selector.add_prompt(default_prompt)

        if not self.vector_store.index_exists:
            print("Создание нового индекса...")
            self.vector_store.create_index()

    def _default_prompt_template(self) -> str:
        return """Ты - ассистент МЧС России. Отвечай СТРОГО по инструкциям:
    1. Используй ТОЛЬКО предоставленный контекст
    2. Формат ответа:
    - Пошаговый алгоритм с нумерацией
    - Ссылки на нормативные документы в квадратных скобках
    - Предупреждения об опасных действиях в блоке ⚠️

    Контекст: {context}

    Вопрос: {query}

    Если информации недостаточно - ответь "Требуется уточнение данных от оператора"."""

    def add_prompt_manual(self, prompt: str):
        """Ручное добавление промпта"""
        self.prompt_selector.add_prompt(prompt)
        print(
            f"✅ Промпт добавлен в базу. Всего промптов: {len(self.prompt_selector.prompts)}"
        )

    def process_query(self, query: str) -> str:
        try:
            context = self._retrieve_context(query)
            selected_prompt = self.prompt_selector.find_best_prompt(query)
            full_prompt = selected_prompt.format(context=context, query=query)

            response = self.generator.generate(full_prompt)
            validation = self.validator.validate_response(
                query, context, response, selected_prompt
            )

            recommendation = None
            if validation["relevance"] < 3 or not validation["accuracy"]:
                recommendation = self.validator.generate_recommendation(
                    query, context, validation
                )

            self._save_validation_result(query, response, validation, recommendation)

            final_response = response
            if recommendation:
                final_response += f"\n\n---\n🔍 Рекомендация:\n{recommendation}"

            return final_response if final_response else "Не удалось сформировать ответ"
        except Exception as e:
            print(f"Ошибка поиска: {str(e)}")
            return []

    def _save_validation_result(self, query, response, validation, recommendation=None):
        self.validation_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "response": response,
                "validation": validation,
                "recommendation": recommendation,
            }
        )

    def get_validation_stats(self):
        stats = defaultdict(list)
        for entry in self.validation_history:
            for k, v in entry["validation"].items():
                stats[k].append(v)
        return stats

    def _retrieve_context(self, query: str, top_k=5) -> str:
        """Оптимизированный поиск с учетом чанков"""
        results = self.vector_store.search(
            query_text=query,
            top_k=top_k * 3,
            min_score=0.6,
        )

        return "\n".join([res["text"] for res in results]) if results else ""

    def _save_to_history(self, query, context, prompt, response):
        """Сохранение истории диалога"""
        self.dialog_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "context": context,
                "prompt": prompt,
                "response": response,
            }
        )

    def add_feedback(self, query: str, ideal_answer: str):
        """Добавление обратной связи"""
        last_interaction = next(
            (item for item in reversed(self.dialog_history) if item["query"] == query),
            None,
        )

        if last_interaction:
            self.feedback_examples.append(
                {**last_interaction, "ideal_answer": ideal_answer}
            )
