import os
from src.core.rag_system import RAGSystem


def demonstrate_rag():
    rag = RAGSystem(api_key=os.getenv("MISTRAL_API_KEY", "test_mode"))

    query = "Действия при пожаре?"
    response = rag.process_query(query)

    print("\n🔍 Пример работы системы:")
    print(f"Запрос: {query}")
    print(f"Ответ: {response}\n")


if __name__ == "__main__":
    demonstrate_rag()
