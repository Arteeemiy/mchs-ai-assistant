import os
from dotenv import load_dotenv
from .core.rag_system import RAGSystem
from .interface.chat_interface import ChatInterface


def main():
    load_dotenv()

    mistral_api_key = os.getenv("MISTRAL_API_KEY")

    if not mistral_api_key:
        raise ValueError(
            "MISTRAL_API_KEY не найден.\n"
            "Добавьте его в .env файл или переменные окружения.\n"
            "Инструкция: https://github.com/yourname/mchs-ai-assistant#setup"
        )

    rag_system = RAGSystem(mistral_api_key)
    chat = ChatInterface(rag_system)
    chat.start_chat()


if __name__ == "__main__":
    main()
