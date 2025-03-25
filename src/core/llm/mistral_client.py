import requests
import time


class MistralAPIClient:
    def __init__(self, api_key: str, model="mistral-medium", max_retries=3):
        self.api_key = api_key
        self.base_url = "https://api.mistral.ai/v1"
        self.model = model
        self.max_retries = max_retries
        self.timeout = 60

    def generate(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4,
            "max_tokens": 4000,
            "top_p": 0.9,
            "stop": ["\n##", "```"],
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"].strip()
                else:
                    print(
                        f"API Error (attempt {attempt+1}): {response.status_code} - {response.text}"
                    )

            except requests.exceptions.Timeout:
                print(f"⚠️ Тайм-аут запроса (попытка {attempt+1}/{self.max_retries})")
                if attempt == self.max_retries - 1:
                    return "Ошибка: превышено время ожидания ответа от сервера"

                time.sleep(2**attempt)

            except Exception as e:
                print(f"🚨 Критическая ошибка: {str(e)}")
                return "Ошибка соединения с сервером"

        return "Не удалось получить ответ после нескольких попыток"
