import re
import random


class ResponseValidator:
    def __init__(self, generator):
        self.generator = generator
        self.validation_prompts = {
            "relevance": """Оцени релевантность ответа вопросу по шкале 1-5. Ответ должен строго соответствовать следующим требованиям МЧС:
    1 - Ответ не соответствует вопросу
    3 - Частичное соответствие с отсутствием ключевых этапов
    5 - Полное соответствие с пошаговым алгоритмом
    
    Вопрос: {query}
    Ответ: {response}
    
    Оценка (только цифра):""",
            "accuracy": """Соответствует ли ответ следующим документам? ({context_sources})
    Контекст: {context}
    Ответ: {response}
    
    Требования:
    1. Все технические параметры должны соответствовать СП 112.13330.2022
    2. Действия должны соответствовать Приказу МЧС №632
    
    Ответ (строго ДА/НЕТ):""",
            "completeness": """Проверь полноту ответа по чек-листу МЧС:
    1. Наличие нумерованных этапов
    2. Указание опасных зон
    3. Ссылки на нормативные документы
    4. Протоколы эвакуации
    5. Средства индивидуальной защиты
    
    Ответ: {response}
    
    Количество выполненных пунктов (цифра 0-5):""",
            "safety": """Выяви нарушения безопасности по:
    - ГОСТ Р 22.9.19-2022
    - ФЗ №123-ФЗ "Технический регламент"
    
    Ответ: {response}
    
    Нарушения обнаружены? (ДА/НЕТ):""",
            "structure": """Соответствие служебной инструкции МЧС:
    1. Четкая структура команд
    2. Использование стандартных формулировок
    3. Выделение зон ответственности
    
    Ответ (ДА/НЕТ):""",
            "sources": """Проверь наличие обязательных ссылок:
    - НПБ 101-03
    - СП 5.13130.2009 
    - Приказы МЧС России
    
    Найдены ссылки? (ДА/НЕТ):""",
        }
        self.regulatory_docs = [
            "СП 112.13330.2022",
            "Приказ МЧС №632",
            "ГОСТ Р 22.9.19-2022",
            "ФЗ №123-ФЗ",
            "НПБ 101-03",
            "СП 5.13130.2009",
        ]

    def validate_response(self, query, context, response, prompt):
        validation = {}

        for key, template in self.validation_prompts.items():
            filled_prompt = template.format(
                query=query,
                context=context,
                response=response,
                prompt=prompt,
                context_sources=self.regulatory_docs,
            )
            llm_response = self.generator.generate(filled_prompt).strip()
            validation[key] = self._parse_response(key, llm_response)

        return validation

    def _parse_response(self, key, response):
        response = response.lower().strip()
        patterns = {
            "relevance": r"\b[1-5]\b",
            "completeness": r"\b[0-5]\b",
            "accuracy": r"\b(да|нет|yes|no)\b",
            "safety": r"\b(да|нет|yes|no)\b",
            "structure": r"\b(да|нет|yes|no)\b",
            "sources": r"\b(да|нет|yes|no)\b",
        }
        match = re.search(patterns[key], response)
        if not match:
            return 0 if key in ["relevance", "completeness"] else False

        value = match.group()
        return int(value) if value.isdigit() else value in ["да", "yes"]

    def generate_recommendation(
        self, query: str, context: str, validation: dict
    ) -> str:
        """
        Генерирует рекомендации для улучшения ответа на основе результатов валидации

        Параметры:
            query: исходный запрос пользователя
            context: извлеченный контекст из RAG
            validation: словарь с результатами проверки

        Возвращает:
            Строка с улучшенным ответом согласно требованиям МЧС
        """
        issues = []
        if validation["relevance"] < 3:
            issues.append("▪ Низкая релевантность исходному запросу")
        if not validation["accuracy"]:
            issues.append("▪ Расхождения с нормативными документами")
        if validation["completeness"] < 3:
            issues.append("▪ Неполное описание процедур")
        if validation["safety"]:
            issues.append("▪ Обнаружены опасные рекомендации")
        if not validation["structure"]:
            issues.append("▪ Нарушена структура служебной инструкции")
        if not validation["sources"]:
            issues.append("▪ Отсутствуют ссылки на нормативные документы")

        prompt = (
            f"Ты эксперт МЧС России. Перепиши ответ, исправляя следующие нарушения:\n"
            f"{'▪ ' + '▪ '.join(issues) if issues else '▪ Общие требования не выполнены'}\n\n"
            f"**Контекст для справки:**\n{context[:2000]}{'...' if len(context)>2000 else ''}\n\n"
            f"**Исходный запрос:**\n{query}\n\n"
            f"**Требования к новому ответу:**\n"
            f"1. Соответствие {random.choice(self.regulatory_docs)}\n"
            "2. Четкая структура:\n"
            "   - Нумерованные этапы действий\n"
            "   - Выделение зон ответственности\n"
            "   - Временные рамки операций\n"
            "3. Обязательные элементы:\n"
            "   ⚠️ Предупреждения об опасностях\n"
            "   [Ссылки на нормативные документы]\n"
            "   ► Указание используемого оборудования\n"
            "4. Использование официальной терминологии МЧС\n\n"
            f"**Формат ответа:**\n"
            "Дай ТОЛЬКО исправленную версию ответа без комментариев, строго соблюдая:\n"
            "- Максимальную конкретику\n"
            "- Нумерацию этапов\n"
            "- Разделение технических требований и действий"
        )

        try:
            recommendation = self.generator.generate(prompt)

            recommendation = self._postprocess_recommendation(recommendation)

            return recommendation

        except Exception as e:
            print(f"Ошибка генерации рекомендации: {str(e)}")
            return "Рекомендации недоступны. Требуется проверка экспертом."

    def _postprocess_recommendation(self, text: str) -> str:
        """Постобработка сгенерированного текста"""
        text = re.sub(r"\[(док|норматив)\s*(\d+)\]", r"[\2]", text, flags=re.IGNORECASE)

        text = re.sub(
            r"(опасно|внимание|предупреждение)", "⚠️", text, flags=re.IGNORECASE
        )

        text = text.replace("**", "").replace("```", "")

        return text.strip()
