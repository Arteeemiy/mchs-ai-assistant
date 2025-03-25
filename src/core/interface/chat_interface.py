class ChatInterface:
    def __init__(self, rag_system: RAGSystem):
        self.rag = rag_system
        self._print_welcome()
        self.batch_mode = False
        self.question_queue = []
        self.current_batch = []

    def _print_welcome(self):
        print(
            "\n🚒 Чат-бот МЧС RAG система\n"
            "───────────────────────────────\n"
            "Команды управления:\n"
            "/добавить_промпт - добавить новый шаблон\n"
            "/удалить_промпт [N] - удалить шаблон по номеру\n"
            "/обучить [ответ] - сохранить пример обучения\n"
            "/список [Передача списка] - Получить ответы из списка вопросов\n"
            "/промпты - список всех шаблонов\n"
            "/история - последние ответы\n"
            "/сброс_промптов - сброс к начальному шаблону\n"
            "/отладка - техническая информация\n"
            "/выход - завершение работы\n"
            "───────────────────────────────"
        )

    def start_chat(self):
        while True:
            user_input = input("\n➤ Пользователь: ").strip()
            if user_input.lower() == "/выход":
                print("\nСеанс завершен. До свидания!")
                break
            self._process_command(user_input)

    def _process_command(self, input_text: str):
        if input_text.startswith("/добавить_промпт"):
            self._handle_add_prompt()
        if self.batch_mode:
            self._handle_batch_input(input_text)
        elif input_text.startswith("/список"):
            self._start_batch_mode()
        elif input_text.startswith("/удалить_промпт"):
            self._handle_remove_prompt(input_text)
        elif input_text.startswith("/обучить "):
            self._handle_training(input_text)
        elif input_text == "/промпты":
            self._show_prompts()
        elif input_text == "/история":
            self._show_history()
        elif input_text == "/сброс_промптов":
            self._reset_prompts()
        elif input_text == "/отладка":
            self._show_debug_info()
        else:
            self._generate_response(input_text)

    def _generate_response(self, query: str):
        """Генерация ответа на запрос с проверкой валидации"""
        response = self.rag.process_query(query)
        print(f"\n🤖 Бот: {response}")

    def _start_batch_mode(self):
        """Активация пакетного режима"""
        self.batch_mode = True
        self.question_queue = []
        print("\n🌀 Режим пакетной обработки активирован.")
        print("Введите вопросы по одному (пустая строка - завершение):")

    def _handle_batch_input(self, input_text: str):
        """Обработка ввода в пакетном режиме"""
        if input_text.strip() == "":
            self._execute_batch_processing()
        else:
            self.question_queue.append(input_text)
            print(f"✅ Вопрос добавлен в очередь ({len(self.question_queue)})")

    def _execute_batch_processing(self):
        """Запуск обработки накопленных вопросов"""
        self.batch_mode = False
        if not self.question_queue:
            print("⚠️ Очередь вопросов пуста!")
            return

        print(f"\n🔍 Начинаю обработку {len(self.question_queue)} вопросов...")
        results = []

        for idx, question in enumerate(self.question_queue, 1):
            print(f"\n📋 Вопрос {idx}/{len(self.question_queue)}: {question}")
            response = self.rag.process_query(question)
            results.append({"question": question, "answer": response})
            print(f"🤖 Ответ: {response[:150]}...")

        self._save_batch_results(results)
        self.question_queue = []
        print("\n✅ Пакетная обработка завершена!")

    def _save_batch_results(self, results: list):
        """Сохранение результатов в JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"batch_results_{timestamp}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"📁 Результаты сохранены в файл: {filename}")

    def _handle_add_prompt(self):
        """Обработчик добавления нового промпта"""
        print("\nВведите новый промпт (Ctrl+D/Ctrl+Z для завершения):")
        print("Пример формата:")
        print(self.rag._default_prompt_template())

        try:
            lines = []
            while True:
                line = input("> ")
                lines.append(line)
        except EOFError:
            pass

        new_prompt = "\n".join(lines).strip()
        if new_prompt:
            self.rag.add_prompt_manual(new_prompt)
        else:
            print("⚠️ Пустой промпт не был добавлен")

    def _handle_remove_prompt(self, command: str):
        """Удаление промпта по номеру"""
        try:
            index = int(command.split()[-1]) - 1
            self.rag.prompt_selector.remove_prompt(index)
            print(f"✅ Промпт #{index+1} удален")
        except Exception as e:
            print(f"❌ Ошибка: {str(e)}")

    def _handle_training(self, command: str):
        """Сохранение примера идеального ответа"""
        try:
            ideal_answer = command[len("/обучить ") :].strip()
            if not self.rag.dialog_history:
                print("⚠️ Сначала задайте вопрос!")
                return

            last_query = self.rag.dialog_history[-1]["query"]
            self.rag.add_feedback(last_query, ideal_answer)
            print("✅ Пример ответа сохранен")

        except Exception as e:
            print(f"❌ Ошибка: {str(e)}")

    def _show_prompts(self):
        """Показать последние 3 промпта"""
        prompts = self.rag.prompt_selector.get_all_prompts()
        print(f"\nТекущие шаблоны ({len(prompts)}):")
        for i, p in enumerate(prompts[-3:], 1):
            print(f"\nШаблон #{i}:\n{p[:300]}...")

    def _show_history(self):
        """Показать историю диалога"""
        print("\nПоследние 3 ответа:")
        for item in self.rag.dialog_history[-3:]:
            print(f"\n▪ Вопрос: {item['query']}")
            print(f"▸ Ответ: {item['response'][:150]}...")
            print(f"⚙ Промпт: {item['prompt'][:80]}...")

    def _reset_prompts(self):
        """Сброс всех промптов"""
        self.rag.prompt_selector.prompts = [self.rag._default_prompt_template()]
        print("\n✅ Все шаблоны сброшены до начального состояния")

    def _show_debug_info(self):
        """Техническая информация"""
        print("\nТехническая информация:")
        print(f"▪ Размер индекса: {len(self.rag.vector_store.documents)} чанков")
        print(f"▪ Примеров обратной связи: {len(self.rag.feedback_examples)}")
        print(f"▪ Последний промпт: {self.rag.prompt_selector.prompts[-1][:200]}...")
