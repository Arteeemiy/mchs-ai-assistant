from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class DocumentWatcher:
    def __init__(self, data_dir: Path, update_handler: callable):
        self.data_dir = data_dir
        self.update_handler = update_handler
        self.observer = Observer()
        self.lock = threading.Lock()

        class Handler(FileSystemEventHandler):
            def __init__(self, outer):
                self.outer = outer

            def on_modified(self, event):
                self._process_event(event)

            def on_created(self, event):
                self._process_event(event)

            def _process_event(self, event):
                if not event.is_directory and event.src_path.endswith(".json"):
                    with self.outer.lock:
                        file_path = Path(event.src_path)
                        self.outer.update_handler(file_path)

        self.event_handler = Handler(self)

    def start(self):
        self.observer.schedule(self.event_handler, str(self.data_dir), recursive=True)
        self.observer.start()
        print(f"🔭 Наблюдение за {self.data_dir} запущено")

    def stop(self):
        self.observer.stop()
        self.observer.join()
        print("👀 Наблюдение остановлено")
