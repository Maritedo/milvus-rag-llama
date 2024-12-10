import os
from pathlib import Path
from datetime import datetime

# filedir = workdir / "data"
# load_dotenv(workdir / ".env")


# 捕获并保存异常
import signal
class KeyboardInterruptTemporaryIgnored:
    def __init__(self, ):
        
        self._interrupted = False
        self._original_signal_handler = signal.signal(signal.SIGINT, self._ignore_interrupt)

    def _ignore_interrupt(self, signum, frame):
        self._interrupted = True

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type == KeyboardInterrupt:
            raise exc_value
        if self._interrupted:
            raise KeyboardInterrupt
        # 恢复原始中断处理
        signal.signal(signal.SIGINT, self._original_signal_handler)


def load_checkpoint(filename, default=None):
    if os.path.exists(filename):
        if os.path.isfile(filename) | os.path.islink(filename):
            with open(filename, "r") as f:
                return f.read()
        else:
            raise Exception(f"{filename} is not a file.")
    else:
        if default is not None:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(default)
            return default


def get_time_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_hf_models():
    models_dir = Path(os.path.expanduser("~")) / '.cache' / 'huggingface' / 'hub'
    models_list = []
    for dir in os.listdir(models_dir):
        if dir.startswith('models--'):
            models_list.append(dir[8:].replace('--', '/'))
    return models_list


def get_size_readable(size):
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    for unit in units:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def _choose_local():
    models_list = get_hf_models()
    if len(models_list) == 0:
        print("No models available.")
        raise Exception("No available models")
    print("Available models:")
    for index, model in enumerate(models_list):
        print(f"{index+1}. {model}")
    while True:
        index_str = input("Choose a model: ")
        if not index_str.isdigit():
            continue
        index = int(index_str) - 1
        if 0 <= index < len(models_list):
            model = models_list[index]
            print(f"Selected model: {model}")
            from .embedder import LocalEmbedder
            return LocalEmbedder(model)

def _choose_server():
    server_url = input("Enter server URL:")
    if not server_url:
        server_url = "http://172.16.129.30:11434"
    from .embedder import ServerEmbedder
    return ServerEmbedder(server_url=server_url, model_name=None)

def choose_embedder():
    embedder = None
    while embedder is None:
        i = input("Choose an embedder (local/server): ").lower()
        if not i.lower() in ["local", "server", "l", "s"]:
            continue
        if i.lower() in ["local", "l"]:
            embedder = _choose_local()
        else:
            embedder = _choose_server()
    return embedder
