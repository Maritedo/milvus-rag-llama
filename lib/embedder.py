import requests
from math import log10

from .utils import get_size_readable


class Embedder:
    def __init__(self) -> None:
        raise NotImplementedError
    def embbed(self, sentences: str | list):
        pass
    def name(self):
        pass
    def dimension(self):
        pass
    def __str__(self):
        return self.name()
    
class LocalEmbedder(Embedder):
    def __init__(self, model_name) -> None:
        self.model_name = model_name.split("/")[-1].replace("-", "_")
        self.__model_path = model_name
        self.__loaded = False
        self.model = None
    
    def __lazy_load(self):
        if not self.__loaded:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.__loaded = True
    
    def embbed(self, sentences):
        self.__lazy_load()
        if type(sentences) == str:
            sentences = [sentences]
        return self.model.encode(sentences).tolist()
    
    def name(self):
        return self.model_name
    
    def dimension(self):
        self.__lazy_load()
        return self.model.get_sentence_embedding_dimension()
    
    def __eq__(self, value):
        return super().__eq__(value) or self.__model_path == value.__model_path


class ServerEmbedder(Embedder):
    def __interactive_choose(self, server_url):
        api_url = f"{server_url}/api/tags"
        response = requests.get(api_url)
        if response.status_code == 200:
            models = response.json()["models"]
            print(f"Available models ({len(models)} in total):")
            if len(models) == 0:
                print("No models available.")
                raise Exception("No available models")
            display_len = int(log10(len(models))) + 1
            max_name_len = max([len(model['name']) for model in models])
            for index, model in enumerate(models):
                print(f"{1+index:{display_len}d}. {model['name']:<{max_name_len}} ({get_size_readable(model['size'])})")
            print()
            while True:
                index_str = input(f"Choose a model: ")
                if not index_str.isdigit():
                    continue
                index = int(index_str) - 1
                if 0 <= index < len(models):
                    model = models[index]
                    model_name = model['name']
                    break
        else: raise Exception(f"Failed to get models from {api_url}")
        return (model, model_name)
    
    def __init__(self, server_url: str, model_name: str | None = None) -> None:
        if model_name is None:
            (model, model_name) = self.__interactive_choose(server_url)
            print(f"Selected model: {model['name']} ({get_size_readable(model['size'])})")
        self.model_name = model_name
        self.server_url = server_url
        self.__dimension = None
    
    def embbed(self, sentences):
        if type(sentences) == str:
            sentences = [sentences]
        api_url = f"{self.server_url}/api/embed"
        payload = {
            "input": sentences,
            "model": self.model_name,
            "seed": 42
        }
        print(f"Requesting embeddings from {api_url} with model {self.model_name}...")
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            return response.json()['embeddings']
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")
    
    def dimension(self):
        if self.__dimension is not None:
            return self.__dimension
        api_url = f"{self.server_url}/api/show"
        payload = {
            "model": self.model_name
        }
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            details = response.json()
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")
        for item in details["model_info"].items():
            if item[0].split(".")[-1] == "embedding_length":
                self.__dimension = item[1]
        return self.__dimension
    
    def name(self):
        return self.model_name.split(":")[0].replace(".", "_")
    
    def __eq__(self, value):
        return super().__eq__(value) or (self.server_url == value.server_url and self.model_name == value.model_name)
