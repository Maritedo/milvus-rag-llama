from math import log10
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

class App:
    def __init__(self) -> None:
        self.collection = None
        
        server = input("Enter server URL: ")
        if server == "":
            server = "172.16.129.30"
        port = input("Enter server port: ")
        if port == "":
            port = "19530"

        connections.connect("default", host=server, port=port)
        print("Connected to Milvus.")
    
    def run(self):
        self.select_model()
    
    def select_model(self):
        collections = utility.list_collections()
        print("Available collections:")
        display_len = int(log10(len(collections))) + 1
        for i, collection_name in enumerate(collections):
            print(f"{i+1:{display_len}d}. {collection_name}")
        index = -1
        while index == -1:
            index_ = input("Choose a collection (0 for exit): ")
            if index_.isdigit():
                index = int(index_) - 1
                if index == -1:
                    return
                elif 0 <= index < len(collections):
                    self.collection = Collection(name=collections[index])
                    break
                else:
                    print("Invalid index.")
            else:
                print("Invalid input.")
        self.deal_with_model()
        
    def deal_with_model(self):
        action = ""
        try:
            while action != "exit":
                action = input("Input an action ('exit' to quit): ")
                if action == "count":
                    print(f"Collection '{self.collection.name}' has {self.collection.num_entities} entities.")
                elif action == "delete":
                    self.delete()
                elif action == "search":
                    self.search()
                elif action == "describe":
                    self.describe()
                elif action == "list":
                    self.list()
                elif action == "exit":
                    break
                else:
                    print("Invalid action.")
        except KeyboardInterrupt:
            print("\n")
            pass
        except Exception as e:
            print(e.with_traceback())
        finally:
            self.select_model()
        

App().run()