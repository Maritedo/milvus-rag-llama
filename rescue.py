from pathlib import Path
import os
import json
from math import log10

workdir = Path(os.getcwd())

file_ls = []
for root, dirs, files in os.walk(workdir / "results"):
    for file in files:
        if file.endswith(".json"):
            file_ls.append(file)
list_display_size = int(log10(len(file_ls))) + 1
print("Available files:")
for index, file in enumerate(file_ls):
    print(f'{index + 1:>{list_display_size}}. {file}')
while True:
    try:
        selected_file = int(input("Select file: ")) - 1
        if selected_file < 0 or selected_file >= len(file_ls):
            raise Exception
        break
    except ValueError:
        print("Please enter a number.")
        continue
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception:
        print("Invalid number.")
        continue

with open(workdir / "results" / file_ls[selected_file], "r") as f:
    generated_datas: dict = json.load(f)

with open(workdir / "data" / "test_sentence.json", "r") as f:
    target_datas = json.load(f)

def check(output):
    if (
        "ner" not in output.keys()
        or len(output["ner"]) == 0
        or type(output["ner"][0]) != list
    ):
        return False
    else:
        len_ner = set(len(i) for i in output["ner"])
        if len(len_ner) > 0 and len_ner != {3}:
            return False
    if len(output.keys()) <= 1:
        return False
    elif "relations" not in output.keys() or (
        len(output["relations"]) != 0
        and type(output["relations"][0]) != list
    ):
        return False
    else:
        len_rel = set(len(i) for i in output["relations"])
        if len(len_rel) > 0 and len_rel != {5}:
            return False
    return True

file_ls = []
for index, item in sorted(generated_datas.items(), key=lambda x: int(x[0])):
    if not check(item):
        print(f"Error at index {index}")
        file_ls.append(index)
print(file_ls)
print(len(file_ls))