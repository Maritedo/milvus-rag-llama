from pathlib import Path
import os
import json
from math import log10
from lib.evaluate import count_contained_intervals, match_list 

workdir = Path(os.getcwd())

list = []
for root, dirs, files in os.walk(workdir / "results"):
    for file in files:
        if file.endswith(".json"):
            list.append(file)
list_display_size = int(log10(len(list))) + 1
print("Available files:")
for index, file in enumerate(list):
    print(f'{index + 1:>{list_display_size}}. {file}')
while True:
    try:
        selected_file = int(input("Select file: ")) - 1
        if selected_file < 0 or selected_file >= len(list):
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
    
import os
if not os.path.exists(workdir / "statis"):
    os.mkdir(workdir / "statis")
outputfile = open(workdir / "statis" / f"{list[selected_file].replace(".json", ".txt")}", "w", encoding="utf-8")

with open(workdir / "results" / list[selected_file], "r", encoding="utf-8") as f:
    generated_datas = json.load(f)

with open(workdir / "data" / "test_sentence.parsed.json", "r", encoding="utf-8") as f:
    target_datas = json.load(f)

total_ent = [0, 0, 0]
total_rel = [0, 0, 0]
# target, generated, matched
for key in generated_datas.keys():
    generated = generated_datas[str(key)]
    target = target_datas[int(key)]
    if "entities" in  generated.keys():
        res_ent = count_contained_intervals(target["entities"], generated["entities"])
        total_ent[0] += len(target["entities"])
        total_ent[1] += len(generated["entities"])
        total_ent[2] += res_ent[0]
    if "relations" in generated.keys():
        res_rel = count_contained_intervals(target["relations"], generated["relations"])
        total_rel[0] += len(target["relations"])
        total_rel[1] += len(generated["relations"])
        total_rel[2] += res_rel[0]

if total_ent[0] != 0:
    print("Entities", total_ent[0], total_ent[1], total_ent[2], file=outputfile)
    print(f"Recall: {total_ent[2] / total_ent[0] if total_ent[0] > 0 else -1}, Precision: {total_ent[2] / total_ent[1] if total_ent[1] > 0 else -1}, F1 Score: {2 * total_ent[2] / (total_ent[0] + total_ent[1]) if total_ent[0] + total_ent[1] > 0 else -1}", file=outputfile)

if total_rel[0] != 0:
    print("Relations", total_rel[0], total_rel[1], total_rel[2], file=outputfile)
    print(f"Recall: {total_rel[2] / total_rel[0] if total_rel[0] > 0 else -1}, Precision: {total_rel[2] / total_rel[1] if total_rel[1] > 0 else -1}, F1 Score: {2 * total_rel[2] / (total_rel[0] + total_rel[1]) if total_rel[0] + total_rel[1] > 0 else -1}", file=outputfile)