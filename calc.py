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

with open(workdir / "results" / list[selected_file], "r") as f:
    generated_datas = json.load(f)

with open(workdir / "data" / "test_sentence.parsed.json", "r") as f:
    target_datas = json.load(f)

print("Comparing...")
total_t = 0
total_g = 0
matched = 0
for key in generated_datas.keys():
    generated = generated_datas[str(key)]
    target = target_datas[int(key)]
    # re1 = match_list(target["ner"], generated["ner"])
    re2 = count_contained_intervals(target["entities"], generated["entities"])
    # if re1 != re2:
    #     print(f"Error at index {index}")
    #     print(f"Target: {target['ner']}")
    #     print(f"Generated: {generated['ner']}")
    #     print(f"match_list: {re1}")
    #     print(f"count_contained_intervals: {re2}")
    total_t += len(target["entities"])
    total_g += len(generated["entities"])
    matched += re2[0]
print(total_t, total_g, matched)
print(f"Recall: {matched / total_t if total_t > 0 else -1}, Precision: {matched / total_g if total_g > 0 else -1}, F1 Score: {2 * matched / (total_t + total_g) if total_t + total_g > 0 else -1}")