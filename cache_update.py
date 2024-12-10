import time
from lib.utils import embedder, collection
import json
from pathlib import Path
from lib.cache import QueryCache, QueryCacheOld

workdir = Path(__file__).parent

if __name__ == "__main__":
    cache = QueryCache(embedder.name())
    cache_old = QueryCacheOld(embedder.name())
    
    with open(Path(__file__).parent / 'data' / 'train_sentence.json', "r") as f:
        json_data = json.load(f)
        train_sentences = [item["sentence"] for item in json_data]
        
    with open(Path(__file__).parent / 'data' / 'test_sentence.json', "r") as f:
        json_data = json.load(f)
        test_sentences = [item["sentence"] for item in json_data]
    
    next = 0
    with open(workdir / 'record' / ('cache_mig_' + embedder.name() + '.next'), 'a+') as f:
        fc = f.read()
        if fc and fc.strip().isdigit():
            next = int(fc)
    
    batch_size = int(input("Enter batch size: "))
    while True:
        if next >= len(test_sentences):
            break
        current_time = time.time()
        print(f"Processing {next} to {min(next + batch_size, len(test_sentences))}...")
        items = test_sentences[next:next+batch_size]
        for index in range(next, min(len(test_sentences) - 1, next + batch_size)):
            res = cache_old.fetch_results(next)
            tobeinserted = []
            for offset, result in enumerate(res):
                tobeinserted.append((result[0], collection.query(expr=f'sentence=="{train_sentences[result[0]]}"')[0]['id'], result[1]))
            cache.insert_results(index, sorted(tobeinserted, key=lambda x: -x[2]))
        
        # results = query(items, n=10)
        # for offset, result in enumerate(results):
        #     # print(f"Input {next+offset}: {items[offset]}")
        #     # print("Results:")
        #     for i, row in enumerate(result):
        #         print(f"  {train_sentences.index(row.entity.get("sentence")):<4}. ({row.distance}) {row.entity.get('sentence')}")
        #     tobeinserted = [(train_sentences.index(row.entity.get("sentence")), row.distance) for row in result]
        #     cache.insert_results(next + offset, tobeinserted)
        #     # input("Press Enter to continue...")
        next = min(next + batch_size, len(test_sentences))
        with open(workdir / 'record' / ('cache_mig_' + embedder.name() + '.next'), 'w') as f:
            f.write(str(next))
        print(f"Time elapsed: {time.time() - current_time:.2f}s")
    data_old = cache_old.fetch_results(0)