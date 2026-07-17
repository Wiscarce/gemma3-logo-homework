"""
将 chat-format JSONL 转为 ms-swift 兼容的扁平 query/response 格式
ms-swift 4.4.1 的 chat 格式路径与 datasets 3.x 不兼容,
扁平格式走的是另一条代码路径, 可绕过 bug
"""
import json
import os

DATA_DIR = "/content/logo-detailed-prompt"

for split in ["train", "valid"]:
    in_path = os.path.join(DATA_DIR, f"{split}.jsonl")
    out_path = os.path.join(DATA_DIR, f"{split}_flat.jsonl")
    records = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            msgs = d["messages"]
            records.append({
                "system": msgs[0]["content"],
                "query": msgs[1]["content"],
                "response": msgs[2]["content"],
            })
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{out_path}: {len(records)} rows")
