"""
导出 SVG 图标对比: Sonnet GT vs LoRA 模型生成
输出到 /content/svg_comparison/ 目录，每个样本一个 HTML 对比文件
也可单独导出 .svg 文件
"""
import json, os, sys, re, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

DATA = "/content/logo-detailed-prompt/valid.jsonl"
MODEL_PATH = "/content/output/v7-20260717-055145/checkpoint-100"
BASE_MODEL = "/content/gemma3-270m"
OUT_DIR = "/content/svg_comparison"

os.makedirs(OUT_DIR, exist_ok=True)

# ---- 1. 加载数据 ----
with open(DATA, encoding="utf-8") as f:
    samples = [json.loads(l) for l in f if l.strip()]

# ---- 2. 加载模型 ----
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, MODEL_PATH)
model.eval()

SYSTEM_PROMPT = samples[0]["messages"][0]["content"]

# ---- 3. 生成 + 保存 ----
for i, s in enumerate(samples):
    prompt = s["messages"][1]["content"]
    svg_gt = s["messages"][2]["content"]

    # 保存 GT SVG
    with open(f"{OUT_DIR}/{i+1:02d}_gt_sonnet.svg", "w", encoding="utf-8") as f:
        f.write(svg_gt)

    # 生成
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.7, top_p=0.9,
                                 do_sample=True, pad_token_id=tokenizer.eos_token_id)
    gen = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    m = re.search(r"<svg[\s\S]*?</svg>", gen, re.IGNORECASE)
    svg_gen = m.group(0) if m else (gen[:gen.index("<svg")] + "</svg>" if "<svg" in gen else gen)

    with open(f"{OUT_DIR}/{i+1:02d}_gen_lora.svg", "w", encoding="utf-8") as f:
        f.write(svg_gen)

    # HTML 对比页
    prompt_short = prompt[:120].replace("`", "'").replace("$", "")
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>#{i+1} SVG Compare</title>
<style>body{{font-family:system-ui;background:#1a1a2e;color:#eee;padding:20px}}
h2{{color:#e94560}}h3{{color:#0f3460;background:#16213e;padding:6px 12px;border-radius:4px}}
.pair{{display:flex;gap:20px;flex-wrap:wrap;margin-bottom:30px}}
.card{{background:#16213e;border-radius:12px;padding:16px;flex:1;min-width:300px}}
.card img,.card svg{{width:256px;height:256px;background:#fff;border-radius:8px;display:block;margin:10px auto}}
.badge{{display:inline-block;padding:2px 10px;border-radius:10px;font-size:12px;margin-left:8px}}
.gt{{background:#0f3460;color:#53d769}} .lora{{background:#0f3460;color:#e94560}}
.prompt{{font-size:13px;color:#aaa;margin:8px 0;line-height:1.4;max-width:600px}}
</style></head><body>
<h2>#{i+1} / {len(samples)}</h2>
<p class="prompt">{prompt_short}…</p>
<div class="pair">
<div class="card"><h3>Sonnet GT <span class="badge gt">ground truth</span></h3>{svg_gt}</div>
<div class="card"><h3>LoRA 270M <span class="badge lora">generated</span></h3>{svg_gen}</div>
</div>
</body></html>"""
    with open(f"{OUT_DIR}/{i+1:02d}_compare.html", "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  [{i+1}/{len(samples)}] done  gt={len(svg_gt)}ch  gen={len(svg_gen)}ch")

print(f"\nSaved to {OUT_DIR}/")
print(f"  *_gt_sonnet.svg   — Sonnet 原始 SVG")
print(f"  *_gen_lora.svg    — LoRA 模型生成")
print(f"  *_compare.html    — 并排对比（浏览器打开）")
