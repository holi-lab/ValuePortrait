#!/usr/bin/env python3
"""Build the static JSON data consumed by the project page (project_page/static/data).

Sources (all inside this repository):
  data/query-response-tagged/query-response-tagged.json   query/response texts
  data/correlation_results/*.json                         Spearman r and p per (item, dimension)
  lm_evaluation/score_results/final_results_0.3_pos_centered/*.json   per-model dimension scores
  lm_evaluation/average_outputs/*_averaged_results.json   per-item model ratings (6 prompts averaged)
  lm_evaluation/outputs/final/*/*_results.json            raw per-prompt model responses

Run from the repository root:  python3 scripts/build_project_page_data.py
"""
import glob
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "project_page", "static", "data")
os.makedirs(os.path.join(OUT, "models"), exist_ok=True)

PVQ = ["Universalism", "Benevolence", "Conformity", "Tradition", "Security",
       "Power", "Achievement", "Hedonism", "Stimulation", "Self_Direction"]
HI = ["Self_Transcendence", "Conservation", "Self_Enhancement", "Openness_to_Change"]
BFI = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
VERSIONS = ["v1", "v1_reversed", "v2", "v2_reversed", "v3", "v3_reversed"]
SRC_BY_PREFIX = {1: "Reddit", 2: "DearAbby", 3: "ShareGPT", 4: "LMSYS"}
# Item-dimension pairs that lm_evaluation/score.py leaves out of the dimension scores.
EXCLUDED_PAIRS = [[3902, 4, "Achievement"], [1901, 1, "Universalism"]]

# Models evaluated through APIs (33) -> display name, family, flags.
MODELS = [
    ("chatgpt-4o-latest", "chatgpt-4o-latest", "OpenAI", {}),
    ("gpt-3.5-turbo", "gpt-3.5-turbo", "OpenAI", {}),
    ("gpt-4o-2024-05-13", "gpt-4o-2024-05-13", "OpenAI", {}),
    ("gpt-4o-2024-08-06", "gpt-4o-2024-08-06", "OpenAI", {}),
    ("gpt-4o-2024-11-20", "gpt-4o-2024-11-20", "OpenAI", {}),
    ("gpt-4o-mini-2024-07-18", "gpt-4o-mini", "OpenAI", {}),
    ("o1-mini-2024-09-12", "o1-mini", "OpenAI", {"reasoning": True}),
    ("o3-mini-2025-01-31", "o3-mini", "OpenAI", {"reasoning": True}),
    ("claude-3-5-haiku-20241022", "claude-3.5-haiku", "Anthropic", {}),
    ("claude-3-5-sonnet-20241022", "claude-3.5-sonnet", "Anthropic", {}),
    ("claude-3-haiku-20240307", "claude-3-haiku", "Anthropic", {}),
    ("claude-3-opus-20240229", "claude-3-opus", "Anthropic", {}),
    ("claude-3-sonnet-20240229", "claude-3-sonnet", "Anthropic", {}),
    ("claude-3.7-sonnet", "claude-3.7-sonnet", "Anthropic", {}),
    ("claude-3.7-sonnet:thinking", "claude-3.7-sonnet-thinking", "Anthropic", {"reasoning": True}),
    ("gemini-2.0-flash-001", "gemini-2.0-flash-001", "Google", {}),
    ("gemini-2.0-flash-thinking-exp-01-21", "gemini-2.0-flash-thinking", "Google", {"reasoning": True}),
    ("gemini-flash-1.5-8b", "gemini-flash-1.5-8b", "Google", {}),
    ("qwen-max", "qwen-max", "Qwen", {}),
    ("qwen-plus", "qwen-plus", "Qwen", {}),
    ("qwen-turbo", "qwen-turbo", "Qwen", {}),
    ("qwq-32b", "qwq-32b", "Qwen", {"reasoning": True}),
    ("mistral-large", "mistral-large-2407", "Mistral", {}),
    ("mistral-medium", "mistral-medium-2312", "Mistral", {}),
    ("mistral-small", "mistral-small-v24.09", "Mistral", {}),
    ("mistral-small-24b-instruct-2501", "mistral-small-v25.01", "Mistral", {}),
    ("mistral-tiny", "mistral-tiny (7b)", "Mistral", {}),
    ("llama-3.1-8b-instruct", "llama-3.1-8b-instruct", "Llama", {}),
    ("llama-3.1-70b-instruct", "llama-3.1-70b-instruct", "Llama", {}),
    ("llama-3.1-405b-instruct", "llama-3.1-405b-instruct", "Llama", {}),
    ("deepseek-chat", "deepseek-v3", "DeepSeek", {}),
    ("deepseek-r1", "deepseek-r1", "DeepSeek", {"reasoning": True}),
    ("grok-2-1212", "grok-2-1212", "xAI", {}),
]

# Open-weight families reported only in the paper's scaling table (Table 12); no item-level data.
PAPER_ONLY = [
    ("qwen2.5-0.5b-instruct", "Qwen2.5-Instruct", "0.5B", [-0.48, 0.19, 0.30, -0.31, 0.08, 0.11, -0.10, -0.18, 0.60, -0.39], 0.113),
    ("qwen2.5-1.5b-instruct", "Qwen2.5-Instruct", "1.5B", [0.01, -0.10, 0.06, 0.03, 0.04, 0.03, -0.03, 0.05, 0.05, -0.00], 0.002),
    ("qwen2.5-3b-instruct", "Qwen2.5-Instruct", "3B", [-0.38, -0.03, -0.06, -0.20, 0.25, 0.06, -0.05, -0.18, 0.40, 0.19], 0.054),
    ("qwen2.5-7b-instruct", "Qwen2.5-Instruct", "7B", [0.25, 0.47, -0.19, -0.44, 0.49, -0.11, -0.37, -0.10, 0.11, 0.23], 0.108),
    ("qwen2.5-14b-instruct", "Qwen2.5-Instruct", "14B", [0.35, 0.68, 0.18, -0.78, 0.81, -0.37, -0.43, 0.09, 0.14, 0.33], 0.249),
    ("deepseek-r1-distill-qwen-1.5b", "DeepSeek-R1-Distill-Qwen", "1.5B", [-0.50, 0.14, -0.04, -0.02, 0.35, 0.08, 0.01, -0.21, 0.05, 0.39], 0.065),
    ("deepseek-r1-distill-qwen-7b", "DeepSeek-R1-Distill-Qwen", "7B", [-0.03, 0.34, 0.09, -0.10, 0.04, -0.08, 0.03, 0.01, -0.12, 0.12], 0.018),
    ("deepseek-r1-distill-qwen-14b", "DeepSeek-R1-Distill-Qwen", "14B", [0.30, 0.72, -0.27, -0.12, 0.47, -0.18, -0.28, -0.15, -0.07, 0.09], 0.115),
    ("gemma3-4b-it", "Gemma3-it", "4B", [-0.04, 0.05, -0.01, -0.18, -0.11, 0.04, -0.01, -0.01, 0.19, -0.13], 0.011),
    ("gemma3-12b-it", "Gemma3-it", "12B", [0.24, 0.40, -0.03, -0.30, 0.14, -0.15, -0.11, -0.04, 0.09, 0.25], 0.045),
    ("gemma3-27b-it", "Gemma3-it", "27B", [0.34, 0.36, 0.15, -0.33, 0.45, -0.15, -0.30, -0.14, 0.20, 0.07], 0.079),
]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def r3(x):
    return None if x is None else round(float(x), 3)


def r4(x):
    return None if x is None else round(float(x), 4)


# ----------------------------------------------------------------------------- items
tagged = load(os.path.join(ROOT, "data/query-response-tagged/query-response-tagged.json"))
corr = {}
for name, dims, key in (("pvq", PVQ, "pvq_dimension"), ("higher_pvq", HI, "higher_pvq_dimension"), ("bfi", BFI, "bfi_dimension")):
    for row in load(os.path.join(ROOT, f"data/correlation_results/{name}_correlation_results.json")):
        corr[(int(row["portrait_id"]), int(row["option_id"]), row[key])] = (row["spearman_corr"], row["spearman_p_value"])

items = []
item_index = []  # (pid, oid) in a fixed order shared by ratings files
for q in sorted(tagged, key=lambda x: int(x["portrait_id"])):
    pid = int(q["portrait_id"])
    src = SRC_BY_PREFIX[pid // 1000]
    responses = []
    for o in sorted(q["outputs"], key=lambda x: int(x["id"])):
        oid = int(o["id"])
        item_index.append((pid, oid))
        def block(dims):
            out = {}
            for d in dims:
                r, p = corr[(pid, oid, d)]
                out[d] = [r3(r), r4(p)]
            return out
        responses.append({"oid": oid, "text": o["content"], "pvq": block(PVQ), "hi": block(HI), "bfi": block(BFI)})
    items.append({"id": pid, "src": src, "title": q["content"].get("title") or "", "text": q["content"]["text"], "resp": responses})

assert len(items) == 104 and len(item_index) == 520
pos = {(p, o) for (p, o, d), (r, pv) in corr.items() if d in PVQ and r >= 0.3 and pv < 0.05}
sig_pvq = sum(1 for (p, o, d), (r, pv) in corr.items() if d in PVQ and abs(r) >= 0.3 and pv < 0.05)
sig_bfi = sum(1 for (p, o, d), (r, pv) in corr.items() if d in BFI and abs(r) >= 0.3 and pv < 0.05)
print(f"items: {len(items)} queries, {len(item_index)} responses; significant |r|>=0.3 & p<0.05: pvq={sig_pvq} bfi={sig_bfi}")

with open(os.path.join(OUT, "items.json"), "w", encoding="utf-8") as f:
    json.dump({"dims": {"pvq": PVQ, "hi": HI, "bfi": BFI}, "excluded": EXCLUDED_PAIRS, "items": items}, f, ensure_ascii=False, separators=(",", ":"))

# ----------------------------------------------------------------------------- model scores
SCORE_DIR = os.path.join(ROOT, "lm_evaluation/score_results/final_results_0.3_pos_centered")
AVG_DIR = os.path.join(ROOT, "lm_evaluation/average_outputs")
RAW_DIR = os.path.join(ROOT, "lm_evaluation/outputs/final")

models_out = []
ratings = {}
idx_of = {k: i for i, k in enumerate(item_index)}


def recompute(entries, key, dims):
    """Mirror lm_evaluation/score.py (threshold 0.3, positive correlations, centered)."""
    per = {d: [] for d in dims}
    valid = []
    for e in entries:
        r = e.get("numeric_response")
        if r is None:
            continue
        ok = False
        for d, c in e.get(key, []) or []:
            if c > 0.3 and [e["portrait_id"], e["option_id"], d] not in EXCLUDED_PAIRS:
                ok = True
                per[d].append(r)
        if ok:
            valid.append(r)
    mean = sum(valid) / len(valid)
    return {d: (sum(v) / len(v) - mean if v else None) for d, v in per.items()}


for mid, name, family, flags in MODELS:
    sc = load(os.path.join(SCORE_DIR, f"{mid}_averaged_results_0.3_pos_centered_scores.json"))["scores"]
    avg = load(os.path.join(AVG_DIR, f"{mid}_averaged_results.json"))
    # consistency check between released score files and a recomputation from the averaged outputs
    rec = recompute(avg, "correlations", PVQ)
    for d in PVQ:
        diff = abs(rec[d] - sc["pvq"][d]["centered_mean"])
        if diff > 1e-6:
            print(f"  WARN {mid} {d}: recomputed {rec[d]:.4f} vs file {sc['pvq'][d]['centered_mean']:.4f}")
    def pack(block, dims):
        return {d: {"c": r3(block[d]["centered_mean"]), "m": r3(block[d]["original_mean"]), "sd": r3(block[d]["std_dev"]), "n": block[d]["n"]} for d in dims}
    entry = {"id": mid, "name": name, "family": family, "kind": "api",
             "reasoning": bool(flags.get("reasoning")),
             "scores": {"pvq": pack(sc["pvq"], PVQ), "bfi": pack(sc["bfi"], BFI), "hi": pack(sc["higher_pvq"], HI)}}
    models_out.append(entry)

    # per-item averaged rating (6 prompts) in the shared item order
    arr = [None] * 520
    ver = [[None] * 6 for _ in range(520)]
    for e in avg:
        i = idx_of[(int(e["portrait_id"]), int(e["option_id"]))]
        arr[i] = r3(e["numeric_response"])
        for k, v in enumerate(VERSIONS):
            ver[i][k] = e["version_responses"].get(v)
    ratings[mid] = arr

    # raw per-prompt responses, kept only where they differ from the parsed Likert phrase
    raw = {}
    for k, v in enumerate(VERSIONS):
        paths = glob.glob(os.path.join(RAW_DIR, "*", f"{mid}_{v}_results.json"))
        if not paths:
            print(f"  WARN no raw file for {mid} {v}")
            continue
        for e in load(paths[0]):
            i = idx_of[(int(e["portrait_id"]), int(e["option_id"]))]
            rr = (e.get("raw_response") or "").strip()
            pr = (e.get("parsed_response") or "").strip()
            if rr and rr.lower() != pr.lower():
                raw[f"{i}:{k}"] = rr[:600]
    with open(os.path.join(OUT, "models", f"{mid.replace(':', '_')}.json"), "w", encoding="utf-8") as f:
        json.dump({"id": mid, "versions": VERSIONS, "scores": ver, "raw": raw}, f, ensure_ascii=False, separators=(",", ":"))

for mid, family, size, vals, var in PAPER_ONLY:
    models_out.append({"id": mid, "name": mid, "family": family, "size": size, "kind": "paper", "reasoning": "distill" in mid,
                       "var": var, "scores": {"pvq": {d: {"c": v} for d, v in zip(PVQ, vals)}}})

print(f"models: {len(models_out)} ({sum(1 for m in models_out if m['kind']=='api')} with item-level data)")
with open(os.path.join(OUT, "models.json"), "w", encoding="utf-8") as f:
    json.dump({"dims": {"pvq": PVQ, "hi": HI, "bfi": BFI}, "models": models_out}, f, ensure_ascii=False, separators=(",", ":"))
with open(os.path.join(OUT, "ratings.json"), "w", encoding="utf-8") as f:
    json.dump({"items": item_index, "models": ratings}, f, ensure_ascii=False, separators=(",", ":"))

for fn in sorted(glob.glob(os.path.join(OUT, "*.json"))):
    print(f"  {os.path.basename(fn):14s} {os.path.getsize(fn)/1024:7.0f} KB")
tot = sum(os.path.getsize(p) for p in glob.glob(os.path.join(OUT, "models", "*.json")))
print(f"  models/*.json  {tot/1024:7.0f} KB total")
