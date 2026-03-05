import os, json
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODELS = {
        "llama3":   {"name": "meta-llama/Meta-Llama-3-8B-Instruct",             "max_length": 7936},
        "qwen3":    {"name": "Qwen/Qwen3-4B-Instruct-2507",                     "max_length": 7936},
        "gemma3":   {"name": "google/gemma-3-4b-it",                            "max_length": 7936},  "deepseek": {"name": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",     "max_length": 3840},
        }

DATASETS = [
            "cif_p1",
            "composition",
            "slices",
            "local_env",
            "atom_sequences_plusplus",
            "crystal_text_llm",
            "zmatrix",
            "robocry",
            "struc2str",
            "mof2text",
            ]

OUT_DIR = "./"

dataset_label = "../dataset/label/manual_Tom.json"
all_data = json.load(open(dataset_label, "r", encoding="utf-8"))

def get_embed(texts, tok, mdl, max_length):
    inputs = tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = mdl(**inputs, output_hidden_states=False, return_dict=True)

    mask = inputs["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.dtype)
    emb = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    emb = F.normalize(emb, p=2, dim=1)
    return emb

device = "cuda" if torch.cuda.is_available() else "cpu"

for mtag, mcfg in MODELS.items():
    model_name = mcfg["name"]
    max_len = mcfg["max_length"]

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    mdl = mdl.to(device).eval()

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl.config.pad_token_id = tok.pad_token_id

    for dset in DATASETS[:]:
        embs_good = {}
        embs_bad = {}
        
        for item in tqdm(all_data, desc=f"{mtag}:{dset}"):
            sid = item
            if sid is None:
                continue

            if int(all_data[sid].get("has_error", -1)) == 0:
                if dset not in ["mof2text", "struc2str"]:
                    text = json.load(open("../dataset/descriptor/"+sid+".json"))[dset]
                    if text is None:
                        continue
                    emb = get_embed([text], tok, mdl, max_len)
                    vec = emb.detach().cpu().numpy().tolist()[0]
                    embs_good[sid] = vec
                else:
                    if dset == "mof2text":
                        text = json.load(open("../dataset/mof2text/"+sid+".json"))["MOF2Text"]
                        if text is None:
                           continue
                        emb = get_embed([text], tok, mdl, max_len)
                        vec = emb.detach().cpu().numpy().tolist()[0]
                        embs_good[sid] = vec
                    elif dset == "struc2str":
                        text = json.load(open("../dataset/struc2str/"+sid+".json"))["MOF2Text"]
                        if text is None:
                            continue
                        emb = get_embed([text], tok, mdl, max_len)
                        vec = emb.detach().cpu().numpy().tolist()[0]
                        embs_good[sid] = vec
            elif int(all_data[sid].get("has_error", -1)) == 1:
                if dset not in ["mof2text", "struc2str"]:
                    text = json.load(open("../dataset/descriptor/"+sid+".json"))[dset]
                    if text is None:
                        continue
                    emb = get_embed([text], tok, mdl, max_len)
                    vec = emb.detach().cpu().numpy().tolist()[0]
                    embs_bad[sid] = vec
                else:
                    if dset == "mof2text":
                        text = json.load(open("../dataset/mof2text/"+sid+".json"))["MOF2Text"]
                        if text is None:
                           continue
                        emb = get_embed([text], tok, mdl, max_len)
                        vec = emb.detach().cpu().numpy().tolist()[0]
                        embs_bad[sid] = vec
                    elif dset == "struc2str":
                        text = json.load(open("../dataset/struc2str/"+sid+".json"))["MOF2Text"]
                        if text is None:
                            continue
                        emb = get_embed([text], tok, mdl, max_len)
                        vec = emb.detach().cpu().numpy().tolist()[0]
                        embs_bad[sid] = vec

        out_good = os.path.join(OUT_DIR, f"{dset}_embs_good_{mtag}.json")
        with open(out_good, "w", encoding="utf-8") as f:
            json.dump(embs_good, f, ensure_ascii=False, indent=2)

        out_bad = os.path.join(OUT_DIR, f"{dset}_embs_bad_{mtag}.json")
        with open(out_bad, "w", encoding="utf-8") as f:
            json.dump(embs_bad, f, ensure_ascii=False, indent=2)