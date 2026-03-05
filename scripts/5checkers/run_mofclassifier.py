from MOFClassifier import CLscore
import glob, json


all_structures = [stuc for stuc in glob.glob("../../dataset/cifs/*cif")[:]]
results = CLscore.predict_batch(root_cifs=all_structures, model="core", batch_size=16)

clscore_pre = {}
for i in range(len(all_structures)):
    clscore_pre[results[i][0]] = 1 if results[i][2] > 0.5 else 0

with open("../../dataset/5checkers/mofclass_pre.json", "w") as f:
    json.dump(clscore_pre, f, indent=2)