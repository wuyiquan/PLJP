import json
import re
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support, classification_report

data = []

testset_path = "data/testset/cail18/testset.json"
with open(testset_path, encoding="utf8") as f:
    for line in f.readlines():
        obj = json.loads(line)
        data.append(obj)

results = []
resp_file = "data/output/llm_out/cail18/CNN/penalty/3shot/gpt-3.5-turbo.json"

with open(resp_file, encoding="utf8") as f:
    for line in f.readlines():
        obj = json.loads(line)
        results.append(obj)

y_true = [case["meta"]["term_of_imprisonment"]["imprisonment"] for case in data]

pt_cls2str = ["其他", "六个月以下", "六到九个月", "九个月到一年", "一到两年", "二到三年", "三到五年", "五到七年", "七到十年", "十年以上"]

def get_pt_cls(pt):
    if pt > 10 * 12:
        pt_cls = 9
    elif pt > 7 * 12:
        pt_cls = 8
    elif pt > 5 * 12:
        pt_cls = 7
    elif pt > 3 * 12:
        pt_cls = 6
    elif pt > 2 * 12:
        pt_cls = 5
    elif pt > 1 * 12:
        pt_cls = 4
    elif pt > 9:
        pt_cls = 3
    elif pt > 6:
        pt_cls = 2
    elif pt > 0:
        pt_cls = 1
    else:
        pt_cls = 0
    return pt_cls

y_true = [pt_cls2str[get_pt_cls(p)] for p in y_true]

y_pred = []

for obj in results:
    # text = obj["choices"][0]["text"] # dav
    text = obj["choices"][0]["message"]['content'] # turbo
    pred_p = ""
    for p_str in pt_cls2str:
        if p_str in text:
            pred_p = p_str
    y_pred.append(pred_p)

acc, _, _, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
map, mar, maf, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

acc = round(acc, 6) * 100
map = round(map, 6) * 100
mar = round(mar, 6) * 100
maf = round(maf, 6) * 100

print(f"acc:{acc}, map:{map}, mar:{mar}, maf:{maf}")