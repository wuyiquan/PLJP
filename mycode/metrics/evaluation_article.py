import json
import re
import cn2an
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import jaccard_score


data = []

testset_path = "data/testset/cail18/testset.json"

with open(testset_path, encoding="utf8") as f:
    for line in f.readlines():
        obj = json.loads(line)
        data.append(obj)

results = []
resp_file = "data/output/llm_out/cail18/CNN/article/3shot/gpt-3.5-turbo.json"

with open(resp_file, encoding="utf8") as f:
    for i, line in enumerate(f.readlines()):
        obj = json.loads(line)
        results.append(obj)

y_true = []

for case in data:
    ar = int(max(case["meta"]["relevant_articles"]))
    y_true.append(ar)


y_pred = []
count = 0
for obj in results:
    count += 1
    # resp = obj["choices"][0]["text"] # dav
    resp = obj["choices"][0]["message"]['content'] # turbo
    res = re.findall("第(.*?)条", resp)
    ars = [0]
    for i in res:
        if i.isdigit():
            i = int(i)
            ars.append(i)
        else:
            try:
                i = cn2an.cn2an(i, mode="smart")
                ars.append(i)
            except:
                ars.append(0)
    y_pred.append(max(ars))

acc, _, _, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
map, mar, maf, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

acc = round(acc, 6) * 100
map = round(map, 6) * 100
mar = round(mar, 6) * 100
maf = round(maf, 6) * 100

print(f"acc:{acc}, map:{map}, mar:{mar}, maf:{maf}")