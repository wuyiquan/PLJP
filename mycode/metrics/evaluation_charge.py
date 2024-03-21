import json
import re
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
from collections import Counter

data = []
testset_path = "data/testset/cail18/testset.json"

with open(testset_path, encoding="utf8") as f:
    for line in f.readlines():
        obj = json.loads(line)
        data.append(obj)

results = []
resp_file = "data/output/llm_out/cail18/CNN/charge/3shot/gpt-3.5-turbo.json"

with open(resp_file, encoding="utf8") as f:
    for line in f.readlines():
        obj = json.loads(line)
        results.append(obj)

charges = [case["meta"]["accusation"][0] for case in data]
c_set = set(charges)

laic_charge_match = {
    '非法生产、买卖、运输制毒物品、走私制毒物品': [
        ('生产', '制毒物品'), ('买卖', '制毒物品'), ('运输', '制毒物品'), ('走私', '制毒物品')
    ],
    '非法经营': [
        ('非法', '经营'),
    ],
    '非法转让、倒卖土地使用权': [
        ('转让', '土地'), ('倒卖', '土地')
    ]
}
cail_charge_match = {
    '重大劳动安全事故': [
        ('劳动', '事故'),
    ],
    '容留他人吸毒': [
        ('容留', '吸毒'),
    ],
    '非法种植毒品原植物': [
        ('种', '毒品'),
        ('植', '毒品'),
    ],
    '盗伐林木': [
        ('盗',' 伐', '林'),
        ('盗',' 伐', '木'),
        ('偷',' 伐', '林'),
        ('偷',' 伐', '木'),
    ],
    '故意杀人': [
        ('故意', '杀人'),
    ],
    '交通肇事': [
        ('肇事'),
    ],
    '污染环境': [
        ('污染', ),
    ],
    '强奸': [
        ('强奸'),
    ],
    '合同诈骗': [
        ('合同', '诈骗'),
    ],
    '生产、销售不符合安全标准的食品': [
        ('产', '不', '安全', '食'),
        ('售', '不', '安全', '食'),
        ('卖', '不', '安全', '食'),
    ],
    '强制猥亵、侮辱妇女': [
        ('猥亵', '女'), ('侮辱', '女'),
    ],
    '妨害信用卡管理': [
        ('信用卡', '管理'),
    ],
    '赌博': [
        ('赌博'),
    ],
    '生产、销售伪劣产品': [
        ('产', '伪', '品'), ('产', '劣', '品'),
        ('售', '伪', '品'), ('售', '劣', '品'),
        ('卖', '伪', '品'), ('卖', '劣', '品'),
    ],
    '妨害公务': [
        ('妨', '公务'),
    ],
    '职务侵占': [
        ('职务', '侵'), ('职务', '占'),
    ],
    '非法采矿': [
        ('采矿'),
    ],
    '滥用职权': [
        ('滥用', '职'), ('滥用', '权'),
    ],
    '破坏广播电视设施、公用电信设施': [
        ('破坏', '广播'), ('破坏', '电信'), 
    ],
    '放火': [
        ('放火'),
    ],
    '伪造、变造、买卖国家机关公文、证件、印章': [
        ('伪造', '印章'),
        ('伪造', '公章'),
    ],
    '非法采伐、毁坏国家重点保护植物': [
        ('采', '保护', '植'),
        ('伐', '保护', '植'),
        ('毁', '保护', '植'),
        ('坏', '保护', '植'),
    ],    
    '开设赌场': [
        ('开', '赌场'), ('开', '设'),
    ],
    '生产、销售假药': [
        ('产', '假', '药'),('售', '假', '药'),('卖', '假', '药'),
    ],   
    '非法吸收公众存款': [
        ('吸', '公众', '款'),
        ('收', '公众', '款'),
    ],
    '玩忽职守': [
        ('忽', '职守'),
    ],   
}
total_charge_match = {**laic_charge_match, **cail_charge_match} 

def get_similar_charge(text, charge_similar):
    contain_charge_set = set()
    for c, precedent_lst in charge_similar.items():
        for precedent_words in precedent_lst:
            if sum([w in text for w in precedent_words]) == len(precedent_words):
                contain_charge_set.add(c)
    if len(contain_charge_set) == 0:
        return "#"
    charge_set = list(contain_charge_set)
    charge_set = sorted(charge_set, key = lambda x: len(x), reverse = True)
    return charge_set[0]

y_true = charges
self_consistency_pred = []
for runs in range(len(results[0]["choices"])):
    y_pred = []
    for output in results:
        # text = output["choices"][runs]["text"].replace("\n", "") # dav
        text = output["choices"][0]["message"]['content'] # turbo
        cur_c = "#"
        contain_charge_set = set()
        for c in c_set:
            if c in text:
                contain_charge_set.add(c)
        if len(contain_charge_set) == 1:
            cur_c = list(contain_charge_set)[0]
        if cur_c != "#":
            y_pred.append(cur_c)
            continue

        cur_c = get_similar_charge(text, total_charge_match)
        y_pred.append(cur_c)
        
    self_consistency_pred.append(y_pred)

voted_y_pred = []

for pred_res in self_consistency_pred:
    cntr = Counter()
    for pred in pred_res:
        if pred == '#':
            continue
        cntr[pred] += 1

    if len(cntr) == 0:
        voted_y_pred.append("#")
        continue
        
    most_cnt = cntr.most_common()[0][1]
    candidate_c = [t[0] for t in cntr.most_common() if t[1] == most_cnt]
    candidate_c = sorted(candidate_c, key = lambda x: len(x), reverse = True)
    voted_y_pred.append(candidate_c[0])

acc, _, _, _ = precision_recall_fscore_support(y_true, y_pred, average = "micro")
mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, average = "macro")
acc = round(acc, 6) * 100
mp = round(mp, 6) * 100
mr = round(mr, 6) * 100
mf = round(mf, 6) * 100

print(f"acc:{acc}, mp:{mp}, mr:{mr}, mf:{mf}")