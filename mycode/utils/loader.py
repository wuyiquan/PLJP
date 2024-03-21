import json
import cn2an
from tqdm import tqdm
import re

def truncate_text(text, max_len=1024):
    if text == "":
        return ""
    text = text.replace("。", "。 ").replace("，", "， ").replace("；", "； ").replace("、", "、 ")
    text = text.split()
    if "经审理" in text[0]:
        text = text[1:]
    prefix = []
    postfix = []
    n = len(text)
    i, j = 0, n-1
    while i < n:
        sent = text[i]
        prefix = prefix + [sent]
        if len(" ".join(prefix)) >= max_len // 2:
            break
        i += 1
    while j > i:
        sent = text[j]
        postfix = [sent] + postfix
        if len(" ".join(postfix)) >= max_len // 2:
            break
        j -= 1
    ret_text = prefix + postfix
    return "".join(ret_text)


def truncate_text_total(text, max_len):
    text = truncate_text(text, max_len, delimiter="。")
    text = truncate_text(text, max_len, delimiter="，")
    text = truncate_text(text, max_len, delimiter="、")
    text = truncate_text(text, max_len, delimiter="；")
    return text


def load_precedent(pool_path, precedent_idx_path):
    pool = []
    precedents = []

    with open(pool_path, encoding="utf-8") as f:
        for line in f.readlines():
            obj = json.loads(line)
            pool.append(obj)

    precedent_idxs = json.load(open(precedent_idx_path))

    for precedent_idx in tqdm(precedent_idxs, desc="load similar case"):
        sc = [pool[idx] for idx in precedent_idx]
        precedents.append(sc)
    return precedents


def ar_idx2text(aid, index_type = "str"):
    if index_type == "str":
        astr = cn2an.an2cn(aid)
    elif index_type == "num":
        astr = str(aid)
    astr = "第" + astr + "条"
    return astr


def load_law_articles(path="data/output/meta/laws.txt"):
    laws = []
    with open(path, encoding="utf8") as f:
        for line in f.readlines():
            laws.append(line.strip())
    aid = 0
    mp = {}

    for line in laws:
        astr = ar_idx2text(aid)
        arr = line.split()

        if len(arr) and arr[0] == ar_idx2text(aid + 1):
            astr = ar_idx2text(aid + 1, index_type="str")
            mp[astr] = line
            aid += 1
        else:
            mp[astr] += line
    mp = {
        k: v.replace(k, "").strip() for k, v in mp.items()
    }
    return mp


def load_topk_option(path):
    topk_label_option = json.load(open(path, encoding="utf8"))
    return topk_label_option


def load_retrieved_articles(path, index_type = "str"):
    retrieved_ar_lst_idx = json.load(open(path, encoding="utf8"))
    law_mp = load_law_articles()

    retrieved_ar_lst = []

    for lst in tqdm(retrieved_ar_lst_idx, desc="load article content"):
        cur_ar_lst = []
        for aid in lst:
            astr = ar_idx2text(aid, index_type = "str")
            anum = ar_idx2text(aid, index_type = "num")
            cur_ar_lst.append(f"{anum}：{law_mp[astr]}")
        retrieved_ar_lst.append(cur_ar_lst)
    return retrieved_ar_lst


def load_step1_resp(path):
    c_rats = []
    with open(path, encoding="utf8") as f:
        for line in f.readlines():
            obj = json.loads(line)
            c_rat = obj["choices"][0]["text"]
            c_rats.append(c_rat)
    return c_rats

def load_predicted_article_content(predicted_article):
    law_mp = load_law_articles()
    if re.findall("\d+",predicted_article):
        predicted_article = re.findall("\d+",predicted_article)[0]
    else:
        predicted_article = "1"
    if int(predicted_article) < 0 or int(predicted_article) > 451:
        return ""
    astr = ar_idx2text(predicted_article, index_type = "str")
    predicted_article_content = law_mp[astr]
    return predicted_article_content    
