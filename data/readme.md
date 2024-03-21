### ./data

- the data directory can be downloaded [here](https://pan.baidu.com/s/1MrJdxvwTOfwhOwANJpTLtQ). Extraction code: vu76.

- Two datasets are included. There are three samples to show the data structural.

- total samples of CJO22: 1698.

- CAIL2018 dataset can be downloaded [here](https://github.com/china-ai-law-challenge/CAIL2018).

  data columns:
    - fact
    - meta
        - relevant_articles
        - accusation
        - punish_of_money
        - criminals
        - term_of_imprisonment
            - death_penalty
            - imprisonment
            - life_imprisonment
        - pt_cls
    - caseID
    - fact_split
        - zhuguan
        - keguan
        - shiwai

  where 'fact' denotes fact description, 'relevant_articles' denotes related articles, 'accusation' denotes charge, 'imprisonment' denotes term of penalty, 'zhuguan' denotes charge subjective motivation, 'keguan' denotes objective behavior, and 'shiwai' denotes ex post facto circumstance.

- Domain model label sets are specified in __./data/output/meta__.

- Inference: 
1. we organize testcase (in __./data/testset__) and precedent database (in __./data/precedent_database__), where facts are reorganized by LLMs for efficiency. _Reorganized fact_ contains _subjective motivation_, _objective behavior_ and _post facto circumstance_ triplet. 
2. the domain models generate _candidate labels_ by predictive models and identify appropriate _precedents_ by the similarity of the _reorganized fact_ of the given case and cases in the case database. 
3.  _reorganized fact_ of the given case, _top-k labels_ and _precedents_ are concatenated and fed into LLMs for the final prediction through in-context precedent comprehension. 

- Output of domain models and LLMs are stored in __./data/output__. 

- Example: 
1. We have the **_fact description_** of a given case 
"At about 2 a.m. on October 9, 2017, defendant A was on the west side of theroad of the C City, following B who passed by here alone and asking B for \$200 in cash, and B was forced to hand over one mobile phone to A. The price certification...During the trial of the case, defendant A refunded \$200 of illegal gains.". 

2. After the reorganization of LLMs, **_reorganized fact_** consists of **_subjective motivation_** "Defendant A deliberately followed and demanded money from B.", **_objective behavior_** "Defendant A forced B to hand over a mobile phone." and **_post facto circumstance_** "After arrest, A returned $200 of illegal gains.". Then, the domain models generates _candidate labels_ of articles "Article 263, Article 264, Article 267...", charges "Robbery, Theft, Intentional injury..." and terms of penalty "24 to 36 months, 36 to 60 months, 12 to 24 months ". 

3. And we retrieve **_precedents_** by similarity. For each candidate label, we pick one case as the precedent. 
4. Last, **_reorganized fact_** of the given case, **_top-k labels_** and **_precedents_** are concatenated and fed into LLMs for the final prediction through in-context precedent comprehension. 

5. The **_predicted judgment_** of the given case is "Article 263", "Robbery" and "24 to 36 months".