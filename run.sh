python llm_api.py --model "text-davinci-003"
python llm_api.py --model "text-davinci-003" --small_model CNN --task charge --use_split_fact True
python llm_api.py --model "text-davinci-003" --small_model CNN --task article --use_split_fact True --dataset "cjo22"
python llm_api.py --model "text-davinci-003" --small_model CNN --task penalty --dataset "cail18"
python llm_api.py --model "gpt-3.5-turbo" --small_model CNN --task penalty --use_split_fact True --dataset "cail18"