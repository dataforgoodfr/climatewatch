poetry run snscrape --jsonl twitter-search "#IPCC since:2022-02-21 until:2022-03-01" > ./data/IPCC1.json
poetry run snscrape --jsonl twitter-search "#GIEC since:2022-02-21 until:2022-03-01" > ./data/GIEC1.json

pause