poetry run snscrape --jsonl twitter-search "#COP26 since:2021-11-01 until:2021-11-02" > ./data/RAW_DAY1.json
poetry run snscrape --jsonl twitter-search "#COP26 since:2021-11-02 until:2021-11-03" > ./data/RAW_DAY2.json
poetry run snscrape --jsonl twitter-search "#COP26 since:2021-11-03 until:2021-11-04" > ./data/RAW_DAY3.json
poetry run snscrape --jsonl twitter-search "#COP26 since:2021-11-04 until:2021-11-05" > ./data/RAWDAY4.json

pause