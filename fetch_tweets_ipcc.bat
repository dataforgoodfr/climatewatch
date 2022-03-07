@REM chcp 65001
@REM https://github.com/JustAnotherArchivist/snscrape/issues/122

poetry run snscrape --jsonl twitter-search "#IPCC since:2022-02-21 until:2022-03-07" > ./data/IPCC2.json
poetry run snscrape --jsonl twitter-search "#GIEC since:2022-02-21 until:2022-03-07" > ./data/GIEC2.json
poetry run snscrape --jsonl twitter-search "#IPCC since:2021-08-02 until:2021-08-16" > ./data/IPCC1.json
poetry run snscrape --jsonl twitter-search "#GIEC since:2021-08-02 until:2021-08-16" > ./data/GIEC1.json

pause