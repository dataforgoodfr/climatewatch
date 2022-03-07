import os
import json
import re
import pandas as pd
from flashtext import KeywordProcessor
from codecarbon import EmissionsTracker


def scrape_twitter(hashtag,max_results = 10000,folder = "../data/",start_date = None,end_date = None,show = True):
    """
    Start and end date must be in the format "2020-03-04"
    """
    #     https://github.com/JustAnotherArchivist/snscrape/issues/81
    date = str(pd.to_datetime("today"))[:19].replace(" ","_").replace(":","-")
    path = os.path.join(folder,f"{date}_{max_results if max_results is not None else 'ALL'}_{hashtag}.json")
    cmd = f'poetry run snscrape --jsonl'
    
    # Add max results filter
    if max_results is not None:
        cmd += f" --max-results {max_results}"
    
    # Prepare search terms
    search = [hashtag]
    if start_date is not None:
        search.append(f'since:{start_date}')
    if end_date is not None:
        search.append(f'until:{end_date}')
    search = " ".join(search)

    # Add search command
    cmd += f' twitter-hashtag "{search}" > {path}'

    # Run command
    if show:
        print(cmd)
    else:
        print(f"... Running extraction with {cmd}")
        os.system(cmd)



def clean_tweet(x,max_token_length = 30,bertopic = True):


    x = re.sub(r"http\S+", "", x)
    x = re.sub(r"[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b\S+","",x)
    x = re.sub(r"bit.ly\S+", "", x)
    x = re.sub(r"twitter.com\S+", "", x)
    x = " ".join(filter(lambda y:len(y) < max_token_length, x.split()))
    x = x.replace("#","").replace("\"","").replace("&amp;","&")
    
    if bertopic:
        x = x.replace("@","")
    else:
        x = remove_usernames(x)
    
    # Specific for BERTopic preprocessing
#     x = " ".join(re.sub("[^a-zA-Z]+", " ", x).split()).lower()
    return x

# Preprocess text (username and link placeholders)
def remove_usernames(text):
    new_text = []
 
 
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        new_text.append(t)
    return " ".join(new_text)



def search_keywords(keywords,text):
    if not isinstance(keywords,list): keywords = [keywords]
    keyword_processor = KeywordProcessor(case_sensitive=False)
    keyword_processor.add_keyword(*keywords)
    search = keyword_processor.extract_keywords(text)
    return len(search) > 0


def search_keywords_in_df(keywords,df,column):
    return df.loc[df[column].map(lambda x : search_keywords(keywords,x))]

def read_json(filepath):
    return json.loads(open(filepath,"r").read())


def save_json(d,filepath):
    with open(filepath,"w") as file:
        file.write(json.dumps(d))

class CodeCarbon:
    def __enter__(self):
        self.tracker = EmissionsTracker()
        self.tracker.start()
        print("[INFO] Measuring carbon emissions with CodeCarbon")
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        self.tracker.stop()