import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.express as px
from tqdm.auto import tqdm
from IPython.display import display,Markdown,HTML
import requests
import json

from .nlp import VaderSentimentClassifier
from .nlp import MetaTweetClassifier,AVAILABLE_TASKS
from .utils import clean_tweet

PREDICTION_COLS = ["climate","emotions","irony","sentiment"] 


def process_raw_data(raw_data,only_cop26hashtag = False):
    # Drop unused column and extract username from metadata
    # Remove all network and personal information
    data = (
        raw_data
        .assign(username = lambda x : x["user"].map(lambda y : y["username"]))
        .drop(columns = ["_type","retweetedTweet","quotedTweet","renderedContent","source","sourceUrl","sourceLabel","user","tcooutlinks","media","retweetedTweet","inReplyToTweetId","inReplyToUser","mentionedUsers","coordinates","place","cashtags"])
    )

    if only_cop26hashtag:
        # Filter on COP26 hashtag
        data = data.loc[data["hashtags"].map(lambda y : ("COP26" in y) if y is not None else False)]

    # Add clean text
    data["clean_bertopic"] = data["content"].map(lambda x : clean_tweet(x,bertopic = True))
    data["clean_sentiment"] = data["content"].map(lambda x : clean_tweet(x,bertopic = False))

    # Clean other columns
    data["likeCat"] = data["likeCount"].map(categorize_count)
    data["retweetCat"] = data["retweetCount"].map(categorize_count)
    data["date"] = pd.to_datetime(data["date"])

    return data

def categorize_count(x):
    if x < 5:
        return "<5"
    elif x < 50:
        return "<50"
    elif x < 250:
        return "<250"
    elif x < 1000:
        return "<1000"
    elif x < 10000:
        return "<10000"
    else:
        return ">10000"


def read_all_datasets(end = 8,start = 1,folder = "../data"):
    data = []
    for i in tqdm(range(start,end+1)):
        filepath = os.path.join(folder,f"RAWDAY{i}.json")
        print(f"... Reading and preparing file {filepath}")
        day_data = open_jsonl_data(filepath,encoding = "utf16")
        day_data = process_raw_data(day_data)
        day_data.to_pickle(os.path.join(folder,f"DATADAY{i}.pkl"))
        data.append(day_data)
    data = pd.concat(data,axis = 0)
    return data



def process_sentiment_vader(data):
    
    # Create Vader classifier
    vader = VaderSentimentClassifier()

    # Predict sentiment and polarity using VADER
    pred = vader.predict(data["clean_sentiment"].tolist())
    pred.index = data.index
    data = pd.concat([data,pred],axis = 1)
    return data


def process_pretrained_classifiers(data,batch_size = None):

    # Create Meta Classifier
    meta = MetaTweetClassifier(tasks = AVAILABLE_TASKS)

    # Make prediction
    pred = meta.predict(data["clean_sentiment"].tolist(),batch_size)
    pred.index = data.index
    data = pd.concat([data,pred],axis = 1)
    return data



def open_jsonl_data(filepath,encoding = "latin1"):

    data = []
    with open(filepath,"r",encoding = encoding) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    
    return pd.DataFrame(data)

