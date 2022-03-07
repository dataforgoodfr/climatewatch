import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.express as px
from tqdm.auto import tqdm
from IPython.display import display,Markdown,HTML
import requests

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

from .nlp import VaderSentimentClassifier
from .nlp import MetaTweetClassifier,AVAILABLE_TASKS

from .utils import clean_tweet
from .utils import search_keywords,search_keywords_in_df

PREDICTION_COLS = ["climate","emotions","irony","sentiment"] 



class Tweet(object):
    def __init__(self, s, embed_str=False):
        """https://github.com/jupyter/notebook/issues/2790
        """
        if not embed_str:
            # Use Twitter's oEmbed API
            # https://dev.twitter.com/web/embedded-tweets
            api = 'https://publish.twitter.com/oembed?url={}'.format(s)
            response = requests.get(api)
            self.text = response.json()["html"]
        else:
            self.text = s

    def _repr_html_(self):
        return self.text



class TweetsDataset:
    def __init__(self,pkl_path = None,data = None,only_cop26hashtag = False,process = False,verbose = True,batch_size = None):

        if pkl_path is not None:

            if isinstance(pkl_path,list):
                self.data = [pd.read_pickle(x) for x in tqdm(pkl_path)]
                self.data = pd.concat(self.data,axis = 0,ignore_index = True) 
            else:
                self.data = self.load_pkl(pkl_path)

        elif data is not None:
            self.data = data.copy()
        else:
            raise Exception("No input file provided")


        self.data["date"] = pd.to_datetime(self.data["date"])


    def search(self,keywords):
        return search_keywords_in_df(keywords,self.data,"clean_bertopic")


    def load_pkl(self,path):
        data = pd.read_pickle(path)
        return data


    def save_pkl(self,path):
        self.data.to_pickle(path)

    def most_liked(self,n = 20):
        most_liked = self.data.sort_values("likeCount",ascending = False).head(n)
        return most_liked

    def show_languages(self,n = 20,kind = "treemap"):
        count = self.data["lang"].value_counts().head(n)

        if kind == "treemap":
            fig = px.treemap(count.reset_index(),path = ["index"],values = "lang")
            return fig

        elif kind == "bar":

            count.plot(kind = "bar",figsize = (15,4),logy = True,title = "Tweets by language (log axis)");
            plt.show()


    def show_hashtags(self,n = 50,kind = "treemap"):
        count = self.data["hashtags"].explode().dropna().str.upper().value_counts().head(n).reset_index()
        count = count.loc[count["index"] != "COP26"]
        fig = px.treemap(count,path = ["index"],values = "hashtags")
        return fig


    def show_evolution(self,period = "30min"):

        count = (pd.DataFrame(pd.to_datetime(self.data["date"]),columns = ["date"])
            .assign(count = lambda x:1)
            .set_index("date")
            .resample(period).sum()
        )

        fig = px.line(count.reset_index(),x = "date",y = "count")
        return fig


    def show(self,data,embed = True):

        if embed:
            self.show_embed(data)
        else:
        
            md = ["##### Tweets"]
            
            for i,row in data.iterrows():
                
                md.append(f"From **{row['username']}** - *Likes {row['likeCount']}* - Date {row['date']}\n> {row['clean_bertopic']}")
            
            md = "\n\n".join(md)
            display(Markdown(md))
    
    def show_embed(self,data):

        if isinstance(data,int):
            t = Tweet(self.data["url"].iloc[data])
            return t

        else:
            for i,row in data.iterrows():

                t = Tweet(row["url"])
                display(HTML(t.text))



    def show_emotions_treemap(self,emotion,hashtags = None,top = 20):
        count = self.get_tweets_top_hashtags(hashtags = hashtags,top = top,remove = remove)

        # Emotions aggregation
        agg = (count
            .groupby([pd.Grouper(freq = period),"hashtags"])
            .agg({k:"mean" for k in emotions})
            .reset_index()
            .melt(id_vars = ["date","hashtags"])
        )


    def show_treemap_emotions_hashtags(self,emotion,hashtags = None,top = 20):
        count = self.get_tweets_top_hashtags(hashtags = hashtags,top = top)
        agg = count.assign(count = lambda x : 1).groupby(["hashtags",f"pred_{emotion}"])["count"].sum().reset_index()
        fig = px.treemap(agg,path = [f"pred_{emotion}","hashtags"],values = "count")
        return fig



    def get_tweets_top_hashtags(self,hashtags = None,top = 20,remove = ["COP26"]):

        if hashtags is None:
            hashtags = self.data["hashtags"].explode().str.upper().value_counts().drop(remove).head(top).index.tolist()
        else:
            if not isinstance(hashtags,list): hashtags = [hashtags]

        hashtags = [x.upper() for x in hashtags]

        # Explode and prepare count
        count = (self.data.set_index("date").explode("hashtags")
            .assign(count = lambda x : 1)
            .assign(hashtags = lambda x : x["hashtags"].str.upper())
        )

        # Filter on top hashtags 
        count = count.loc[count["hashtags"].str.upper().isin(hashtags)]
        return count


    def show_strongest_emotions(self,emotion,top = 20,return_data = False,embed = True):
        strongest = self.data.sort_values(emotion,ascending = False).head(top)
        if return_data:
            return strongest
        else:
            self.show(strongest,embed)


    def make_dataset(self,query):

        # Get dataset using Pandas query
        data = self.data.query(query)

        # Create TweetsDataset
        dataset = TweetsDataset(data = data)
        return dataset




    def show_evolution_emotions_by_hashtags(self,emotions,hashtags = None,period = "30min",top = 20,remove = ["COP26"]):
        count = self.get_tweets_top_hashtags(hashtags = hashtags,top = top,remove = remove)

        # Emotions aggregation
        agg = (count
            .groupby([pd.Grouper(freq = period),"hashtags"])
            .agg({k:"mean" for k in emotions})
            .reset_index()
            .melt(id_vars = ["date","hashtags"])
        )

        fig = px.line(agg,x = "date",y = "value",color = "hashtags",facet_row = "variable")
        return fig


    def show_evolution_by_hashtags(self,hashtags = None,period = "30min",top = 20,remove = ["COP26"]):

        if hashtags is None:
            hashtags = self.data["hashtags"].explode().value_counts().head(top).index.tolist()
        else:
            if not isinstance(hashtags,list): hashtags = [hashtags]

        hashtags = [x.upper() for x in hashtags]
        hashtags = [x for x in hashtags if x not in remove]


        # Explode and prepare count
        count = (self.data.set_index("date").explode("hashtags")
            .assign(count = lambda x : 1)
            .assign(hashtags = lambda x : x["hashtags"].str.upper())
        )

        # Filter on top hashtags 
        count = count.loc[count["hashtags"].str.upper().isin(hashtags)]

        # Groupby with time aggregator
        count = (
            count
            .groupby([pd.Grouper(freq = "1H"),"hashtags"])
            ["count"].sum()
        )

        fig = px.line(count.reset_index(),x = "date",y = "count",color = "hashtags")
        return fig


    def show_evolution_emotions(self,pred,period = "1H",return_data = False,normalized = True,area = False):

        if isinstance(pred,list):
            count = pd.concat([
                self.show_evolution_emotions(x,period,return_data = True) for x in pred],
            axis = 0)

            if return_data:
                return count

        else:
            assert pred in PREDICTION_COLS
            col = f"pred_{pred}"
            count = (
                self.data
                .set_index("date")
                .assign(count = lambda x : 1)
                .groupby([pd.Grouper(freq = period),col])
                ["count"].sum()
                .reset_index()
                .rename(columns = {col:"value"})
                .assign(key=lambda y:pred)
            )
            if return_data:
                return count

        facet_row = "key" if isinstance(pred,list) else None

        if normalized or area:
            fig = px.area(count,x = "date",y = "count",color = "value",facet_row = "key",groupnorm="percent" if normalized else None)
        else:
            fig = px.line(count,x = "date",y = "count",color = "value",facet_row = "key")

        return fig



    def get_columns_probas(self,cat):
        return [x for x in self.data.columns if x.startswith(f"{cat.lower()}_")]


    def add_pred_columns(self,cats):

        for cat in cats:
            columns = self.get_columns_probas(cat)
            print(columns)
            self.data[f"pred_{cat}"] = self.data[columns].idxmax(axis = 1).map(lambda x : x.split("_")[1])


    def fit_topic_modeling(self,min_topic_size = 20,ngram_range = (1,3),nr_topics = "auto",seed_topic_list = None,low_memory = False,**kwargs):
        """Topic modeling with BERTopic
        - https://github.com/MaartenGr/BERTopic
        - Full tutorial https://colab.research.google.com/drive/1FieRA9fLdkQEGDIMYl0I3MCjSUKVF8C-?usp=sharing#scrollTo=y_eHBI1jSb6i
        - DTW https://colab.research.google.com/drive/1un8ooI-7ZNlRoK0maVkYhmNRl0XGK88f?usp=sharing
        """
        
        # Instantiate Vectorizer and BERTopic models
        print("... Create vectorizer model")
        self.vectorizer_model = CountVectorizer(ngram_range=ngram_range, stop_words="english")

        print("... Create BERTopic model")
        # Parameters documentation at https://maartengr.github.io/BERTopic/api/bertopic.html
        self.topic_model = BERTopic(
            min_topic_size=min_topic_size, 
            vectorizer_model = self.vectorizer_model, 
            verbose=True,
            top_n_words=10,
            nr_topics=nr_topics,
            seed_topic_list=seed_topic_list,
            calculate_probabilities = False,
            low_memory=low_memory,
            **kwargs,
        )

        # Fit models
        print("... Fit BERTopic model")
        topics, probs = self.topic_model.fit_transform(self.data["clean_bertopic"].tolist())

        # Add results
        self.data["topic_number"] = topics
        self.data["topic_proba"] = probs

        # Get topic summary
        self.topic_summary = self.topic_model.get_topic_info()

    def understand_topic(self,i):
        topic_nr = self.topic_summary.iloc[i]["Topic"] # select a frequent topic
        return self.topic_model.get_topic(topic_nr)

