import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import requests
import textwrap
import networkx as nx
from tqdm.auto import tqdm
from IPython.display import display,Markdown,HTML
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNEfrom sklearn.cluster import DBSCAN,KMeans,MeanShift
from bertopic import BERTopic

# from .nlp import VaderSentimentClassifier
# from .nlp import MetaTweetClassifier,AVAILABLE_TASKS
# PREDICTION_COLS = ["climate","emotions","irony","sentiment"] 

from .utils import clean_tweet
from .utils import search_keywords,search_keywords_in_df
from .nlp.stopwords import stopwords_fr,stopwords_en
from .nlp.stopwords import stopwords
from .data import process_raw_data,process_sentiment_vader
from .data import open_jsonl_data

# CONFIG AND THEMES
COLOR_DFG1_BLUE1 = "rgb(37, 147, 156)"
COLOR_DFG1_BLUE2 = "rgb(53, 196, 215)"
COLOR_DFG1_ORANGE = "rgb(252, 163, 17)"
COLOR_DFG1_GREEN = "rgb(204, 224, 61)"
COLOR_SEQUENCE = [COLOR_DFG1_BLUE1,COLOR_DFG1_BLUE2,COLOR_DFG1_ORANGE,COLOR_DFG1_GREEN] + px.colors.qualitative.Prism
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = COLOR_SEQUENCE

DFG_THEME = go.layout.Template()
DFG_THEME.layout.treemapcolorway = COLOR_SEQUENCE
DFG_THEME.layout.sunburstcolorway = COLOR_SEQUENCE
DFG_THEME.layout.font = {"family":"Source Sans Pro"}
px.defaults.template = DFG_THEME




def process_json_files(json_files,folder,vader = False,encoding = "utf8"):
        
    data = []

    for json_file in json_files:

        data_file = open_jsonl_data(os.path.join(folder,json_file),encoding = encoding)
        data_file = process_raw_data(data_file)
        if vader:
            data_file = process_sentiment_vader(data_file)
        data_file["source_file"] = json_file

        data.append(data_file)

    data = pd.concat(data,ignore_index = True)
    data["date_day"] = pd.to_datetime(data["date"].dt.date)
    data["date"] = pd.to_datetime(data["date"])
    data["count"] = 1
    data["reach"] = data["likeCount"] + data["replyCount"] + data["retweetCount"] + data["quoteCount"]
    data["is_reply"] = data["inReplyToUser"].map(lambda x : x is not None)
    data["reply_username"] = data["inReplyToUser"].map(lambda x : x["username"] if x is not None else np.nan)

    return data

def add_edge(G,row):
    G.add_node(row["user"]["displayname"],**row["user"])
    
    if "inReplyToUser" in row and row["inReplyToUser"] is not None:
        user = User(row["inReplyToUser"])
        try:
            G.add_node(user.get_name(),**user.get_props())
            G.add_edge(row["user"]["displayname"],user.get_name(),label = "Reply")
        except:
            print(row["inReplyToUser"])
            raise Exception("yo")
    
    if "mentionedUsers" in row and row["mentionedUsers"] is not None:
        
        for user_props in row["mentionedUsers"]:
            user = User(user_props)
            G.add_node(user.get_name(),**user.get_props())
            G.add_edge(row["user"]["displayname"],user.get_name(),label = "Mention")


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

class User:
    def __init__(self,props):
        self.props = props
        
    def get_name(self):
        
        name = self.props.get("displayname")
        if name is not None:
            return name
        
        name = self.props.get("username")
        return name
        
    
    def get_props(self):
#         return {"followersCount":self.props.get("followersCount")}
        return self.props


class TweetsDataset:
    def __init__(self,pkl_path = None,data = None,process = False):

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

        # Compute helper variables
        self.data["date"] = pd.to_datetime(self.data["date"])
        self.data["count"] = 1
        self.data["reach"] = self.data["likeCount"] + self.data["replyCount"] + self.data["retweetCount"] + self.data["quoteCount"]

    def __repr__(self):
        return f"TweetsDataset(n_tweets={len(self.data)})"

    def get_all_hashtags(self,top = 100):
        hashtags = list(self.data["hashtags"].explode().dropna().str.upper().value_counts().head(top).index)
        return hashtags
        

    def search(self,text):
        return self.data.loc[self.data["content"].str.lower().map(lambda x : text.lower() in x)]

    def load_pkl(self,path):
        data = pd.read_pickle(path)
        return data


    def save_pkl(self,path):
        self.data.to_pickle(path)


    def query_hashtags(self,hashtags):

        if not isinstance(hashtags,list): hashtags = [hashtags]
        hashtags = list(map(lambda x : x.upper(),hashtags))

        t = self.data.explode("hashtags").dropna(subset = ["hashtags"])
        t["hashtags"] = t["hashtags"].str.upper() 

        data = self.data.loc[t.loc[t["hashtags"].isin(hashtags)].index.drop_duplicates().tolist()]
        
        # Create TweetsDataset
        dataset = TweetsDataset(data = data)
        return dataset

    def query(self,query):

        # Get dataset using Pandas query
        data = self.data.query(query)

        # Create TweetsDataset
        dataset = TweetsDataset(data = data)
        return dataset

    def query_user(self,user):
        data = self.data.loc[self.data["username"].str.lower() == user.lower()]
        return TweetsDataset(data = data)

    def query_date(self,date = None,min_date = None,max_date = None):

        if date is not None:
            min_date = date
            max_date = str(pd.to_datetime(date) + pd.DateOffset(days = 1))[:10]
            
        if min_date is not None and max_date is not None:
            data = self.data.loc[(self.data["date"] >= min_date) & (self.data["date"] <= max_date)]
        elif min_date is not None:
            data = self.data.loc[(self.data["date"] >= min_date)]
        elif max_date is not None:
            data = self.data.loc[(self.data["date"] <= max_date)]
        else:
            raise Exception("You have to provide at least one date")
        
        return TweetsDataset(data = data)


    def query_most_liked(self,n = 20):
        data = self.data.sort_values("likeCount",ascending = False).head(n)

        # Create TweetsDataset
        dataset = TweetsDataset(data = data)
        return dataset


    #----------------------------------------------------------------------------------------------------
    # NETWORKS
    #----------------------------------------------------------------------------------------------------

    def make_network(self,min_degree = 3,keep_only_largest_component = True):


        G = nx.Graph()

        for i,row in tqdm(self.data.iterrows()):
            add_edge(G,row)

        # remove low-degree nodes
        low_degree = [n for n, d in G.degree() if d < min_degree]
        G.remove_nodes_from(low_degree)

        # largest connected component
        if keep_only_largest_component:
            components = nx.connected_components(G)
            largest_component = max(components, key=len)
            G = G.subgraph(largest_component)

        return G


    def show_network(self,G = None,k = 10,**kwargs):

        if G is None:
            G = self.make_network(**kwargs)

        # https://networkx.org/documentation/stable/auto_examples/algorithms/plot_betweenness_centrality.html
        # compute centrality
        centrality = nx.betweenness_centrality(G, k=k, endpoints=True)

        # compute community structure
        lpc = nx.community.label_propagation_communities(G)
        community_index = {n: i for i, com in enumerate(lpc) for n in com}

        #### draw graph ####
        fig, ax = plt.subplots(figsize=(20, 15))
        pos = nx.spring_layout(G, k = 0.1, seed=4572321)
        node_color = [community_index[n] for n in G]
        node_size = [v * 10000 for v in centrality.values()]
        nx.draw_networkx(
            G,
            pos=pos,
            with_labels=True,
            node_color=node_color,
            node_size=node_size,
            edge_color="gainsboro",
            alpha=0.4,
        )

        # Resize figure for label readibility
        ax.margins(0.1, 0.05)
        fig.tight_layout()
        plt.axis("off")
        plt.show()

    def export_network_gephi(self,G,folder = "."):

        # Compute nodes
        nodes_data = []

        for node in G.nodes():
            node_data = G.nodes[node]
            nodes_data.append({"Id":node,"Label":node,"followersCount":node_data["followersCount"],"description":node_data["description"]})

        nodes_data = pd.DataFrame(nodes_data)

        # Compute edges
        edges_data = []

        for source,destination in G.edges():
            edge_data = G.edges[source,destination]
            edges_data.append({"Source":source,"Target":destination,"Category":edge_data["label"],"Type":"Undirected"})

        edges_data = pd.DataFrame(edges_data)
        edges_data = edges_data.loc[edges_data["Source"] != edges_data["Target"]].reset_index(drop = True)

        nodes_data.to_csv(os.path.join(folder,"nodes.csv"),encoding = "utf8",index = False)
        edges_data.to_csv(os.path.join(folder,"edges.csv"),encoding = "utf8",index = False)

        print(f"Exported nodes.csv and edges.csv in folder '{folder}'")

        return nodes_data,edges_data

    #----------------------------------------------------------------------------------------------------
    # VISUALIZATION
    #----------------------------------------------------------------------------------------------------

    def show_lang(self,**kwargs):
        count = self.data.groupby(["lang"])["count"].sum()
        fig = px.sunburst(
            count.reset_index(),path = ["lang"],values = "count",**kwargs,
        )
        return fig

    def show_hashtags(self,n = 50,kind = "treemap",**kwargs):
        count = self.data["hashtags"].explode().dropna().str.upper().value_counts().head(n).reset_index()
        # count = count.loc[count["index"] != "COP26"]
        fig = px.treemap(count,path = ["index"],values = "hashtags",**kwargs)
        return fig


    def show_evolution(self,period = "1W",agg_variable = "count",height = 300):

        count = (self.data
            .set_index("date")
            .groupby([pd.Grouper(freq = period)])
            [agg_variable].sum()
            .reset_index()
        )

        fig = px.bar(
            count,x = "date",y = agg_variable,
            height = height,
        )
        return fig


    def show(self,head = None,data = None,embed = False):

        if head is None and data is None:
            head = len(self.data)

        if head is not None:
            data = self.query_most_liked(head).data
            self.show(data = data,embed = embed,head = None)
        else:
            if embed:
                self.show_embed(data)
            else:
            
                for i,row in data.iterrows():
                    
                    md = [
                        f">> From {row['username']} - Likes {row['likeCount']} - Date {row['date']}",
                        f"{row['content']}\n",
                        "------------------------------------------"
                    ]
                    print("\n".join(md))
    
    def show_embed(self,data):

        if isinstance(data,int):
            t = Tweet(self.data["url"].iloc[data])
            return t

        else:
            for i,row in data.iterrows():

                t = Tweet(row["url"])
                display(HTML(t.text))

    def show_most_mentioned_users(self,top = 30,height = 300,**kwargs):

        count = (self.data
            .explode("mentionedUsers")
            ["mentionedUsers"]
            .map(lambda x : x["username"] if x is not None else None)
            .dropna()
            .value_counts()
            .head(top)
            .reset_index()
        )

        fig = px.bar(count,
            height = height,
            x = "index",y = "mentionedUsers",**kwargs
        )
        return fig

    def show_most_liked_users(self,variable = "likeCount",by = None,top = 40,height = 400,**kwargs):

        if by is None:
            count = self.data.groupby(["username"],as_index = False)[variable].sum()
        else:
            count = self.data.groupby(["username",by],as_index = False)[variable].sum()
        count = count.sort_values(variable,ascending = False).head(top)

        fig = px.bar(count,
            category_orders={'username': count['username'].to_list()},
            color = by,
            height = height,
            x = "username",y = variable,**kwargs
        )
        return fig

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


    def show_strongest_emotions(self,emotion,top = 20,return_data = False,embed = True):
        strongest = self.data.sort_values(emotion,ascending = False).head(top)
        if return_data:
            return strongest
        else:
            self.show(strongest,embed)

    def show_wordcloud(self,show = True,height = 600,width = 1000,without_hashtags = True):

        text = self.data["clean_text"].tolist()
        text = " ".join(text)

        stopwords = set(STOPWORDS)
        stopwords.add("said")
        stopwords.update(set(stopwords_en))
        stopwords.update(set(stopwords_fr))

        wc = WordCloud(
            background_color="black", max_words=1000,height = height,width = width,
            stopwords=stopwords, contour_width=3, contour_color='black',colormap = "coolwarm",
            min_word_length=1
        )

        # generate word cloud
        wc.generate(text)

        # show
        fig = plt.figure(figsize = (15,15))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")

        if show:
            plt.show()
        else:
            return fig








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
        topics, probs = self.topic_model.fit_transform(self.data["clean_text"].tolist())

        # Add results
        self.data["topic_number"] = topics
        self.data["topic_proba"] = probs

        # Get topic summary
        self.topic_summary = self.topic_model.get_topic_info()

    def understand_topic(self,i):
        topic_nr = self.topic_summary.iloc[i]["Topic"] # select a frequent topic
        return self.topic_model.get_topic(topic_nr)



    def fit_tfidf(self):

        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words = stopwords,
            
        )

        X = self.vectorizer.fit_transform(
            tweets_total.data["clean_text"],
        )

        X = pd.DataFrame(X.toarray(),columns = self.vectorizer.get_feature_names_out())

        return X


    def fit_clusters(self,X,model = None,n_clusters = 10,reduce_2D = True):

        # Use default model
        if model is None:
            model = KMeans(n_clusters=n_clusters)
        
        # Store model as attribute
        self.clustering_model= model 
        
        # Apply clustering methodology
        clusters = model.fit_predict(X)
        self.data["cluster"] = clusters
        self.data["cluster"] = self.data["cluster"].astype(str)

        # Prepare wrapping text for visualization
        self.data["clean_text_wrapped"] = self.data["clean_text"].map(lambda x : '<br>'.join(textwrap.wrap(x, width=50)))

        # Reduce dimensionality to 2D
        if reduce_2D:
            tsne = TSNE()
            X_2D = tsne.fit_transform(X)
            X_2D = pd.DataFrame(X_2D,columns = ["x","y"])
            self.data = pd.concat([self.data,X_2D],axis = 1,ignore_index = True)


    def show_tweets2D(self,X_2D):
        fig = px.scatter(X_2D,x = "x",y = "y",color = "cluster",size = "likeCount",hover_data=["clean_text_wrapped"])
        return fig


    def explain_clusters



