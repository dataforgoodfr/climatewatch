import streamlit as st
import plotly.express as px
import pandas as pd
import os
from copy import deepcopy

from climatewatch.twitter import TweetsDataset

st.set_page_config(page_title="ClimateWatch", page_icon="🔥", layout="wide", initial_sidebar_state="auto", menu_items=None)


PATH = "./data/EACOP/full_data_EACOP.pkl"

@st.cache(allow_output_mutation=True)
def get_data():

    tweets = TweetsDataset(pkl_path = PATH)

    def find_hashtags(x,l):
        if isinstance(l,list):
            return x in [y.upper() for y in l]
        else:
            return False
        
    tweets.data["is_STOPTOTAL"] = tweets.data["hashtags"].map(lambda y : find_hashtags("STOPTOTAL",y))
    tweets.data = tweets.data.query("not is_STOPTOTAL and not is_EACOP")
    # tweets.data = tweets.data.query("is_STOPTOTAL or is_EACOP")

    return tweets


st.sidebar.write("# 🔥 ClimateWatch - Twitter")

tweets_raw = get_data()
tweets = deepcopy(tweets_raw)

# FILTERS
st.sidebar.write("### Options")
lang = st.sidebar.multiselect("Language",list(tweets.data["lang"].unique()))
hashtags = st.sidebar.multiselect("Hashtags",tweets.get_all_hashtags(100))
search_text = st.sidebar.text_input("Search")


if len(lang) > 0: tweets.data = tweets.data.loc[tweets.data["lang"].isin(lang)]
if len(hashtags) > 0: tweets.filter_by_hashtags(hashtags)
if len(search_text) > 0: tweets.data = tweets.search(search_text)

st.sidebar.write("### Stats")
st.sidebar.metric("Number of tweets",len(tweets.data))
# st.sidebar.metric(f"{len(tweets.data)} tweets")

st.sidebar.image("./assets/logo.png")




st.write("## 📉 Data exploration")

col1,col2 = st.columns(2)

fig = tweets.show_evolution(period = "1W",height = 400)
col1.plotly_chart(fig,use_container_width = True)

fig = tweets.show_lang(height = 400)
col2.plotly_chart(fig,use_container_width = True)

fig = tweets.show_hashtags(height = 300)
st.plotly_chart(fig,use_container_width = True)



def show_tweet(row):
    md = [
        f"**📣 From {row['username']}** - *Likes* {row['likeCount']} - *Retweets* {row['retweetCount']} - *Date* {str(row['date'])[:10]}\n",
        f"{row['content']}",
    ]
    st.info("\n".join(md))

def show_tweets(df,top = 5):
    for i,row in df.head(top).iterrows():
        show_tweet(row)

st.write("## 📑 Tweets explorer")

col1,col2 = st.columns(2)
options = ["likeCount","retweetCount","reach"]
metric = col1.selectbox("Select metric",options)
if metric not in options: metric = "likeCount"
top_tweets = int(col2.number_input("Select number of tweets",min_value = 10,step = 5))
tweets_sample = tweets.data.sort_values(metric,ascending = False)

with st.expander("See tweets"):
    show_tweets(tweets_sample,top_tweets)

with st.expander("See wordcloud"):
    fig = TweetsDataset(data = tweets_sample).show_wordcloud(show = False)
    st.write(fig)



st.write("## 🙋‍♀️ Users analysis")
fig = tweets.show_most_liked_users("likeCount",by = "lang",top = 25,height = 300)
st.plotly_chart(fig,use_container_width = True)

users = tweets.data.groupby("username",as_index = False)["likeCount"].sum().sort_values("likeCount",ascending = False)
user = st.selectbox("Select user",users["username"].tolist())
tweets_user = tweets.data.query(f"username=='{user}'").sort_values("likeCount",ascending = False)
metrics_user = [
    len(tweets_user),
    tweets_user["likeCount"].sum(),
    tweets_user["retweetCount"].sum(),
]
col1,col2,col3 = st.columns(3)
col1.metric("Number of tweets",metrics_user[0])
col2.metric("Likes",metrics_user[1])
col3.metric("Retweets",metrics_user[2])


with st.expander("See user tweets"):
    show_tweets(tweets_user,5)

with st.expander("See wordcloud"):
    fig = TweetsDataset(data = tweets_user).show_wordcloud(show = False)
    st.write(fig)





