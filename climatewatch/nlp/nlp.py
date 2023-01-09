"""
Other ideas

# Pipelines
pipe = pipeline(
    "text-classification",
    model = clf.model,
    tokenizer = clf.tokenizer,

)



"""
import numpy as np
import pandas as pd
import os
import csv
import urllib.request
import torch
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.special import softmax

from ..utils import read_json,save_json


TASKS = ["emoji", "emotion", "hate", "irony", "offensive", "sentiment","stance-abortion","stance-atheism", "stance-climate", "stance-feminist", "stance-hillary"]
AVAILABLE_TASKS = ["sentiment","irony","emotion","emoji","stance-climate"]


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class TweetClassifier:
    def __init__(self,task:str,verbose:bool = True,framework = "pt") -> None:
        """Tweet Classifier
        - https://paperswithcode.com/paper/bertweet-a-pre-trained-language-model-for
        """
        # Check valid task
        assert task in TASKS
        self.task = task
        self.model_name = f"cardiffnlp/bertweet-base-{task}"
        if verbose: print(f"[CLASSIFIER] {self.model_name}")

        # Same framework
        assert framework in ["pt","tf"]
        self.framework = framework

        # Prepare model and classifier
        if verbose: print("... Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,model_max_length = 128)
        
        if verbose: print("... Loading classifier")

        if self.framework == "pt":
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        else:
            self.model = TFAutoModelForSequenceClassification.from_pretrained(self.model_name)
            
        # Save pretrained model
        if verbose: print("... Saving models")
        self.tokenizer.save_pretrained(self.model_name)
        self.model.save_pretrained(self.model_name)

        # Get mapping
        if verbose: print("... Fetching mapping")
        self.mapping = self.fetch_mapping()
        

    
    def fetch_mapping(self) -> list:

        filepath = os.path.join(self.model_name,"mapping.json")

        # Avoid reloading from external source if already saved as json in model folder
        if os.path.exists(filepath):
            labels = read_json(filepath)["labels"]
        else:
            # download label mapping
            labels=[]
            if "stance" not in self.task:
                mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{self.task}/mapping.txt"
            else:
                mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/stance/mapping.txt"
            
            # Read and convert labels to list
            with urllib.request.urlopen(mapping_link) as f:
                html = f.read().decode('utf-8').split("\n")
                csvreader = csv.reader(html, delimiter='\t')
            labels = [row[1] for row in csvreader if len(row) > 1]
        
            # Save mapping to avoid reloading from source
            save_json({"labels":labels},filepath)

        return labels


    def predict_proba(self,texts:list,batch_size = None,using_dataloader = True) -> pd.DataFrame:
        """
        TODO
        Speeding inference with 
        - https://medium.com/gumgum-tech/improving-inference-speeds-of-transformer-models-e03944a018aa*
        - https://medium.com/microsoftazure/faster-and-smaller-quantized-nlp-with-hugging-face-and-onnx-runtime-ec5525473bb7
        - https://towardsdatascience.com/divide-hugging-face-transformers-training-time-by-2-or-more-21bf7129db9q-21bf7129db9e
        - https://stackoverflow.com/questions/69517460/bert-get-sentence-embedding/69521616#69521616
        """

        assert isinstance(texts,list)

        if batch_size is None:
            # Get tokenized mapping using pretrained tokenizer
            encoded_input = self.tokenizer(texts, return_tensors=self.framework,padding = True,truncation = True)

            # # Convert to CPU
            # device = torch.device('cpu')
            # for key in encoded_input.keys():
            #     encoded_input[key] = encoded_input[key].to(device)

            # Get prediction as scores
            if self.framework == "pt":
                output = self.model(**encoded_input)
                scores = output[0].detach().numpy()
            else:
                output = self.model(encoded_input)
                scores = output[0].numpy()
            
            # Convert scores to "probas" using softmax function 
            # Add to Pandas dataframe for convenience
            probas = softmax(scores,axis = 1)
            probas = pd.DataFrame(probas,columns = self.mapping)
            return probas
        else:

            if using_dataloader:
                        

                probas = []
                dataloader = DataLoader(texts,batch_size = batch_size,num_workers = 4)
                
                for text_chunk in tqdm(dataloader):

                    # Get tokenized mapping using pretrained tokenizer
                    encoded_input = self.tokenizer(text_chunk, return_tensors=self.framework,padding = True,truncation = True)

                    # Get prediction as scores
                    if self.framework == "pt":
                        output = self.model(**encoded_input)
                        scores = output[0].detach().numpy()
                    else:
                        output = self.model(encoded_input)
                        scores = output[0].numpy()
                    probas.append(scores)

                probas = np.concatenate(probas,axis = 0)

            else:
                        
                probas = []
                texts = list(chunks(texts,batch_size))
                
                for text_chunk in tqdm(texts):

                    # Get tokenized mapping using pretrained tokenizer
                    encoded_input = self.tokenizer(text_chunk, return_tensors=self.framework,padding = True,truncation = True)

                    # Get prediction as scores
                    if self.framework == "pt":
                        output = self.model(**encoded_input)
                        scores = output[0].detach().numpy()
                    else:
                        output = self.model(encoded_input)
                        scores = output[0].numpy()
                    probas.append(scores)

                probas = np.concatenate(probas,axis = 0)
                
            # Convert scores to "probas" using softmax function 
            # Add to Pandas dataframe for convenience
            probas = softmax(probas,axis = 1)
            probas = pd.DataFrame(probas,columns = self.mapping)
            return probas

    def predict(self,texts:list,batch_size = None) -> pd.Series:

        # Predict probabilities for each classes
        probas = self.predict_proba(texts,batch_size)

        # Get prediction using Pandas function idxmax to find the highest proba
        pred = probas.idxmax(axis = 1) 
        return pred




class MetaTweetClassifier:
    def __init__(self,tasks:list = AVAILABLE_TASKS):

        self.tasks = tasks
        self.models = {}

        print(f"... Loading Meta Classifier on tasks {tasks}")
        for task in tasks:
            print(f"    ... Loading model {task}")
            model = TweetClassifier(task,verbose = False)
            self.models[task] = model

    
    def predict(self,text,batch_size = None):

        preds = []

        print("... Predicting classes for all tasks")
        for task in self.tasks:
            print(f"    ... Predicting for task {task}")
            pred = self.models[task].predict(text,batch_size)
            pred.name = task
            preds.append(pred)

        preds = pd.concat(preds,axis = 1)
        return preds



class VaderSentimentClassifier:
    def __init__(self):

        self.model = SentimentIntensityAnalyzer() 


    def predict(self,text):

        if not isinstance(text,list): text = [text]

        pred = []

        for x in text:
            assert isinstance(x,str)
            scores = self.model.polarity_scores(x)
            classes = ["negative","neutral","positive"]

            # Get class
            pred_class = np.argmax([scores["neg"],scores["neu"],scores["pos"]])
            pred_class = classes[pred_class]

            pred.append({"vader_sentiment_class":pred_class,"vader_sentiment_score":scores["compound"]})
            
        
        pred = pd.DataFrame(pred)
        return pred


# files = list(range(0,15))

# tracker.start()

# for i in files:
#     data = read_pickle(f"./drive/MyDrive/DataCOP26/DATADAY{i}.pkl")
#     preds = clf.predict_proba(data["clean_sentiment"].tolist(),batch_size = 128,using_dataloader = False)
#     preds.to_csv(f"./drive/MyDrive/DataCOP26/DATADAY{i}_CLIMATE.csv")


# tracker.stop()