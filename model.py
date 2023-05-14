from transformers import pipeline
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import pickle
text=input()
model_sentiment="cardiffnlp/twitter-roberta-base-sentiment"
model=AutoModelForSequenceClassification.from_pretrained(model_sentiment)
tokenizer=AutoTokenizer.from_pretrained(model_sentiment)
classifier=pipeline("sentiment-analysis",model=model,tokenizer=tokenizer)
results=classifier(text)
for result in results:
  if result['label']=='LABEL_2':
     result['label']='positive'
  if result['label']=='LABEL_1':
     result['label']='neutral'
  if result['label']=='LABEL_0':
     result['label']='negetive'   
  label=result['label']
  score=round(result['score'],2)
  print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
with open('sentiment_model.pkl','wb') as f:
  pickle.dump(model,f)
