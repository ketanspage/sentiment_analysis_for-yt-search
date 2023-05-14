from flask import Flask, render_template, request
import requests
from datetime import datetime
import pickle

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

app = Flask(__name__)

with open("sentiment_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    keyword = request.form['keyword']
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "maxResults": 10,
        "q": keyword,
        "type": "video",
        "key": "AIzaSyCOdnLItrausVI7sALHLmf4IECYQYuZl8Y"
    }
    response = requests.get(url, params=params)
    yt_data = response.json()
    yt_data = yt_data['items']
    video_data = []
    sentiment_analysis = pipeline('text-classification', model=model, tokenizer=tokenizer)
    for data in yt_data:
        title = data['snippet']['title']
        description = data['snippet']['description']
        date_time_str = data['snippet']['publishTime']
        date_time_obj = datetime.strptime(date_time_str, '%Y-%m-%dT%H:%M:%SZ')
        date_time = date_time_obj.strftime('%Y-%m-%d %H:%M:%S')
        timestamp = date_time_obj.timestamp()
        channel_title = data['snippet']['channelTitle']
        video_sentiment = sentiment_analysis(title + ' ' + description)
        for result in video_sentiment:
            if result['label'] == 'LABEL_2':
                result['label'] = 'Positive'
            elif result['label'] == 'LABEL_1':
                result['label'] = 'Neutral'
            elif result['label'] == 'LABEL_0':
                result['label'] = 'Negative'
            result['score'] = round(result['score'], 2)
        video_data.append({'title': title, 'description': description, 'date_time': date_time, 'timestamp': timestamp,
                           'channel_title': channel_title, 'sentiment_analysis': video_sentiment})
    return render_template('result.html', video_data=video_data)

if __name__ == '__main__':
    app.run(debug=True)
