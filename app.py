from flask import Flask, render_template, url_for, flash, redirect, request, jsonify, json
from forms import FindTweetsForm
from textblob import TextBlob
from langdetect import detect
import os
from twitter import TwitterObject
from twitter2 import TwitterObject2, TwitterObject3
from utils import get_original_data, get_formatted_data
from app_settings import APP_CONFIG_KEY, PATH_TO_ORIGINAL_DATA, PATH_TO_FORMATTED_DATA_3C, IMAGES_FOLDER, PATH_TO_FORMATTED_DATA_2C

app = Flask(__name__)
#import secrets
#secrets.token_hex(16)
app.config['SECRET_KEY'] = APP_CONFIG_KEY
app.config['UPLOAD_FOLDER'] = IMAGES_FOLDER

# A welcome message to test our server
@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html", title="About")

@app.route("/orginal_dataset")
def original_dataset():
    df = get_original_data(PATH_TO_ORIGINAL_DATA)
    count_data = df['sentiment'].value_counts(normalize=True)
    count_data = count_data.sort_index(axis=0, ascending=True)
    return render_template("original_dataset.html", title="Original Dataset", data=df.values, count_data=count_data.values)

@app.route("/formatted_dataset_3classes")
def formatted_dataset_3classes():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'word_cloud_3classes.png')
    df = get_formatted_data(PATH_TO_FORMATTED_DATA_3C)
    count_data = df['sentiment'].value_counts(normalize=True)
    count_data = count_data.sort_index(axis=0, ascending=True)
    return render_template("formatted_dataset_3classes.html", title="Formatted Dataset with 3 classes", data=df.values, count_data=count_data.values, image=full_filename)

@app.route("/formatted_dataset_2classes")
def formatted_dataset_2classes():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'word_cloud_2classes.png')
    df = get_formatted_data(PATH_TO_FORMATTED_DATA_2C)
    count_data = df['sentiment'].value_counts(normalize=True)
    count_data = count_data.sort_index(axis=0, ascending=True)
    return render_template("formatted_dataset_2classes.html", title="Formatted Dataset with 2 classes", data=df.values, count_data=count_data.values, image=full_filename)


@app.route("/find-tweets", methods=["GET", "POST"])
def find_tweets():
    form = FindTweetsForm()
    show_result = False
    if form.validate_on_submit():
        flash(f"Searching successful!!!", "success")
        show_result = True
        api = TwitterObject(form.searching.data)
        tweets = api.get_tweets()
        number_of_tweets = len(tweets)
        title = "Found: " + str(number_of_tweets) + " tweets!!!"

        n_neu, n_pos, n_neg = 0, 0, 0
        for i in range(0, number_of_tweets):
            if tweets[i]["sentiment"] == "neutral":
                n_neu = n_neu + 1
            if tweets[i]["sentiment"] == "positive":
                n_pos = n_pos + 1
            if tweets[i]["sentiment"] == "negative":
                n_neg = n_neg + 1
        data = [n_neu / number_of_tweets, n_pos / number_of_tweets, n_neg / number_of_tweets]
        return render_template("find_tweets.html", title=title, form=form, data=data, tweets = tweets, no_tweets=number_of_tweets, show_result=show_result)

    return render_template("find_tweets.html", title="Find tweets", form=form, show_result=show_result)


@app.route("/find-tweets3", methods=["GET", "POST"])
def find_tweets3():
    form = FindTweetsForm()
    show_result = False
    if form.validate_on_submit():
        flash(f"Searching successful!!!", "success")
        show_result = True
        api = TwitterObject3(form.searching.data)
        tweets = api.get_tweets()
        number_of_tweets = len(tweets)
        title = "Found: " + str(number_of_tweets) + " tweets!!!"
        count_data = tweets['sentiment'].value_counts(normalize=True)
        count_data = count_data.sort_index(axis=0, ascending=True)

        return render_template("find_tweets3.html", title=title, form=form, count_data=count_data.values,
        polarity=map(json.dumps, tweets.polarity.values), tweets=tweets.values, no_tweets=number_of_tweets, show_result=show_result)

    return render_template("find_tweets3.html", title="Find tweets 3 classes", form=form, show_result=show_result)




@app.route("/find-tweets2", methods=["GET", "POST"])
def find_tweets2():
    form = FindTweetsForm()
    show_result = False
    if form.validate_on_submit():
        api = TwitterObject2(form.searching.data)
        tweets = api.get_tweets()
        flash(f"Searching successful!!!", "success")
        show_result = True
        number_of_tweets = len(tweets)
        title = "Found: " + str(number_of_tweets) + " tweets!!!"
        count_data = tweets['sentiment_textblob'].value_counts(normalize=True)
        count_data = count_data.sort_index(axis=0, ascending=True)
        return render_template("find_tweets2.html", title=title, form=form, count_data=count_data.values,
                               polarity=map(json.dumps, tweets.polarity.values), tweets=tweets.values,
                               no_tweets=number_of_tweets, show_result=show_result)
    return render_template("find_tweets2.html", title="Find tweets", form=form, show_result=show_result)


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(debug=True, threaded=True, port=5000)

