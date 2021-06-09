import os
from io import BytesIO
import base64
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from io import BytesIO
import base64
from dictionary_file import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
model = pickle.load(open('revenue_model.pkl', 'rb'))

@app.route('/', methods=['GET'])
def Home():

    return render_template('index.html')

@app.route("/predict", methods=['GET','POST'])
def predict_revenue():
    if request.method == 'POST':
        collection = request.form['collection']
        if collection in collection_dict.keys():
            collection_val=collection_dict[collection]
        else:
            collection_val=0
        budget=int(request.form['budget'])
        popularity = int(request.form['popular'])
        rel_yr = 2021-int(request.form['release'])
        runtime = int(request.form['runtime'])
        lead = request.form['actor']
        if lead in cast_dict.keys():
            actor_val = cast_dict[lead]
        else:
            actor_val=0
        homepage=request.form['homepage']
        if homepage=='yes':
             homepage=1
        else:
             homepage=0

        tagline = request.form['tagline']
        if tagline == 'yes':
             tagline = 1
        else:
             tagline = 0




        genre=request.form['Genre']
        genre_arr=[]
        if (genre == 'Drama'):
            genre_arr=[1,0,0,0,0,0,0,0,0,0]
        elif(genre == 'Comedy'):
            genre_arr=[0,1,0,0,0,0,0,0,0,0]
        elif(genre=='Action'):
            genre_arr=[0,0,1,0,0,0,0,0,0,0]
        elif (genre == 'Adventure'):
            genre_arr = [0,0,0,1,0,0,0,0,0,0]
        elif (genre == 'Horror'):
            genre_arr = [0,0,0,0,1,0,0,0,0,0]
        elif (genre == 'Crime'):
            genre_arr = [0,0,0,0,0,1,0,0,0,0]
        elif (genre == 'Thriller'):
            genre_arr = [0,0,0,0,0,0,1,0,0,0]
        elif (genre == 'Animation'):
            genre_arr = [0,0,0,0,0,0,0,1,0,0]
        elif (genre == 'Documentary'):
            genre_arr = [0,0,0,0,0,0,0,0,1,0]
        elif (genre == 'Fantasy'):
            genre_arr = [0,0,0,0,0,0,0,0,0,1]

        else:
            genre_arr = [0,0,0,0,0,0,0,0,0,0]

        language = request.form['Language']
        language_arr = []
        if (language == 'English'):
            language_arr = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif (language == 'French'):
            language_arr = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif (language == 'Russian'):
            language_arr = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif (language == 'Español'):
            language_arr = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif (language == 'Hindi'):
            language_arr = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif (language == 'Japanese'):
            language_arr = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif (language == 'Italiano'):
            language_arr = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif (language == 'chinese'):
            language_arr = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif (language == 'Korean'):
            language_arr = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        elif (language == 'zh'):
            language_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

        else:
            language_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        production_company = request.form['Company']
        company_arr = []
        if (production_company == 'Paramount Pictures'):
             company_arr = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif (production_company == 'Universal Pictures'):
             company_arr = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif (production_company == 'Twentieth Century Fox Film Corporation'):
             company_arr = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif (production_company == 'Columbia Pictures'):
             company_arr = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif (production_company == 'Warner Bros.'):
             company_arr = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif (production_company == 'New Line Cinema'):
             company_arr = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif (production_company == 'Walt Disney Pictures'):
             company_arr = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif (production_company == 'Columbia Pictures Corporation'):
             company_arr = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif (production_company == 'TriStar Pictures'):
             company_arr = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        elif (production_company == 'United Artists'):
             company_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

        else:
             company_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        production_country = request.form['Country']
        country_arr = []
        if (production_country == 'United States of America'):
              country_arr = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif (production_country == 'United Kingdom'):
              country_arr = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif (production_country == 'France'):
              country_arr = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif (production_country == 'Canada'):
              country_arr = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif (production_country == 'Germany'):
              country_arr = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif (production_country == 'India'):
              country_arr = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif (production_country == 'Australia'):
              country_arr = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif (production_country == 'Japan'):
              country_arr = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif (production_country == 'Russia'):
              country_arr = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        elif (production_country == 'Italy'):
              country_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

        else:
              country_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        features=[collection_val,budget,popularity,rel_yr,runtime,actor_val,homepage,tagline]+genre_arr+language_arr+company_arr+country_arr
        prediction = model.predict([features])
        output = round(prediction[0], 2)
        print(output)
        return render_template('home.html', prediction_texts="Predicted box office revenue is  {}".format(output))

    else:
        return render_template('home.html')


@app.route("/predict_tv_series", methods=['GET','POST'])
def predict_success():
    model = pickle.load(open('tv_success.pkl', 'rb'))
    if request.method == 'POST':
        runtime = request.form['runtime']
        lead=request.form['actor']
        if lead in star1_dict.keys():
            lead_val=star1_dict[lead]
        else:
            lead_val=0

        votes= request.form['votes']
        Start_year=request.form['start']
        End_year = request.form['end']


        genre=request.form['Genre']
        genre_arr=[]
        if (genre == 'Comedy'):
            genre_arr=[1,0,0,0,0,0,0,0,0,0]
        elif(genre == 'Drama'):
            genre_arr=[0,1,0,0,0,0,0,0,0,0]
        elif(genre=='Animation'):
            genre_arr=[0,0,1,0,0,0,0,0,0,0]
        elif (genre == 'Action'):
            genre_arr = [0,0,0,1,0,0,0,0,0,0]
        elif (genre == 'Crime'):
            genre_arr = [0,0,0,0,1,0,0,0,0,0]
        elif (genre == 'Adventure'):
            genre_arr = [0,0,0,0,0,1,0,0,0,0]
        elif (genre == 'Documentary'):
            genre_arr = [0,0,0,0,0,0,1,0,0,0]
        elif (genre == 'Biography'):
            genre_arr = [0,0,0,0,0,0,0,1,0,0]
        elif (genre == 'Game-Show'):
            genre_arr = [0,0,0,0,0,0,0,0,1,0]
        elif (genre == 'Reality-TV'):
            genre_arr = [0,0,0,0,0,0,0,0,0,1]

        else:
            genre_arr = [0,0,0,0,0,0,0,0,0,0]

        certificate = request.form['certificate']
        certificate_arr = []
        if (certificate == 'Unknown'):
            certificate_arr = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif (certificate == '18'):
            certificate_arr = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif (certificate == '16'):
            certificate_arr = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif (certificate == '16+'):
            certificate_arr = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif (certificate == '15+'):
            certificate_arr = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif (certificate == '13'):
            certificate_arr = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif (certificate == 'A'):
            certificate_arr = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif (certificate == '18+'):
            certificate_arr = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif (certificate == 'U'):
            certificate_arr = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        elif (certificate == '12+'):
            certificate_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

        else:
            certificate_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


        features=[runtime,lead_val,votes,Start_year,End_year]+genre_arr+certificate_arr
        prediction = model.predict_proba([features])
        output = round(prediction[0][1]*100, 2)
        return render_template('home_tv.html', prediction_texts="There is {}% chance that this TV-Series will be popular".format(output))

    else:
        return render_template('home_tv.html')

@app.route("/sentiment", methods=['GET','POST'])
def sentiment_analysis():
    if request.method == 'POST':
        name=request.form['name']
        movie_df = pd.read_csv('rotten_tomatoes_movies.csv')
        review_df = pd.read_csv('rotten_tomatoes_critic_reviews.csv')
        inner_merged_total = pd.merge(movie_df, review_df, on=["rotten_tomatoes_link"])
        filter_movie_df = inner_merged_total[inner_merged_total["movie_title"] == name]
        filter_movie_df["review_content"]=filter_movie_df["review_content"].fillna("")
        review_list = list(filter_movie_df["review_content"].values)
        print(type(review_list[0]))
        print(review_list[0])
        ps = WordNetLemmatizer()
        corpus = []
        for i in range(0, len(review_list)):
            review = re.sub('[^a-zA-Z]', ' ', review_list[i])
            review = review.lower()
            review = review.split()

            review = [ps.lemmatize(word) for word in review if not word in stopwords.words('english')]
            review = ' '.join(review)
            corpus.append(review)
        poor_reviews = []
        avg_reviews=[]
        best_reviews = []

        for review in corpus:
            blob = TextBlob(review)
            sentiment = blob.sentiment.polarity
            if sentiment in list(range(-1, 5)):
                poor_reviews.append(review)
            else:
                best_reviews.append(review)
        img= BytesIO()
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('equal')
        langs = ["Bad","Good"]

        students = [len(poor_reviews),len(best_reviews)]
        ax.pie(students, labels=langs, autopct='%1.2f%%')

        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        return render_template('movie_sentiment.html', plot_url=plot_url)
    else:
        return render_template('movie_sentiment.html')

@app.route("/recommend", methods=['GET', 'POST'])
def movie_recommend():
    if request.method == 'POST':
        submit=True
        name = request.form['name']
        df=pd.read_csv("rotten_tomatoes_top_movies.csv")
        def get_index_from_title(name):
            return df[df.title == name].index.values[0]

        movie_index = get_index_from_title(name)
        features = ['type', 'rating', 'genre']

        for feature in features:
            df[feature] = df[feature].fillna('')

        def combined_features(row):
            return str(row['rating']) + " " + str(row['type']) + " " + row['genre']

        df["combined_features"] = df.apply(combined_features, axis=1)
        cv = CountVectorizer()
        count_matrix = cv.fit_transform(df["combined_features"])
        print("Count Matrix:", count_matrix.toarray())
        cosine_sim = cosine_similarity(count_matrix)
        similar_movies = list(enumerate(cosine_sim[movie_index]))
        sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

        def get_title_from_index(index):
            return df[df.index == index]["title"].values[0]

        def get_link_from_index(index):
            return df[df.index == index]["link"].values[0]
        i = 0
        movie_names = []
        links=[]
        for movie in sorted_similar_movies:
            movie_names.append(get_title_from_index(movie[0]))
            links.append((get_link_from_index(movie[0])))
            i = i + 1
            if i > 7:
                break
        movie_names=list(movie_names)
        links=list(links)
        dictionary = dict(zip(movie_names, links))

        print(dictionary)
        return render_template('movie_recommendation.html',movie_names=dictionary,len=len(movie_names),val=submit)
    else:
        submit=False

        return render_template('movie_recommendation.html',val=submit)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)

