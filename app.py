import pandas as pd
import numpy as np

import re

import pickle
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from flask import Flask, request, render_template

from nltk.stem import WordNetLemmatizer
from nltk import tokenize
from nltk.corpus import stopwords

from textblob import TextBlob

import json
import plotly
import plotly.express as px

stopwords = stopwords.words("english")
lemmatizer = WordNetLemmatizer()
w_tokenizer = tokenize.WhitespaceTokenizer()

# Remove newline
def remove_nline(value):
    return ''.join(value.splitlines())

def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text) if not w in set(stopwords)])

# Load ML model (chosen the logistic regression model)
model = pickle.load(open('SVCmodel.pkl', 'rb')) 

# Load TF IDF document matrix
#tf_idf = TfidfVectorizer(ngram_range = (1,3), analyzer='word')
tf_idf = pickle.load(open('TFIDF.pkl', 'rb'))

# Load trained dataset feature coefficients
feature_coef = pickle.load(open('SVCfeature_coefs_df.pkl','rb'))


# Create application
app = Flask(__name__)

# Bind home function to URL
@app.route('/')
def home():
    return render_template('AUD Classifier.html')#home page

# Bind predict function to URL
@app.route('/predict', methods =['POST'])
def predict():
    
    # Get the user input to display into predict page
    default_text = request.form["post_text"] 
    # Get the user input
    raw_text = request.form["post_text"]


    # Text Preprocessing
    #Process whole raw_txt for overall TFIDF and Coefficients Score
    raw_txt = remove_nline(raw_text)
    raw_txt =raw_txt.lower()
    #Remove ALL punctions
    whole_raw_txt = re.sub('[^a-zA-Z]', ' ', raw_txt)
    whole_raw_txt = whole_raw_txt.split()
    whole_raw_txt = [lemmatizer.lemmatize(word) for word in whole_raw_txt if not word in set(stopwords)]
    processed_whole_raw_txt =[' '.join(whole_raw_txt)]
    #processed_whole_raw_txt = pd.Series([str(processed_whole_raw_txt)], index=[1 ])
    processed_whole_raw_txt_input = tf_idf.transform(processed_whole_raw_txt)


    # Remove most punctuations except !?,.'
    raw_txt = str(raw_txt).translate(str.maketrans('', '', '"#$%&\()*+-/:;<=>@[\\]^_`{|}~'))

    test = tokenize.sent_tokenize(str(raw_txt), language="english")

    testsample_df = pd.DataFrame(test, columns= ['selftext'])

    testsample_df = testsample_df.explode('selftext')

    testsample_df['s_selftext']: list[str] = [re.sub('[^a-zA-Z]', ' ', x) for x in testsample_df['selftext']]

    testsample_df['lemmatized_text'] = testsample_df.s_selftext.apply(lemmatize_text)

    #Get subjectivity score
    testsample_df['subjectivity'] = [TextBlob(str(i)).sentiment.subjectivity for i in testsample_df['lemmatized_text']]
    XTest = tf_idf.transform(testsample_df['lemmatized_text'].values)
    
    # Prediction features
    YTest = model.predict(XTest)
    testsample_df['prediction'] = YTest

    avgPrediction, avgSubjectivity = testsample_df[['prediction','subjectivity']].mean()
    
    
    # Top 25 Explanatory words
    words_input = tf_idf.transform(processed_whole_raw_txt)
    feature_array = np.array(tf_idf.get_feature_names_out())
    tfidf_sorting = np.argsort(words_input.toarray()).flatten()[::-1]

    #idx = feature_coef.index[feature_coef['features']]
    idx = feature_coef.index.tolist()
    feature_coef['idx'] = idx

    # Top 25 feature coefficient
    target_idx = [words_input.nonzero()[1]]
    target_idx :list[str] = np.transpose(target_idx)
    input_coefs = pd.DataFrame(columns = feature_coef.columns )
    for index, row in feature_coef.iterrows():
        if index in target_idx:
            input_coefs = input_coefs.append({
                'features': row['features'],
                'coef': row['coef'],
                'idx': row['idx'],
                }, ignore_index =True)
    input_coefs = input_coefs.sort_values(by='coef', ascending=False)

    if len(input_coefs.index) < 25:
        explanatory_words = input_coefs.loc[:, ['features','coef']]
    else:
        explanatory_words = input_coefs.loc[:25, ['features','coef']]


    ans = ""
    for index, row in explanatory_words.iterrows():
        ans += str(row['features']) + ' - ' + str(row['coef']) + "\n"

    
    dfdata = explanatory_words[['features','coef']][:25]
    X= dfdata['features'].values.tolist()
    Y= dfdata['coef'].values.tolist()
    df = pd.DataFrame({
        "features" : X,
        "coef": Y
    })
    df['coef'] = df['coef'].astype(float)

    #Visualisations
    fig= px.bar(df,  x="features", y="coef", text="coef", title='Top 25 Feature Coefficient Scores')

    #Viz Object to Return
    graphJSON = json.dumps(fig, 
    cls=plotly.utils.PlotlyJSONEncoder)

    headings = ("Sentence","Prediction", "Subjectivity")
    data = testsample_df[['selftext','prediction','subjectivity']].values.tolist()
    
    # Check the output values and retrive the result with html tag based on the value
    if avgPrediction  < 0.5:
        return render_template('AUD Classifier.html', default_text = default_text,
                               result = 'Likely NOT to have AUD!', explanatory_words = ans, avgPrediction = avgPrediction, avgSubjectivity=avgSubjectivity
                               , headings=headings, data=data, graphJSON=graphJSON 
                               )
    elif avgPrediction  >= 0.5:
        return render_template('AUD Classifier.html', default_text = default_text,
                               result = 'Likely have AUD!', explanatory_words = ans, avgPrediction = avgPrediction, avgSubjectivity=avgSubjectivity
                               , headings=headings, data=data, graphJSON=graphJSON 
                               )

@app.route('/about')
def about():
    return render_template("about.html")

if __name__ == '__main__':
#Run the application
    app.run(host="0.0.0.0")#to debug & test locally
