import argparse
import datetime
import pandas as pd
import mlflow
import mlflow.sklearn
import re
import nltk
from nltk.tokenize import RegexpTokenizer 
from nltk.stem import WordNetLemmatizer,PorterStemmer, SnowballStemmer
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB         # Naive Bayes
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import os

def preprocess(sentence):
     
    sentence = str(sentence)
    sentence = sentence.lower()
    # sentence = sentence.replace('{html}',"") 
    # For Regex Pattern Object
    cleanr = re.compile(r'<\s*[^>]*\s*>')
    cleantext = re.sub(cleanr, ' ', sentence)
    rem_hyp = re.sub(r'(\w+)-(\w+)', r'\1 \2', cleantext)
    rem_punc = re.sub(r'[^\w\s]', '', rem_hyp)
    # re_clean = re.sub(r'[^a-z0-9A-Z_]',' ', cleantext)
    rem_http = re.sub(r'http\S+', '', rem_punc)
    rem_url = re.sub(r"www.\S+", " ", rem_http)
    rem_pat = re.sub("\s*\b(?=\w*(\w)\1{2,})\w*\b",' ', rem_url)
    rem_num = re.sub('[0-9]+', '', rem_pat)

    return rem_num

def tokenize_data(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)  
    return tokens

def remove_stop_words(cleant_text):
    
    # Lemmatization and Stemming
    lemmatizer = WordNetLemmatizer()
    stemmer = SnowballStemmer('english') # SnowballStemmer() and other options
    nltk.download('stopwords')

    #Removing the word 'not' from stopwords
    default_stopwords = set(stopwords.words('english'))
    #excluding some useful words from stop words list as we doing sentiment analysis
    excluding = set(['against','not','don', "don't",'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
                 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 
                 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",'shouldn', "shouldn't", 'wasn',
                 "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])

    custom_stopwords = default_stopwords - excluding
    
    tokens = tokenize_data(cleant_text)
    filtered_words = [w for w in tokens if len(w) > 2 if not w in custom_stopwords]
    # stem_words=[stemmer.stem(w) for w in filtered_words]
    # lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    
    return " ".join(filtered_words)

def prepare_text(text):
    
    cleant_text = preprocess(text)
    remove_sw = remove_stop_words(cleant_text)

    return remove_sw

def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.25)
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    args = parser.parse_args()
   
    # Start Logging
    mlflow.start_run()

    # enable autologging
    mlflow.sklearn.autolog()

    ###################
    #<prepare the data>
    ###################
    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.data)
    
    reviews = pd.read_csv(args.data, header=0, index_col=0)

    mlflow.log_metric("num_samples", reviews.shape[0])
    mlflow.log_metric("num_features", reviews.shape[1] - 1)
    
    print(reviews.columns)
    
    # Create deduplicated frame
    # First by sorting the values by their ProductId
    # Second by dropping duplicates using all columns (which in proven cases are mostly similar) but the product id (which we have seen it differs)
    # reviews_dd = reviews.sort_values(by='ProductId').drop_duplicates(subset=['UserId', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Score', 'Time', 'Summary', 'Text'])
    # reviews_dd = reviews.sort_values(by='ProductId').drop_duplicates(subset=['UserId', 'ProfileName', 'Score', 'Time', 'Summary', 'Text'])
    # Either keeping or removing Time is valid depending on the approach
    reviews_dd = reviews.sort_values(by='ProductId').drop_duplicates(subset=['UserId', 'Text'])

    # Convert time
    reviews_dd['DateTime']=reviews_dd['Time'].apply(lambda x: datetime.datetime.fromtimestamp(x))

    # Remove null values
    reviews_dd.dropna(inplace=True)

    reviews_dd['Year']=reviews_dd['Time'].apply(lambda x: datetime.datetime.fromtimestamp(x).year)

    # Create empty column for binarized score
    reviews_dd['binary_score'] = 0
    
    # Replace only the positive reviews by 1, leave the others with 0 (being negative)
    reviews_dd.loc[reviews_dd['Score'] > 3,'binary_score'] = 1

    reviews_dd['CleanedText'] = reviews_dd['Text'].apply(lambda x:preprocess(x))

    # Removed SWR
    reviews_dd['CleanedSwrText'] = reviews_dd['Text'].apply(lambda x:prepare_text(x))
    
    print(reviews_dd.head(10))
    
    reviews_dd_train, reviews_dd_test = train_test_split(
        reviews_dd,
        test_size=args.test_train_ratio,
    )

    # output paths are mounted as folder, therefore, we are adding a filename to the path
    reviews_dd_train.to_csv(os.path.join(args.train_data, "data.csv"), index=False)

    reviews_dd_test.to_csv(os.path.join(args.test_data, "data.csv"), index=False)

    ####################
    #</prepare the data>
    ####################

    # Stop Logging
    mlflow.end_run()

if __name__ == "__main__":
    main()
