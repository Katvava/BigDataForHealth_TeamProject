from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk import word_tokenize
import string

def tokenizer_better(text):
    # tokenize the text by replacing punctuation and numbers with spaces and lowercase all words

    punc_list = string.punctuation + '0123456789'
    t = str.maketrans(dict.fromkeys(punc_list, " "))
    text = text.lower().translate(t)
    tokens = word_tokenize(text)
    return tokens

def get_bow_feature(df_train, df_valid):
    my_stop_words = ['the','and','to','of','was','with','a','on','in','for','name',
                     'is','patient','s','he','at','as','or','one','she','his','her','am',
                     'were','you','pt','pm','by','be','had','your','this','date',
                    'from','there','an','that','p','are','have','has','h','but','o',
                    'namepattern','which','every','also']

    vect = CountVectorizer(max_features = 3000,
                           tokenizer = tokenizer_better,
                           stop_words = my_stop_words)
    # this could take a while
    vect.fit(df_train.TEXT.values)

    X_train_tf = vect.transform(df_train.TEXT.values)
    X_valid_tf = vect.transform(df_valid.TEXT.values)

    y_train = df_train.OUTPUT_LABEL
    y_valid = df_valid.OUTPUT_LABEL

    return X_train_tf, y_train, X_valid_tf, y_valid