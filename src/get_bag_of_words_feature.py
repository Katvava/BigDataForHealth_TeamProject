from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk import word_tokenize
import string

SENTENCE_LEN = 60

def extract_clean_diagnosis_content(texts):
    poi = 'Discharge Medications'.lower()

    pois = ['Discharge Medications'.lower(),
            'Discharge Disposition'.lower(),
            'Discharge Diagnosis'.lower(),
            'Discharge Condition'.lower(),
            'Discharge Status'.lower(),
            'Discharge Instructions'.lower()]

    new_texts = []
    count = 0
    for text in texts:
        text = text.lower()

        empty_str = True

        for poi in pois:
            i = text.find(poi)
            if i != -1:
                new_text = text[i:]
                if len(new_text)>SENTENCE_LEN:
                    new_text = new_text[:SENTENCE_LEN]
                new_texts.append(new_text)
                empty_str = False
                break
        if empty_str == True:
            if len(text)>SENTENCE_LEN:
                text = text[:SENTENCE_LEN]
            new_texts.append(text)
            print('Empty string found, count: {}.'.format(count))
            count = count + 1
            continue

    return new_texts

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

    df_train_text = extract_clean_diagnosis_content(df_train.TEXT.values)
    df_valid_text = extract_clean_diagnosis_content(df_valid.TEXT.values)

    # this could take a while
    vect.fit(df_train_text)

    X_train_tf = vect.transform(df_train_text)
    X_valid_tf = vect.transform(df_valid_text)

    y_train = df_train.OUTPUT_LABEL
    y_valid = df_valid.OUTPUT_LABEL

    return X_train_tf, y_train, X_valid_tf, y_valid