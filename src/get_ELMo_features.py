# set up notebook
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from get_bag_of_words_feature import tokenizer_better
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_curve, accuracy_score, auc
from baseline_ml_model import clean_text, create_training_testing_data
from allennlp.commands.elmo import ElmoEmbedder
import nltk
from nltk.corpus import stopwords
import pickle
import os
import sys
from python_get_adm_notes import clean_data

SENTENCE_LEN = 60

# Lookup the ELMo embeddings for all documents (all sentences) in our dataset. Store those
# in a numpy matrix so that we must compute the ELMo embeddings only once.
def create_elmo_embeddings(elmo, documents):
    num_sentences = len(documents)
    print("\n\n:: Lookup of " + str(num_sentences) + " ELMo representations. This takes a while ::")
    embeddings = []
    labels = []
    tokens = [document['tokens'] for document in documents]

    documentIdx = 0
    for elmo_embedding in elmo.embed_sentences(tokens):
        document = documents[documentIdx]
        # Average the 3 layers returned from ELMo
        avg_elmo_embedding = np.average(elmo_embedding, axis=0)

        embeddings.append(avg_elmo_embedding)
        labels.append(document['label'])

        # Some progress info
        documentIdx += 1
        percent = 100.0 * documentIdx / num_sentences
        line = '[{0}{1}]'.format('=' * int(percent / 2), ' ' * (50 - int(percent / 2)))
        status = '\r{0:3.0f}%{1} {2:3d}/{3:3d} sentences'
        sys.stdout.write(status.format(percent, line, documentIdx, num_sentences))

    return embeddings, labels

# :: Pad the x matrix to uniform length ::
def pad_x_matrix(x_matrix):
    max_tokens = SENTENCE_LEN

    for sentenceIdx in range(len(x_matrix)):
        sent = x_matrix[sentenceIdx]
        sentence_vec = np.array(sent, dtype=np.float32)
        padding_length = max_tokens - sentence_vec.shape[0]
        if padding_length > 0:
            x_matrix[sentenceIdx] = np.append(sent, np.zeros((padding_length, sentence_vec.shape[1])), axis=0)
        if padding_length < 0:
            x_matrix[sentenceIdx] = sent[:max_tokens]
    matrix = np.asarray(x_matrix, dtype=np.float32)
    return matrix

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
                new_texts.append(text[i:])
                empty_str = False
                break
        if empty_str == True:
            new_texts.append(text)
            print('Empty string found, count: {}.'.format(count))
            count = count + 1
            continue

    return new_texts

def remove_stop_words(all_sentences_tokens):
    stop_words = set(stopwords.words('english'))

    filtered_all_sentence_tokens = []

    for word_tokens in all_sentences_tokens:
        filtered_sentence_tokens = [w for w in word_tokens if not w in stop_words]
        filtered_all_sentence_tokens.append(filtered_sentence_tokens)

    return filtered_all_sentence_tokens

def build_document(Xs, ys):

    documents = []
    for X, y in zip(Xs, ys):
        doc = {}

        if len(X)==0:
            continue

        doc['tokens'] = X[:SENTENCE_LEN]
        doc['label'] = y
        documents.append(doc)

    return documents

def get_sentence_len(Xs):
    l = []
    for X in Xs:
        l.append(len(X))

    return l

def main():
    df_adm_notes_clean = clean_data()



    # df_adm_notes_clean = pd.read_pickle("/home/xi/Desktop/df_adm_notes_clean.pkl")

    df_train, df_test, df_valid = create_training_testing_data(df_adm_notes_clean)

    # preprocess the text to deal with known issues
    df_train = clean_text(df_train)
    df_valid = clean_text(df_valid)
    df_test = clean_text(df_test)

    df_train_text = df_train.TEXT.values
    df_valid_text = df_valid.TEXT.values
    df_test_text = df_test.TEXT.values

    df_train_text = extract_clean_diagnosis_content(df_train_text)
    df_valid_text = extract_clean_diagnosis_content(df_valid_text)
    df_test_text = extract_clean_diagnosis_content(df_test_text)

    df_train_y = df_train.OUTPUT_LABEL.values
    df_valid_y = df_valid.OUTPUT_LABEL.values
    df_test_y = df_test.OUTPUT_LABEL.values

    if not os.path.exists('../data/saved_tokens/df_train_tokens.pkl'):
        df_train_tokens = [tokenizer_better(x) for x in df_train_text]
        df_train_tokens = remove_stop_words(df_train_tokens)
        with open('../data/saved_tokens/df_train_tokens.pkl', 'wb') as f:
            pickle.dump(df_train_tokens, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('../data/saved_tokens/df_train_tokens.pkl', 'rb') as f:
            df_train_tokens = pickle.load(f)

    if not os.path.exists('../data/saved_tokens/df_valid_tokens.pkl'):
        df_valid_tokens = [tokenizer_better(x) for x in df_valid_text]
        df_valid_tokens = remove_stop_words(df_valid_tokens)
        with open('../data/saved_tokens/df_valid_tokens.pkl', 'wb') as f:
            pickle.dump(df_valid_tokens, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('../data/saved_tokens/df_valid_tokens.pkl', 'rb') as f:
            df_valid_tokens = pickle.load(f)

    if not os.path.exists('../data/saved_tokens/df_test_tokens.pkl'):
        df_test_tokens = [tokenizer_better(x) for x in df_test_text]
        df_test_tokens = remove_stop_words(df_test_tokens)
        with open('../data/saved_tokens/df_test_tokens.pkl', 'wb') as f:
            pickle.dump(df_test_tokens, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('../data/saved_tokens/df_test_tokens.pkl', 'rb') as f:
            df_test_tokens = pickle.load(f)

    train_docs = build_document(df_train_tokens, df_train_y)
    valid_docs = build_document(df_valid_tokens, df_valid_y)
    test_docs = build_document(df_test_tokens, df_test_y)

    elmo = ElmoEmbedder(cuda_device=0)  # Set cuda_device to the ID of your GPU if you have one
    train_X, train_y = create_elmo_embeddings(elmo, train_docs)
    valid_X, valid_y = create_elmo_embeddings(elmo, valid_docs)
    test_X, test_y = create_elmo_embeddings(elmo, test_docs)

    train_l_before_padding = get_sentence_len(train_X)
    valid_l_before_padding = get_sentence_len(valid_X)
    test_l_before_padding = get_sentence_len(test_X)

    train_X = pad_x_matrix(train_X)
    train_y = np.array(train_y)
    valid_X = pad_x_matrix(valid_X)
    valid_y = np.array(valid_y)
    test_X = pad_x_matrix(test_X)
    test_y = np.array(test_y)

    train_data = {}
    train_data['X'] = train_X
    train_data['y'] = train_y
    with open('../data/saved_features/train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open('../data/saved_features/train_l_before_padding.pkl', 'wb') as f:
        pickle.dump(train_l_before_padding, f, protocol=pickle.HIGHEST_PROTOCOL)

    valid_data = {}
    valid_data['X'] = valid_X
    valid_data['y'] = valid_y
    with open('../data/saved_features/valid_data.pkl', 'wb') as f:
        pickle.dump(valid_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open('../data/saved_features/valid_l_before_padding.pkl', 'wb') as f:
        pickle.dump(valid_l_before_padding, f, protocol=pickle.HIGHEST_PROTOCOL)

    test_data = {}
    test_data['X'] = test_X
    test_data['y'] = test_y
    with open('../data/saved_features/test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open('../data/saved_features/test_l_before_padding.pkl', 'wb') as f:
        pickle.dump(test_l_before_padding, f, protocol=pickle.HIGHEST_PROTOCOL)



    zx = 1

if __name__ == '__main__':
    main()