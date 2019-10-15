import re
import argparse
import pandas as pd
import pickle
import numpy as np
import types

import mwapi
from mwapi.errors import APIError
from nltk.tokenize import sent_tokenize

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


from keras import backend as K
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=10, inter_op_parallelism_threads=10)))

def get_arguments():
    parser = argparse.ArgumentParser(
        description='This script determines whether a statement needs a citation or not.')
    parser.add_argument('-o', '--out_dir', help='The output directory where we store the results', required=True)
    parser.add_argument('-m', '--model', help='The path to the model which we use for classifying the statements.', required=True)
    parser.add_argument('-v', '--vocab', help='The path to the vocabulary of words we use to represent the statements.', required=True)
    parser.add_argument('-s', '--sections', help='The path to the vocabulary of section with which we trained our model.', required=True)
    
    return parser.parse_args()





def text_to_word_list(text):
    # check first if the statements is longer than a single sentence.
    sentences = re.compile('\.\s+').split(str(text))
    if len(sentences) != 1:
        # text = sentences[random.randint(0, len(sentences) - 1)]
        text = sentences[0]

    text = str(text).lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"nUse t't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.strip().split()

    return text




def construct_instance_reasons(section_dict_path, vocab_w2v_path, max_len=-1):
    # Load the vocabulary
    vocab_w2v = pickle.load(open(vocab_w2v_path, 'rb'), encoding='latin1')

    # load the section dictionary.
    section_dict = pickle.load(open(section_dict_path, 'rb'), encoding='latin1')

    # Load the statements
    statements = pd.DataFrame(data_, columns=['title', 'section', 'statement'])
    

    # construct the training data
    X = []
    sections = []
    y = []
    outstring=[]
    for index, row in statements.iterrows():
        try:
            statement_text = text_to_word_list(row['statement'])

            X_inst = []
            for word in statement_text:
                if max_len != -1 and len(X_inst) >= max_len:
                    continue
                if word not in vocab_w2v:
                    X_inst.append(vocab_w2v['UNK'])
                else:
                    X_inst.append(vocab_w2v[word])

            # extract the section, and in case the section does not exist in the model, then assign UNK
            section = row['section'].strip().lower()
            sections.append(np.array([section_dict[section] if section in section_dict else 0]))

            X.append(X_inst)
            outstring.append(str(row["statement"]))
            

        except Exception as e:
            print(row)
            print(e.message)
    X = pad_sequences(X, maxlen=max_len, value=vocab_w2v['UNK'], padding='pre')

  
    return X, np.array(sections), outstring

if __name__ == '__main__':
    p = get_arguments()

    # save extracted text
    data_ = []

    title_query = input('Type in the title of the article: \n> ')  # query for Wikipedia title
    session = mwapi.Session('https://en.wikipedia.org/') # creating new session


    continued = session.get(
        action= "query",
        format= "json",
        titles= title_query,
        prop = 'extracts',
        exintro = 1,
        explaintext = 1,
        )

 
    try:
        for i in continued['query']['pages']:
            if 'missing' in  continued['query']['pages'][i]:
                print("Title not found. Kindly check if you typed it correctly")

            else:
                statements = continued['query']['pages'][i]['extract']
                title = continued['query']['pages'][i]['title']
                statements = sent_tokenize(statements)
                section_title = 'MAIN_SECTION'
                for statement in statements:
                    data_.append([title, section_title, statement])
        
    except APIError:
        raise ValueError(
            "MediaWiki returned an error:", str(APIError)
            )
    
    print(data_)
    # load the model
    model = load_model(p.model)

    # load the data
    max_seq_length = model.input[0].shape[1].value

    X, sections, outstring = construct_instance_reasons(p.sections, p.vocab, max_seq_length)

    # classify the data
    pred = model.predict([X, sections])
    output = []

    # adding results to a list
    for idx, y_pred in enumerate(pred):
        output.append([data_[idx][0],outstring[idx],y_pred[0]])


    output.sort(key=lambda x: x[2]) #sort  prediction score
    save_output = pd.DataFrame(output)
    save_output.to_csv(p.out_dir + "/" + "predictions.csv", index=False)# save prediction to file
    # printing out the sentence text and the prediction score
    print('Title\tStatement\tPrediction\n')
    for result in output:
        print(result[0]+'\t'+result[1]+'\t'+str(result[2]))
  
