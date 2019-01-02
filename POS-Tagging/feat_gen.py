#!/bin/python
# -*- coding: utf-8 -*-
from gensim import models
import numpy as np


clusters = {}
word_to_cluster = {}

# Referenced this function from data.py to obtain the training data
def read_file(filename):
    """Read the file in CONLL format, assumes one token and label per line."""
    sents = []
    labels = []
    with open(filename, 'r') as f:
        curr_sent = []
        curr_labels = []
        for line in f.readlines():
            if len(line.strip()) == 0:
                # sometimes there are empty sentences?
                if len(curr_sent) != 0:
                    # end of sentence
                    sents.append(curr_sent)
                    labels.append(curr_labels)
                    curr_sent = []
                    curr_labels = []
            else:
                token, label = line.split()
                curr_sent.append(unicode(token, 'utf-8'))
                curr_labels.append(label)
    return sents, labels


#Function to help me analyze the common suffixes of the label of interest. I used the output of this function to predict the suffixes as a feature.

def get_dict(words,labels,label_of_interest):

    suffix = {}
    for word,label in zip(words,labels):

        if label=="X":
            if word[0] not in suffix:
                suffix[word[0]] = 1
            else:
                suffix[word[0]] += 1

        else:

            if label==label_of_interest:

                if word[-3:] in suffix:
                    suffix[word[-3:]]+=1
                else:
                    suffix[word[-3:]]=1

                if word[-2:] in suffix:
                    suffix[word[-2:]]+=1
                else:
                    suffix[word[-2:]]=1

    return suffix


def brown_clustering(k):

    file = open("Brown_Clustering_output.txt","r")

    for line in file:

        word,binary,_ = line.split()

        cluster_id = int(binary[:k],2)

        if cluster_id in clusters:
           clusters[cluster_id].append(word)
        else: 
            clusters[cluster_id] = [word]
        
        word_to_cluster[word] = cluster_id

    #for key,val in word_to_cluster.items():
    #    print(key,val)
    #    print("\n\n")


def preprocess_corpus(train_sents):
    """Use the sentences to do whatever preprocessing you think is suitable,
    such as counts, keeping track of rare features/words to remove, matches to lexicons,
    loading files, and so on. Avoid doing any of this in token2features, since
    that will be called on every token of every sentence.

    Of course, this is an optional function.

    Note that you can also call token2features here to aggregate feature counts, etc.
    """

    
    train_sents, train_labels = read_file("data/twitter_train.pos")

    poss_x = [":","http","@","#"]
    words = []

    for sent in train_sents:
        for word in sent:
            flag = 1
            for x in poss_x:
                if word.find(x)!=-1:
                    flag = 0
                    break
            if flag==1:
                words.append(word)
    labels = [label for l in train_labels for label in l]

    
    verb_suffix = get_dict(words,labels,"VERB")
    adj_suffix = get_dict(words,labels,"ADJ")
    adv_suffix = get_dict(words,labels,"ADV")
    x = get_dict(words,labels,"X")
    #for key, val in sorted(x.items(), key=lambda (k,v): (-v,k)):
    #    print(key,val)
    
    
    file = open("Input_Brown_Clustering.txt", "w")

    file_words = " ".join(words)
    file.write(file_words)
    file.close()

    #Ran the Brown Clustering program to generate the output.

    brown_clustering(2)
    
    '''            

    print("X constructed ")

    kmeans = KMeans(n_clusters = 20, n_init = 5, n_jobs = -1)
    kmeans.fit(X)

    common_words = kmeans.cluster_centers_.argsort()[:,-1:-11:-1]

    for num, centroid in enumerate(common_words):
        print(str(num) + ' : ' + ', '.join(X_inverse[word] for word in centroid))
    
    '''
    
    pass

    
    
def token2features(sent, i, add_neighs = True):
    """Compute the features of a token.

    All the features are boolean, i.e. they appear or they do not. For the token,
    you have to return a set of strings that represent the features that *fire*
    for the token. See the code below.

    The token is at position i, and the rest of the sentence is provided as well.
    Try to make this efficient, since it is called on every token.

    One thing to note is that it is only called once per token, i.e. we do not call
    this function in the inner loops of training. So if your training is slow, it's
    not because of how long it's taking to run this code. That said, if your number
    of features is quite large, that will cause slowdowns for sure.

    add_neighs is a parameter that allows us to use this function itself in order to
    recursively add the same features, as computed for the neighbors. Of course, we do
    not want to recurse on the neighbors again, and then it is set to False (see code).
    """
    ftrs = []
    # bias
    ftrs.append("BIAS")
    # position features
    if i == 0:
        ftrs.append("SENT_BEGIN")
    if i == len(sent)-1:
        ftrs.append("SENT_END")

    # the word itself
    word = unicode(sent[i])
    ftrs.append("WORD=" + word)
    ftrs.append("LCASE=" + word.lower())
    # some features of the word
    if word.isalnum():
        ftrs.append("IS_ALNUM")
    if word.isnumeric():
        ftrs.append("IS_NUMERIC")
    if word.isdigit():
        ftrs.append("IS_DIGIT")
    if word.isupper():
        ftrs.append("IS_UPPER")
    if word.islower():
        ftrs.append("IS_LOWER")

    # add Suffixes
    
    possible_verb_suffixes = ["ing","ify","ed","ill"]

    if word[-3:].lower() in possible_verb_suffixes or word[-2:] in possible_verb_suffixes:
       ftrs.append("HAS_VERB_SUFFIX")

    possible_adj_suffixes = ["ble","ous","ful","od","er","st"]

    if word[-3:].lower() in possible_adj_suffixes or word[-2:] in possible_adj_suffixes:
        ftrs.append("HAS_ADJ_SUFFIX")

    possible_adv_suffixes = ["ly","ard","en","ow","re","n't","lly"]

    if word[-3:].lower() in possible_adv_suffixes or word[-2:] in possible_adv_suffixes:
        ftrs.append("HAS_ADV_SUFFIX")

        
    # Nouns with first letter capital

    if word[0].isupper() and word[1:].islower():
        ftrs.append("FIRST_LETTER_CAPITAL")

    possible_x_prefixes = ['#','@',':','htt']

    if word[0] in possible_x_prefixes or word[:3] in possible_x_prefixes:
        ftrs.append("X_CLASS")

    # Clustering features
    k = len(clusters)

    #for j in range(k):
    #    if word in word_to_cluster:
    #        if word_to_cluster[word] == j:
    #            ftrs.append(str(j)+"_CLUSTER")
    
    # previous/next word feats
    if add_neighs:
        if i > 0:
            for pf in token2features(sent, i-1, add_neighs = False):
                ftrs.append("PREV_" + pf)
        if i < len(sent)-1:
            for pf in token2features(sent, i+1, add_neighs = False):
                ftrs.append("NEXT_" + pf)

    # return it!
    return ftrs

if __name__ == "__main__":
    sents = [
    [ "I", "love", "food" ]
    ]
    preprocess_corpus(sents)
    for sent in sents:
        for i in xrange(len(sent)):
            print sent[i], ":", token2features(sent, i)
