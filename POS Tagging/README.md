# POS Tagging

In this project, I have implemented Part-Of-Speech Tagging on Twitter data with an accuracy of 86.09%.

I performed feature engineering and implemented Viterbi Decoding for sequence POS tagging using:

a) Conditional Random Field (CRF)

b) Max Entropy Markov Model (MEMM)

## Files contained are:

1. data.py: The primary entry point that reads the data, and trains and evaluates the tagger implementation.

	usage: python data.py [-h] [-m MODEL] [--test]

	optional arguments:
	  -h, --help            show this help message and exit
	  -m MODEL, --model MODEL
	                        'LR'/'lr' for logistic regression tagger
	                        'CRF'/'crf' for conditional random field tagger
	  --test                Make predictions for test dataset
    
    
2. tagger.py: Code for two sequence taggers, logistic regression and CRF. Both of these taggers rely on 'feats.py' 
and 'feat_gen.py' to compute the features for each token. The CRF tagger also relies on 'viterbi.py' to decode,
and on 'struct_perceptron.py' for the training algorithm (which also needs Viterbi to be working).

3. feats.py & feat_gen.py: Code to compute, index, and maintain the token features.
The primary purpose of 'feats.py' is to map the boolean features computed in 'feats_gen.py' to integers, 
and do the reverse mapping (if you want to know the name of a feature from its index). 'feats_gen.py' is used to compute
the features of a token in a sentence, which you will be extending. The method there returns the computed features for a
token as a list of string.

4. struct_perceptron.py: A direct port (with negligible changes) of the structured perceptron trainer from the 'pystruct' project. Only used for the CRF tagger. The description of the various hyperparameters of the trainer are available here, but you should change them from the constructor in 'tagger.py'.

5. viterbi.py (and viterbi_test.py): General purpose interface to a sequence Viterbi decoder in 'viterbi.py',
Running 'python viterbi_test.py' should result in succesful execution without any exceptions.

6. conlleval.pl: This is the official evaluation script for the CONLL evaluation.
  Although it computes the same metrics as the python code does, it supports a bunch of features, such as: 
  	(a) Latex formatted tables, by using -l, 
  	(b) BIO annotation by default, turned off using -r.
  In particular, when evaluating the output prediction files (~.pred) for POS tagging, 

  $ ./conlleval.pl -r -d \\t < ./predictions/twitter_dev.pos.pred


