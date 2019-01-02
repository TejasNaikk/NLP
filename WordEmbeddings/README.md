# Word Embeddings 

In this project, I implemented Skip-gram model to learn word representations and computed loss using 
Cross Entropy and NCE methods. I then used the above model (along with some hyperparameter tuning) to determine relations between word pairs.

For example, if given the below query:

Among these word pairs (Relation R),

  (1) pig:mud
  
  (2) politician:votes
  
  (3) dog:bone
  
  (4) bird:worm

The model gives the answer to the below two questions:

  Q1. Which word pairs has the MOST illustrative(similar) example of the relation R?
  
  Q2. Which word pairs has the LEAST illustrative(similar) example of the relation R?
  
## Files in this repository:

1. word2vec_basic.py : This file is the main script for training word2vec model (using skip-gram model)

2. loss_func.py :    This file have two loss functions 

a) Cross_Entropy_loss 

b). nce_loss - Noise contrastive estimation loss.
  
3. word_analogy.py : This file is used for evaluating relation between pairs of words using the similarities of difference vectors.
