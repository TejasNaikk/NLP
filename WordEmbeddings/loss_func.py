import tensorflow as tf

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """
    epsilon = 1e-10
    A = tf.log(tf.exp(tf.reduce_sum(tf.multiply(inputs,true_w),axis=1))+epsilon)
    B = tf.log(tf.reduce_sum(tf.exp(tf.matmul(true_w,inputs,transpose_b=True)),axis=1)+epsilon)

    return tf.subtract(B, A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """

    epsilon = 1e-10
    num = len(sample)

    labels = tf.reshape(labels,[labels.get_shape()[0],])
    outer_wts = tf.gather(weights,labels)
    outer_bias = tf.gather(biases,labels)
    outer_probs = tf.gather(unigram_prob,labels)

    sample = tf.reshape(sample,[num,])
    neg_wts = tf.gather(weights,sample)
    neg_bias = tf.gather(biases,sample)
    neg_probs = tf.gather(unigram_prob,sample)

    print(neg_wts.get_shape(),neg_bias.get_shape(),neg_probs.get_shape())
    s1 = tf.diag_part(tf.matmul(inputs,outer_wts,transpose_b=True))+outer_bias
    sigmoid1 = tf.sigmoid(s1-tf.log(tf.scalar_mul(num,outer_probs)+epsilon))
    term1 = tf.scalar_mul(-1,tf.log(sigmoid1+epsilon))

    s2 = tf.matmul(inputs,neg_wts,transpose_b=True)+tf.transpose(neg_bias)
    sigmoid2 = tf.sigmoid(s2-tf.transpose(tf.log(tf.scalar_mul(num,neg_probs)+epsilon)))
    term2 = tf.scalar_mul(-1,tf.reduce_sum(tf.log(1-sigmoid2+epsilon),axis=1))

    term1 = tf.reshape(term1,[128,1])
    term2 = tf.reshape(term2,[128,1])
    
    return term1+term2









