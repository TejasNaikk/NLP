import numpy as np

def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is the size N array/seq of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]

    T = np.zeros((L,N))
    indices = np.ones((L,N))*-1

    
    T[:,0] = np.transpose(emission_scores[0,:])+start_scores

    for col in range(1,N+1):
        
        if col!=N:
            for row in range(L):

                score_idx = -1
                score_max = float('-inf')

                for i in range(L):

                    if(T[i][col-1]+emission_scores[col][row]+trans_scores[i][row] > score_max):
                        score_max = T[i][col-1]+emission_scores[col][row]+trans_scores[i][row]
                        score_idx = i
                
                T[row][col] = score_max
                indices[row][col] = score_idx

        else:
            score_max = float('-inf')
            score_idx = -1

            for i in range(L):

                if(T[i][-1] + end_scores[i] > score_max):
                    score_max = T[i][-1] + end_scores[i]
                    score_idx = i

    i = N-1
    y = []
    while i>=0:
        y.append(score_idx)
        score_idx = int(indices[score_idx][i])
        i-=1

    return (score_max, y[::-1])
