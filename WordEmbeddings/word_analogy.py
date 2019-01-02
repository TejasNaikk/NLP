import os
import pickle
import numpy as np
from scipy import spatial
import heapq

model_path = './models/'
#loss_model = 'cross_entropy'
loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'r'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""

def parse(embedding,dictionary):

	with open("word_analogy_dev.txt","r") as file:

		answer_file = open("word_analogy_test_predictions_nce.txt","w")

		for line in file:

			examples,possibilities = line.split("||")
			poss = possibilities.split(",")
			example_len = len(examples.split(","))
			
			word_pairs = [word.split(":") for pairs in line.split("||") for word in pairs.split(",")]

			example_count, diff = 0,[]
			min_dissimilarity,max_dissimilarity = float('inf'),float('-inf')
			most_illustrative,least_illustrative = "",""

			for word1,word2 in word_pairs:
				word_1 = word1[1:]
				word_2 = word2.strip("\n")[:-1]				
				word_id1,word_id2 = dictionary[word_1],dictionary[word_2]
				
				embed1,embed2 = embeddings[word_id1],embeddings[word_id2]

				if example_count < example_len:	
					diff.append((embed2-embed1))
					example_count += 1
					continue

				avg_diff = sum(diff)/example_len
				example_len += 1
				dissimilarity = spatial.distance.cosine(avg_diff,(embed2-embed1))

				if dissimilarity > max_dissimilarity:
					least_illustrative = word1+":"+word2
					max_dissimilarity = dissimilarity

				if dissimilarity < min_dissimilarity:
					most_illustrative = word1+":"+word2
					min_dissimilarity = dissimilarity


			answer_line = (" ".join(poss)).strip('\n')+" "+least_illustrative+" "+most_illustrative
			answer_line += '\n'
			
			answer_file.write(answer_line)	

def similar_words(embeddings,dictionary,word_list,num_of_similar):

	similar1,similar2,similar3, = list(),list(),list()
	answer1,answer2,answer3 = [],[],[]

	curr_embed_1 = embeddings[dictionary[word_list[0]]]
	curr_embed_2 = embeddings[dictionary[word_list[1]]]
	curr_embed_3 = embeddings[dictionary[word_list[2]]]


	for word,word_id in dictionary.items():

		similarity1 = 1-spatial.distance.cosine(embeddings[word_id],curr_embed_1)
		similarity2 = 1-spatial.distance.cosine(embeddings[word_id],curr_embed_2)
		similarity3 = 1-spatial.distance.cosine(embeddings[word_id],curr_embed_3)

		heapq.heappush(similar1,(-similarity1,word))
		heapq.heappush(similar2,(-similarity2,word))
		heapq.heappush(similar3,(-similarity3,word))

	i = 0

	while(i<=num_of_similar):
		answer1.append(heapq.heappop(similar1)[1])
		answer2.append(heapq.heappop(similar2)[1])
		answer3.append(heapq.heappop(similar3)[1])
		i+=1


	return answer1,answer2,answer3

parse(embeddings,dictionary)

#a1,a2,a3 = similar_words(embeddings,dictionary,list(["first","american","would"]),21)

#print(a1)
#print(a2)
#print(a3)
