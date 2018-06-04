#!/usr/bin/env python
# -*- encoding: utf8 -*-
import csv
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
import codecs
from stop_words import get_stop_words

def load_data(fileName):
	if fileName.endswith('.txt'):
		f = open(fileName, 'rU')
		lines = [line.decode('utf-8', 'ignore').encode('utf-8').rstrip() for line in f.readlines()]
		return lines
	else:
		matrix = []
		with open(fileName, 'rU') as f:
			reader = csv.reader(f)
			for row in reader:
				line = []
				for val in row:
					line.append(val.decode('utf-8', 'ignore').encode('utf-8').rstrip())
				matrix.append(line)
		return matrix

def LDA_topic_model(data):
	if len(data) == 0:
		print "No data available."
		return
	tokenizer = RegexpTokenizer(r'\w+')
	stop_words = get_stop_words('en') # words that don't add much to meaning / topic
	stop_words.extend(['happy','really','got','one','good','time','great','made','came',\
		'today','day','yesterday','went','took','get','will','happiness','just','since'])
	porter_stemmer = PorterStemmer() # treat similar words as one term (i.e. family vs. families)
	happy_moments = [data[row][1] for row in range(len(data))]	

	texts = []
	for line in happy_moments:
		raw = line.lower()
		all_tokens = tokenizer.tokenize(raw)
		# to remove stop words
		tokens_without_stop = [tok for tok in all_tokens if tok not in stop_words and len(tok)>1] 
		stemmed_tokens = [porter_stemmer.stem(i) for i in tokens_without_stop]
		texts.append(stemmed_tokens)

	word_dict = corpora.Dictionary(texts)
	word_corpus = [word_dict.doc2bow(text) for text in texts] # bag of words
	
	# goal: find optimal number of topics to have
	# runs LDA with different number of topics
	# returns LDA model with the highest topic coherence score 
	def compute_opt_model(dictionary, corpus, lines, max_num_topics, start, step):		
		max_coherence_val = float('-inf')
		opt_num_topics = None
		opt_lda_model = None
		for num_topics in range(start, max_num_topics, step):
			lda_model = gensim.models.ldamodel.LdaModel(word_corpus, num_topics, id2word=word_dict, passes=5)
			coherence_model = CoherenceModel(model=lda_model, texts=lines, corpus=corpus, dictionary=dictionary, coherence='c_v')
			coherence_val = coherence_model.get_coherence()
			print "num topics:", num_topics, "// coherence val:", coherence_val
			if coherence_val > max_coherence_val:
				max_coherence_val = coherence_val
				opt_num_topics = num_topics
				opt_lda_model = lda_model
		return opt_lda_model, opt_num_topics
	
	lda_model, opt_num_topics = compute_opt_model(dictionary=word_dict, corpus=word_corpus, lines=texts, \
		max_num_topics=20, start=3, step=3)
	topics = lda_model.print_topics(opt_num_topics, num_words=6)
	print '\n'.join('{}'.format(item) for item in topics)

def main():
	all_data = load_data('data/data.csv')
	female_data = []
	male_data = []
	single_data = []
	married_data = []
	young_data = []
	mid_data = []
	old_data = []
	parent_data = []
	not_parent_data = []

	for row in range(len(all_data)):
		entry = all_data[row]	
		try: # some people put "prefer not to say"	
			age = int(entry[3])
		except:
			age = -1
		gender = entry[4]
		marital_status = entry[5]
		parental_status = entry[6]

		if age > 0 and age < 31:
			young_data.append(entry)
		elif age < 60:
			mid_data.append(entry)
		else: # age > 60
		 	old_data.append(entry)
		if gender == 'f':
			female_data.append(entry)
		if gender == 'm':
			male_data.append(entry)
		if marital_status == 'single':
			single_data.append(entry)
		elif marital_status == 'married':
			married_data.append(entry)
		if parental_status == 'y':
			parent_data.append(entry)
		elif parental_status == 'n':
			not_parent_data.append(entry)

	print "all data:"
	LDA_topic_model(all_data)
	print "gender: female:"
	LDA_topic_model(female_data)
	print "gender: male:"
	LDA_topic_model(male_data)
	print "marital status: single:"
	LDA_topic_model(single_data)
	print "marital stats: married:"
	LDA_topic_model(married_data)
	print "age: young (< 31):"
	LDA_topic_model(young_data)
	print "age: mid (31 - 60):"
	LDA_topic_model(mid_data)
	print "age: old (> 60):"
	LDA_topic_model(old_data)
	print "parental status: parent"
	LDA_topic_model(parent_data)
	print "parental status: not parent"
	LDA_topic_model(not_parent_data)

if __name__ == '__main__':
	main()