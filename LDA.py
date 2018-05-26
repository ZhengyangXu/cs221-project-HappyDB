#!/usr/bin/env python
# -*- encoding: utf8 -*-
import csv
import nltk
from nltk.tokenize import RegexpTokenizer
import gensim
from gensim import corpora
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
		'today','day','yesterday','went','took','get','will','happiness','just'])
	happy_moments = [data[row][1] for row in range(len(data))]	

	texts = []
	for line in happy_moments:
		raw = line.lower()
		all_tokens = tokenizer.tokenize(raw)
		tokens = [tok for tok in all_tokens if tok not in stop_words and len(tok)>1]
		texts.append(tokens)

	dictionary = corpora.Dictionary(texts)
	corpus = [dictionary.doc2bow(text) for text in texts] # bag of words
	num_topics = 9 # arbitrary for now
	ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics, id2word=dictionary, passes=20)
	topics = ldamodel.print_topics(num_topics, num_words=6)
	print '\n'.join('{}'.format(item) for item in topics)

def main():
	all_data = load_data('data.csv')
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
		elif parental_status == 'n'
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