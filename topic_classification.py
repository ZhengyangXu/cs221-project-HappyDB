#!/usr/bin/env python
# -*- encoding: utf8 -*-
import csv
import nltk
from nltk.tokenize import RegexpTokenizer
import codecs
from stop_words import get_stop_words
import numpy as np
import random

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

entertainment = load_data('data/topic_dict/entertainment-dict.txt')
exercise = load_data('data/topic_dict/exercise-dict.txt')
family = load_data('data/topic_dict/family-dict.txt')
food = load_data('data/topic_dict/food-dict.txt')
people = load_data('data/topic_dict/people-dict.txt')
pets = load_data('data/topic_dict/pets-dict.txt')
school = load_data('data/topic_dict/school-dict.txt')
shopping = load_data('data/topic_dict/shopping-dict.txt')
work = load_data('data/topic_dict/work-dict.txt')

tokenizer = RegexpTokenizer(r'\w+')
stop_words = get_stop_words('en')
stop_words.extend(['happy','really','got','one','good','time','great','made','came',\
		'today','day','yesterday','went','morning','took','get'])


def label_category(data):
	happy_moments = [data[row][1] for row in range(len(data))]

	# counts for each of the 9 categories
	# [0: entertainment, 1: exercise, 2: family, 3: food, 4: people, 5: pets, \
	#  6: school, 7: shopping, 8: work]
	categories = ['entertainment', 'exercise', 'family', 'food', 'people', 'pets',\
	'school', 'shopping', 'work']

	assignments = {} # moment -> category
	category_clusters = {category:[] for category in categories} # category -> list of moments

	for row in range(len(data)):
		moment = data[row][1]
		counts = np.zeros(9)
		raw = moment.lower()
		all_tokens = tokenizer.tokenize(raw)
		tokens = [tok for tok in all_tokens if tok.lower() not in stop_words]
		for tok in tokens:
			# used if instead of elif in case words appeared in multiple categories
			# but can be improved
			if tok in entertainment:
				counts[0] += 1
			if tok in exercise:
				counts[1] += 1
			if tok in family:
				counts[2] += 1
			if tok in food:
				counts[3] += 1
			if tok in people:
				counts[4] += 1
			if tok in pets:
				counts[5] += 1
			if tok in school:
				counts[6] += 1
			if tok in shopping:
				counts[7] += 1	
			if tok in work:
				counts[8] += 1

		if np.all(counts==0):
			top_category = categories[random.randint(0,8)]
		else :
			top_category = categories[np.argmax(counts)]

		assignments[moment] = top_category
		category_clusters[top_category].append(moment)

	# for line in assignments:
	# 	print line, "// CATEGORY:", assignments[line]
	for category in category_clusters:
		print category, ":", category_clusters[category]

def main():
	all_data = load_data('data/data.csv')
	label_category(all_data)

if __name__ == '__main__':
	main()