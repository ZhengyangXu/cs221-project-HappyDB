#!/usr/bin/env python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import csv
import numpy as np
import re
filename = "data/data.csv"

def run_naive_bayes(data, labels, test_data, test_labels):
	# get data from get_features
	clf = MultinomialNB().fit(data, labels)
	predicted = clf.predict(data) # change to test_data intead of data when ready for testing
	print("Naive Bayes training accuracy: ")
	print(np.mean(predicted == labels))

def run_svm(data, labels, test_data, test_labels):
	clf = SGDClassifier(max_iter = 10).fit(data, labels)
	predicted = clf.predict(data) # change to test_data intead of data when ready for testing
	print("SVM training accuracy: ")
	print(np.mean(predicted == labels))

def get_features_bag_of_words():
	with open(filename, 'rU') as csvfile:
		reader = csv.DictReader(csvfile)
		text_data = []
		age_bucket_labels = [] # 1 for <=30, 2 for 31-60, 3 for >60
		gender_labels = [] # 1 for female, 0 male, None otherwise
		marital_labels = [] # 1 for married, 0 otherwise
		parent_labels = [] # 1 for y, 0 otherwise
		for row in reader:
			text_cleaned=row['cleaned_hm'].decode('utf-8','ignore').encode("utf-8") # get rid of non utf-8 chars
			text_data.append(' '.join(text_cleaned.split())) # strip extra whitespace
			# age labels
			if row['age'].isdigit():
				age = int(row['age'])
				if age <= 30:
					age_bucket_labels.append(1)
				elif age <= 60:
					age_bucket_labels.append(2)
				else:
					age_bucket_labels.append(3)
			else:
				age_bucket_labels.append(1) # for now, assume 1 for bad input, TODO: change later!!!
			# gender labels
			if row['gender'] == 'm':
				gender_labels.append(0)
			elif row['gender'] == 'f':
				gender_labels.append(1)
			else:
				gender_labels.append(0) # for now, assume 0 for bad input, TODO: change later
			# marital status labels
			if row['marital'] == 'married':
				marital_labels.append(1)
			else:
				marital_labels.append(0)
			# parenthood status labels
			if row['parenthood'] == 'y':
				parent_labels.append(1)
			else:
				parent_labels.append(0)
	count_vect = CountVectorizer()
	X_counts = count_vect.fit_transform(text_data)

	# tfidf_transformer = TfidfTransformer()
	# X_counts = tfidf_transformer.fit_transform(X_counts) # for ignoring length of text and reducing weight of common words
	
	training_size = int(X_counts.shape[0]*0.8) # 80:20 split for train/test data
	X_train_counts = X_counts[:training_size, :] 
	age_bucket_labels_train = age_bucket_labels[:training_size]
	marital_labels_train = marital_labels[:training_size]
	gender_labels_train = gender_labels[:training_size]
	parent_labels_train = parent_labels[:training_size]
	y_dict_train = {'age': age_bucket_labels_train, 'marital': marital_labels_train, 'gender': gender_labels_train, 'parenthood': parent_labels_train}
	X_test_counts = X_counts[training_size:, :]
	age_bucket_labels_test = age_bucket_labels[training_size:]
	marital_labels_test = marital_labels[training_size:]
	gender_labels_test = gender_labels[training_size:]
	parent_labels_test = parent_labels[training_size:]
	y_dict_test = {'age': age_bucket_labels_test, 'marital': marital_labels_test, 'gender': gender_labels_test, 'parenthood': parent_labels_test}

	return X_train_counts, y_dict_train, X_test_counts, y_dict_test

def main():
	x_train, y_dict_train, x_test, y_dict_test = get_features_bag_of_words()
	run_naive_bayes(x_train, y_dict_train['marital'], x_test, y_dict_test['marital'])
	run_naive_bayes(x_train, y_dict_train['gender'], x_test, y_dict_test['gender'])
	run_naive_bayes(x_train, y_dict_train['parenthood'], x_test, y_dict_test['parenthood'])
	run_naive_bayes(x_train, y_dict_train['age'], x_test, y_dict_test['age'])
	run_svm(x_train, y_dict_train['marital'], x_test, y_dict_test['marital'])
	run_svm(x_train, y_dict_train['gender'], x_test, y_dict_test['gender'])
	run_svm(x_train, y_dict_train['parenthood'], x_test, y_dict_test['parenthood'])
	run_svm(x_train, y_dict_train['age'], x_test, y_dict_test['age'])

if __name__ == '__main__':
	main()
