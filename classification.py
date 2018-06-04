#!/usr/bin/env python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
import csv
import numpy as np

filename = "data/data.csv"

def get_accuracy_breakdown(predicted, labels, category):
	if category == 'marital':	
		print("Percent correct for married: {}".format(np.mean(predicted[labels == 1] == labels[labels == 1])))
		print("Percent correct for unmarried: {}".format(np.mean(predicted[labels == 0] == labels[labels == 0])))
	elif category == 'gender':
		print("Percent correct for male: {}".format(np.mean(predicted[labels == 0] == labels[labels == 0])))
		print("Percent correct for female: {}".format(np.mean(predicted[labels == 1] == labels[labels == 1])))
	elif category == 'age':
		print("Percent correct for < 31: {}".format(np.mean(predicted[labels == 1] == labels[labels == 1])))
		print("Percent correct for > 30: {}".format(np.mean(predicted[labels == 0] == labels[labels == 0])))
	elif category == 'parenthood':
		print("Percent correct for parents: {}".format(np.mean(predicted[labels == 1] == labels[labels == 1])))
		print("Percent correct for non-parents: {}".format(np.mean(predicted[labels == 0] == labels[labels == 0])))

def run_naive_bayes(data, labels, test_data, test_labels, category):
	labels_cleaned = np.array(labels)
	test_labels_cleaned = np.array(test_labels)
	# filter out bad input
	data_cleaned = data[labels_cleaned != -1]
	labels_cleaned = labels_cleaned[labels_cleaned != -1]
	test_data_cleaned = test_data[test_labels_cleaned != -1]
	test_labels_cleaned = test_labels_cleaned[test_labels_cleaned != -1]
	clf = MultinomialNB().fit(data_cleaned, labels_cleaned)
	predicted = clf.predict(data_cleaned) # change to test_data intead of data when ready for testing
	print("Naive Bayes training accuracy ({}): ".format(category))
	print(np.mean(predicted == labels_cleaned))
	print("Naive Bayes test set accuracy ({}): ".format(category))
	print(np.mean(clf.predict(test_data_cleaned) == test_labels_cleaned))
	# get_accuracy_breakdown(np.array(predicted), labels_cleaned, category)

def run_svm(data, labels, test_data, test_labels, category):
	labels_cleaned = np.array(labels)
	test_labels_cleaned = np.array(test_labels)
	# filter out bad input
	data_cleaned = data[labels_cleaned != -1]
	labels_cleaned = labels_cleaned[labels_cleaned != -1]
	test_data_cleaned = test_data[test_labels_cleaned != -1]
	test_labels_cleaned = test_labels_cleaned[test_labels_cleaned != -1]
	clf = SGDClassifier(max_iter = 10).fit(data_cleaned, labels_cleaned)
	predicted = clf.predict(data_cleaned) # change to test_data intead of data when ready for testing
	print("SVM training accuracy ({}): ".format(category))
	print(np.mean(predicted == labels_cleaned))
	print("SVM test set accuracy ({}): ".format(category))
	print(np.mean(clf.predict(test_data_cleaned) == test_labels_cleaned))
	# get_accuracy_breakdown(np.array(predicted), labels_cleaned, category)

def run_lr(data, labels, test_data, test_labels, category):
	labels_cleaned = np.array(labels)
	test_labels_cleaned = np.array(test_labels)
	# filter out bad input
	data_cleaned = data[labels_cleaned != -1]
	labels_cleaned = labels_cleaned[labels_cleaned != -1]
	test_data_cleaned = test_data[test_labels_cleaned != -1]
	test_labels_cleaned = test_labels_cleaned[test_labels_cleaned != -1]
	clf = LogisticRegression().fit(data_cleaned, labels_cleaned)
	predicted = clf.predict(data_cleaned) # change to test_data intead of data when ready for testing
	print("Logistic Regression training accuracy ({}): ".format(category))
	print(np.mean(predicted == labels_cleaned))
	print("Logistic Regression test set accuracy ({}): ".format(category))
	print(np.mean(clf.predict(test_data_cleaned) == test_labels_cleaned))
	# get_accuracy_breakdown(predicted, labels_cleaned, category)

def get_features_bag_of_words():
	with open(filename, 'rU') as csvfile:
		reader = csv.DictReader(csvfile)
		text_data = []
		age_bucket_labels = [] # 1 for <=30, 0 otherwise
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
				else:
					age_bucket_labels.append(0)
			else:
				age_bucket_labels.append(-1) # bad input
			# gender labels
			if row['gender'] == 'm':
				gender_labels.append(0)
			elif row['gender'] == 'f':
				gender_labels.append(1)
			else:
				gender_labels.append(-1) # bad input
			# marital status labels
			if row['marital'] == '0':
				marital_labels.append(-1) # bad input
			elif row['marital'] == 'married':
				marital_labels.append(1)
			else:
				marital_labels.append(0)
			# parenthood status labels
			if row['parenthood'] == '0':
				parent_labels.append(-1) # bad input
			elif row['parenthood'] == 'y':
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
	run_naive_bayes(x_train, y_dict_train['marital'], x_test, y_dict_test['marital'], 'marital')
	run_naive_bayes(x_train, y_dict_train['gender'], x_test, y_dict_test['gender'], 'gender')
	run_naive_bayes(x_train, y_dict_train['parenthood'], x_test, y_dict_test['parenthood'], 'parenthood')
	run_naive_bayes(x_train, y_dict_train['age'], x_test, y_dict_test['age'], 'age')
	run_svm(x_train, y_dict_train['marital'], x_test, y_dict_test['marital'], 'marital')
	run_svm(x_train, y_dict_train['gender'], x_test, y_dict_test['gender'], 'gender')
	run_svm(x_train, y_dict_train['parenthood'], x_test, y_dict_test['parenthood'], 'parenthood')
	run_svm(x_train, y_dict_train['age'], x_test, y_dict_test['age'], 'age')
	run_lr(x_train, y_dict_train['marital'], x_test, y_dict_test['marital'], 'marital')
	run_lr(x_train, y_dict_train['gender'], x_test, y_dict_test['gender'], 'gender')
	run_lr(x_train, y_dict_train['parenthood'], x_test, y_dict_test['parenthood'], 'parenthood')
	run_lr(x_train, y_dict_train['age'], x_test, y_dict_test['age'], 'age')

if __name__ == '__main__':
	main()
