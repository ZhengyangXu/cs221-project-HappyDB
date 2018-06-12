#!/usr/bin/env python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from keras.models import Sequential
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Embedding
from keras.models import Model
import gensim
import csv
import numpy as np
import os

filename = "data/data.csv"
glove_dir = 'glove.twitter.27B'
glove_filename = 'glove.twitter.27B.25d.txt'

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
	get_accuracy_breakdown(np.array(predicted), labels_cleaned, category)

def run_linear_svm(data, labels, test_data, test_labels, category):
	labels_cleaned = np.array(labels)
	test_labels_cleaned = np.array(test_labels)
	# filter out bad input
	data_cleaned = data[labels_cleaned != -1]
	labels_cleaned = labels_cleaned[labels_cleaned != -1]
	test_data_cleaned = test_data[test_labels_cleaned != -1]
	test_labels_cleaned = test_labels_cleaned[test_labels_cleaned != -1]
	clf = SGDClassifier(max_iter = 10).fit(data_cleaned, labels_cleaned)
	predicted = clf.predict(data_cleaned) # change to test_data intead of data when ready for testing
	print("Linear SVM training accuracy ({}): ".format(category))
	print(np.mean(predicted == labels_cleaned))
	print("Linear SVM test set accuracy ({}): ".format(category))
	print(np.mean(clf.predict(test_data_cleaned) == test_labels_cleaned))
	get_accuracy_breakdown(np.array(predicted), labels_cleaned, category)

def run_lr(data, labels, test_data, test_labels, category):
	labels_cleaned = np.array(labels)
	test_labels_cleaned = np.array(test_labels)
	# filter out bad input
	data_cleaned = data[labels_cleaned != -1]
	labels_cleaned = labels_cleaned[labels_cleaned != -1]
	test_data_cleaned = test_data[test_labels_cleaned != -1]
	test_labels_cleaned = test_labels_cleaned[test_labels_cleaned != -1]
	if category == 'age' and (2 in labels):
		multi_class = 'multinomial'
	clf = LogisticRegression(solver = 'sag').fit(data_cleaned, labels_cleaned)
	predicted = clf.predict(data_cleaned) # change to test_data intead of data when ready for testing
	print("Logistic Regression training accuracy ({}): ".format(category))
	print(np.mean(predicted == labels_cleaned))
	print("Logistic Regression test set accuracy ({}): ".format(category))
	print(np.mean(clf.predict(test_data_cleaned) == test_labels_cleaned))
	get_accuracy_breakdown(predicted, labels_cleaned, category)

def run_mlp(data, labels, test_data, test_labels, category):
	labels_cleaned = np.array(labels)
	test_labels_cleaned = np.array(test_labels)
	# filter out bad input
	data_cleaned = data[labels_cleaned != -1]
	labels_cleaned = labels_cleaned[labels_cleaned != -1]
	test_data_cleaned = test_data[test_labels_cleaned != -1]
	test_labels_cleaned = test_labels_cleaned[test_labels_cleaned != -1]
	clf = MLPClassifier(hidden_layer_sizes=(5,), early_stopping = True).fit(data_cleaned, labels_cleaned)
	predicted = clf.predict(data_cleaned) # change to test_data intead of data when ready for testing
	print("MLP training accuracy ({}): ".format(category))
	print(np.mean(predicted == labels_cleaned))
	print("MLP test set accuracy ({}): ".format(category))
	print(np.mean(clf.predict(test_data_cleaned) == test_labels_cleaned))
	# get_accuracy_breakdown(np.array(predicted), labels_cleaned, category)

def clean_data(age_buckets = 2):
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
			if not row['age'].isdigit():
				age_bucket_labels.append(-1) # bad input
			else:
				age = int(row['age'])
				if age < 17:
					age_bucket_labels.append(-1) # bad input
				elif age_buckets == 2:
					if age <= 30:
						age_bucket_labels.append(1)
					else:
						age_bucket_labels.append(0)
				elif age_buckets == 3:
					if age <= 25:
						age_bucket_labels.append(1)
					elif age <= 50:
						age_bucket_labels.append(2)
					else:
						age_bucket_labels.append(3)
				elif age_buckets == 5:
					if age <=20:
						age_bucket_labels.append(1)
					elif age <= 30:
						age_bucket_labels.append(2)
					elif age <= 40:
						age_bucket_labels.append(3)
					elif age <= 50:
						age_bucket_labels.append(4)
					else:
						age_bucket_labels.append(5)
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
	return text_data, age_bucket_labels, gender_labels, marital_labels, parent_labels

def print_age_breakdown(age_bucket_labels, age_buckets):
	print("Age bucket breakdown: ")
	age_labels = np.array(age_bucket_labels)
	for i in range(1, age_buckets+1):
		print(len(age_labels[age_labels == i]))

def binary_bag_of_words(age_buckets = 2):
	text_data, age_bucket_labels, gender_labels, marital_labels, parent_labels = clean_data(age_buckets=age_buckets)
	print_age_breakdown(age_bucket_labels, age_buckets)

	count_vect = CountVectorizer(binary=True)
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

def get_features_bag_of_words(age_buckets = 2):
	text_data, age_bucket_labels, gender_labels, marital_labels, parent_labels = clean_data(age_buckets=age_buckets)
	print_age_breakdown(age_bucket_labels, age_buckets)

	count_vect = CountVectorizer()
	X_counts = count_vect.fit_transform(text_data)

	tfidf_transformer = TfidfTransformer()
	X_counts = tfidf_transformer.fit_transform(X_counts) # for ignoring length of text and reducing weight of common words
	
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

def bag_of_words_ngram(age_buckets = 2):
	text_data, age_bucket_labels, gender_labels, marital_labels, parent_labels = clean_data(age_buckets = age_buckets)
	print_age_breakdown(age_bucket_labels, age_buckets)

	count_vect = CountVectorizer(analyzer='word', ngram_range=(2,2)) # use bigram
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

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

def word_vec(category, age_buckets=2):
	text_data, age_bucket_labels, gender_labels, marital_labels, parent_labels = clean_data(age_buckets = age_buckets)
	training_size = int(len(text_data)*0.8)
	if category == 'marital':
		labels_cleaned = np.array(marital_labels)
	elif category == 'gender':
		labels_cleaned = np.array(gender_labels)
	elif category == 'parenthood':
		labels_cleaned = np.array(parent_labels)
	elif category == 'age':
		labels_cleaned = np.array(age_bucket_labels)
	x_data = []
	y_data = []
	for i in range(len(text_data)):
		if labels_cleaned[i] != -1:
			x_data.append(text_data[i])
			y_data.append(labels_cleaned[i])
	x_train = x_data[:training_size]
	y_train = y_data[:training_size]
	x_test = x_data[training_size:]
	y_test = y_data[training_size:]
	
	# build our own
	model = gensim.models.Word2Vec(text_data, size=100)
	w2v = dict(zip(model.wv.index2word, model.wv.syn0))
	etree_w2v = Pipeline([
    	("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
    	("random forest", RandomForestClassifier(n_estimators=100, max_depth=10))])
    	etree_w2v.fit(x_train, y_train)
    	predicted = etree_w2v.predict(x_train)
    	print(np.mean(predicted == y_train))
    	predicted_test = etree_w2v.predict(x_test)
    	print(np.mean(predicted_test == y_test))

    # glove
	"""
	with open(os.path.join(glove_dir, glove_filename)) as f:
		w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
           for line in f}
    	etree_w2v = Pipeline([
    	("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
    	("linear svc", SGDClassifier(max_iter = 10))])
    	etree_w2v.fit(x_train, y_train)
    	predicted = etree_w2v.predict(x_train)
    	print(np.mean(predicted == y_train))
    	predicted_test = etree_w2v.predict(x_test)
    	print(np.mean(predicted_test == y_test))
    """

def build_word_embeddings(category, age_buckets=2):
	# doesn't work well
	text_data, age_bucket_labels, gender_labels, marital_labels, parent_labels = clean_data(age_buckets = age_buckets)
	training_size = int(len(text_data)*0.8)
	vocab_size = 100000
	encoded_docs = [one_hot(d, vocab_size) for d in text_data]
	max_length = max(len(l) for l in text_data)
	data = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

	if category == 'marital':
		labels_cleaned = np.array(marital_labels)
	elif category == 'gender':
		labels_cleaned = np.array(gender_labels)
	elif category == 'parenthood':
		labels_cleaned = np.array(parent_labels)
	elif category == 'age':
		labels_cleaned = np.array(age_bucket_labels)

	x_data = data[labels_cleaned != -1]
	labels_cleaned = labels_cleaned[labels_cleaned != -1]
	
	x_train = x_data[:training_size]
	y_train = labels_cleaned[:training_size]
	x_test = x_data[training_size:]
	y_test = labels_cleaned[training_size:]
	model = Sequential()
	model.add(Embedding(vocab_size, 8, input_length=max_length))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
	print(model.summary())
	loss, accuracy = model.evaluate(x_train, y_train, verbose=0)
	print('Accuracy: %f' % (accuracy*100))
	loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
	print('Accuracy: %f' % (accuracy*100))

def word_embeddings(category, age_buckets=2):
	# takes forever, doesn't work well at all
	text_data, age_bucket_labels, gender_labels, marital_labels, parent_labels = clean_data(age_buckets = age_buckets)
	training_size = int(len(text_data)*0.8)
	embeddings_index = {}
	with open(os.path.join(glove_dir, glove_filename)) as f:
		for line in f:
			values = line.split()
			word = values[0]
    		coefs = np.asarray(values[1:], dtype='float32')
    		embeddings_index[word] = coefs

	t = Tokenizer(num_words = 100000)
	t.fit_on_texts(text_data)
	vocab_size = len(t.word_index) + 1
	encoded_texts = t.texts_to_sequences(text_data)
	max_length = max(len(l) for l in text_data)
	data = pad_sequences(encoded_texts, maxlen=max_length, padding='post')

	if category == 'marital':
		labels_cleaned = np.array(marital_labels)
	elif category == 'gender':
		labels_cleaned = np.array(gender_labels)
	elif category == 'parenthood':
		labels_cleaned = np.array(parent_labels)
	elif category == 'age':
		labels_cleaned = np.array(age_bucket_labels)

	x_data = data[labels_cleaned != -1]
	labels_cleaned = labels_cleaned[labels_cleaned != -1]
	
	x_train = x_data[:training_size]
	y_train = labels_cleaned[:training_size]
	x_test = x_data[training_size:]
	y_test = labels_cleaned[training_size:]
	
	embedding_matrix = np.zeros((vocab_size, 100))
	for word, i in t.word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector

	embedding_layer = Embedding(vocab_size,
                            100,
                            weights=[embedding_matrix],
                            input_length=max_length,
                            trainable=False)
	convnet(embedding_layer, max_length, x_train, y_train, x_test, y_test)
	"""
	model = Sequential()
	model.add(embedding_layer)
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
	# marital
	model.fit(x_train, y_train, epochs=50, verbose=0)
	loss, accuracy = model.evaluate(x_train, y_train, verbose=0)
	print('Accuracy: %f' % (accuracy*100))

	loss_, accuracy_ = model.evaluate(x_test, y_test, verbose=0)
	print('Accuracy: %f' % (accuracy_*100))
	"""

def convnet(embedding_layer, max_length, x_train, y_train, x_test, y_test):
	# takes forever, doesn't work well at all
	sequence_input = Input(shape=(max_length,), dtype='int32')
	embedded_sequences = embedding_layer(sequence_input)
	x = Conv1D(128, 5, activation='relu')(embedded_sequences)
	x = MaxPooling1D(5)(x)
	x = Conv1D(128, 5, activation='relu')(x)
	x = MaxPooling1D(5)(x)
	x = Conv1D(128, 5, activation='relu')(x)
	x = GlobalMaxPooling1D()(x)
	x = Dense(128, activation='relu')(x)
	preds = Dense(1, activation='softmax')(x)

	model = Model(sequence_input, preds)
	model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

	model.fit(x_train, y_train,
          batch_size=128,
          epochs=5,
          validation_data=(x_test, y_test))
	loss, accuracy = model.evaluate(x_train, y_train, verbose=0)
	print('Accuracy: %f' % (accuracy*100))

	loss_, accuracy_ = model.evaluate(x_test, y_test, verbose=0)
	print('Accuracy: %f' % (accuracy_*100))

def main():
	age_buckets = 2
	x_train, y_dict_train, x_test, y_dict_test = get_features_bag_of_words(age_buckets = age_buckets) # unigram
	# x_train, y_dict_train, x_test, y_dict_test = bag_of_words_ngram(age_buckets = age_buckets) # bigram
	# x_train, y_dict_train, x_test, y_dict_test = binary_bag_of_words(age_buckets = age_buckets) # binary bag of words
	
	run_naive_bayes(x_train, y_dict_train['marital'], x_test, y_dict_test['marital'], 'marital')
	run_naive_bayes(x_train, y_dict_train['gender'], x_test, y_dict_test['gender'], 'gender')
	run_naive_bayes(x_train, y_dict_train['parenthood'], x_test, y_dict_test['parenthood'], 'parenthood')
	run_naive_bayes(x_train, y_dict_train['age'], x_test, y_dict_test['age'], 'age')
	run_linear_svm(x_train, y_dict_train['marital'], x_test, y_dict_test['marital'], 'marital')
	run_linear_svm(x_train, y_dict_train['gender'], x_test, y_dict_test['gender'], 'gender')
	run_linear_svm(x_train, y_dict_train['parenthood'], x_test, y_dict_test['parenthood'], 'parenthood')
	run_linear_svm(x_train, y_dict_train['age'], x_test, y_dict_test['age'], 'age')
	run_lr(x_train, y_dict_train['marital'], x_test, y_dict_test['marital'], 'marital')
	run_lr(x_train, y_dict_train['gender'], x_test, y_dict_test['gender'], 'gender')
	run_lr(x_train, y_dict_train['parenthood'], x_test, y_dict_test['parenthood'], 'parenthood')
	run_lr(x_train, y_dict_train['age'], x_test, y_dict_test['age'], 'age')
	"""
	# takes a long time
	run_mlp(x_train, y_dict_train['marital'], x_test, y_dict_test['marital'], 'marital')
	run_mlp(x_train, y_dict_train['gender'], x_test, y_dict_test['gender'], 'gender')
	run_mlp(x_train, y_dict_train['parenthood'], x_test, y_dict_test['parenthood'], 'parenthood')
	run_mlp(x_train, y_dict_train['age'], x_test, y_dict_test['age'], 'age')
	"""
	# word_embeddings('marital', age_buckets=age_buckets)
	# word_embeddings('gender', age_buckets=age_buckets)
	# build_word_embeddings('marital', age_buckets=age_buckets)
	# word_vec('marital', age_buckets)
	# word_vec('gender', age_buckets)
	# word_vec('parenthood', age_buckets)
	# word_vec('age', age_buckets)

if __name__ == '__main__':
	main()
