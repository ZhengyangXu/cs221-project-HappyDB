#!/usr/bin/env python
import csv
import random
import numpy

filename = "data/data.csv"

def run_baseline():
	num_correct_age = 0
	num_correct_gender = 0
	num_correct_marital = 0
	num_correct_parenthood = 0
	total_rows = 0
	total_rows_age = 0
	total_rows_gender = 0
	total_rows_marital = 0
	total_rows_parenthood = 0
	with open(filename, 'rU') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			total_rows += 1
			# marital status
			if row['marital'] != '0':
				total_rows_marital += 1
				married = False
				married_strings = ['wife', 'husband', 'spouse']
				if any(string in row['cleaned_hm'] for string in married_strings):
					married = True
				if married and row['marital'] == 'married':
					num_correct_marital += 1
				elif not married and row['marital'] != 'married':
					num_correct_marital += 1
			# parenthood status
			parent_strings = ['children', 'kids', 'daughter', 'son', 'child']
			parent = False
			if row['parenthood'] in ['y', 'n']:
				total_rows_parenthood += 1
				if any(string in row['cleaned_hm'] for string in parent_strings):
					parent = True
				if parent and row['parenthood'] == 'y':
					num_correct_parenthood += 1
				elif not parent and row['parenthood'] == 'n':
					num_correct_parenthood += 1
			# age prediction
			young_strings = ['girlfriend', 'boyfriend', 'girl friend', 'boy friend', 'school']
			if row['age'].isdigit():
				total_rows_age += 1
				correct_age = int(row['age'])
				age_classified = False
				if correct_age < 31 and any(string in row['cleaned_hm'] for string in young_strings):
					age_classified = True
					num_correct_age += 1
				elif correct_age > 30 and (parent or married):
					age_classified = True
					num_correct_age += 1
				elif not age_classified:
					rand_age = random.randint(1,2)
					if (rand_age == 1 and correct_age < 31) or (rand_age == 2 and correct_age > 30):
						num_correct_age += 1
			# gender prediction
			if row['gender'] in ['f', 'm']:
				total_rows_gender += 1
				female_strings = ['boyfriend', 'husband', 'boy friend']
				male_strings = ['girlfriend', 'wife', 'girl friend']
				gender = ''
				if any(string in row['cleaned_hm'] for string in female_strings):
					gender = 'f'
				elif any(string in row['cleaned_hm'] for string in male_strings):
					gender = 'm'
				elif gender == '':
					gender = random.choice(['f', 'm'])
				if gender == row['gender']:
					num_correct_gender += 1
	marital_percent_correct = num_correct_marital/float(total_rows_marital)
	parenthood_percent_correct = num_correct_parenthood/float(total_rows_parenthood)
	age_percent_correct = num_correct_age/float(total_rows_age)
	gender_percent_correct = num_correct_gender/float(total_rows_gender)
	print("Percentage of correct marital status predictions: {} (total: {})".format(marital_percent_correct, total_rows_marital))
	print("Percentage of correct parenthood status predictions: {} (total: {})".format(parenthood_percent_correct, total_rows_parenthood))
	print("Percentage of correct age bucket predictions: {} (total: {})".format(age_percent_correct, total_rows_age))
	print("Percentage of correct gender predictions: {} (total: {})".format(gender_percent_correct, total_rows_gender))

if __name__ == '__main__':
	run_baseline()

	

