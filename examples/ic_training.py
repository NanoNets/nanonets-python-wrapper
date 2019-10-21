from nanonets import ImageClassification as ic
import csv

key = 'YOUR_API_KEY'
categories = ['number_plate', 'no_plate']

model = ic(key, categories)

files = []
labels = []
with open('../data/ic.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	for i, row in enumerate(reader):
		if i > 0:
			files.append('../data/images/' + row[1])
			labels.append(row[2])

training_dict = dict(zip(files, labels))

resp = model.train(training_dict, data_path_type='files')

