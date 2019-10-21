from nanonets import MultilabelClassification as mlc
import csv 

key = 'YOUR_API_KEY'
categories = ['car', 'road', 'plate']

model = mlc(key, categories)

files = []
labels = []
with open('../data/mlc.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	for i, row in enumerate(reader):
		if i > 0:
			files.append('../data/images/' + row[1])
			labels.append(eval(row[2]))

training_dict = dict(zip(files, labels))

resp = model.train(training_dict, data_path_type='files')

