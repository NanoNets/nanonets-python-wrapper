from nanonets import OCR as ocr
import csv

key = 'YOUR_API_KEY'
categories = ['number_plate']

model = ocr(key, categories)

files = []
labels = []
with open('../data/ocr.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	for i, row in enumerate(reader):
		if i > 0:
			files.append('../data/images/' + row[1])
			labels.append('..data/annotations/json/' + row[2])

training_dict = dict(zip(files, labels))

resp = model.train(training_dict)

