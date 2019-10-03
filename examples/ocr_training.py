from pynanonets import OCR as ocr
import pandas as pd

key = 'YOUR_API_KEY'
categories = ['number_plate']

model = ocr(key, categories)

image_files = pd.read_csv('../data/ocr.csv')
files = ['../data/images/' + x for x in image_files['files'].values]
labels = ['../data/annotations/json/' + x for x in image_files['labels'].values]

training_dict = dict(zip(files, labels))

resp = model.train(training_dict)

