from pynanonets import ObjectDetection as od
import pandas as pd

key = 'YOUR_API_KEY'
categories = ['number_plate']

model = od(key, categories)

image_files = pd.read_csv('../data/ocr.csv')
files = ['../data/ocr.csv' + x for x in image_files['files'].values]
labels = [x for x in image_files['labels'].values]

training_dict = dict(zip(files, labels))

resp = model.train(training_dict)

