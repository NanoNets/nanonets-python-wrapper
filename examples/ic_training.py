from nanonets import ImageClassification as ic
import pandas as pd

key = 'YOUR_API_KEY'
categories = ['number_plate', 'no_plate']

model = ic(key, categories)

image_files = pd.read_csv('../data/classification.csv')
files = ['../data/images/' + x for x in image_files['files'].values]
labels = [x for x in image_files['labels'].values]

training_dict = dict(zip(files, labels))

resp = model.train(training_dict, data_path_type='files')

