from nanonets import MultilabelClassification as mlc
import pandas as pd

key = 'YOUR_API_KEY'
categories = ['car', 'road', 'plate']

model = mlc(key, categories)

image_files = pd.read_csv('../data/mlc.csv')
files = ['../data/images/' + x for x in image_files['files'].values]
labels = [eval(x) for x in image_files['labels'].values]

training_dict = dict(zip(files, labels))

resp = model.train(training_dict, data_path_type='files')

