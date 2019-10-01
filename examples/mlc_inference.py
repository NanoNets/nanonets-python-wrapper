import sys
sys.path.append('/Users/anuj/Desktop/nanonets-github/nanonets-pip')

from source.multilabel_classification import MultilabelClassification as mlc
import json
import os


key = 'YOUR_API_KEY'
categories = ['car', 'road', 'plate']

midmlc = 'YOUR_MODEL_ID'

mmlc = mlc(key, categories, mmlcel_id=midmlc)

## list of file paths of several test images
imglist = os.listdir('sample_data/images')
imglist = ['sample_data/images/' + x for x in imglist]

## urls of several test images
file = open('sample_data/Indian_Number_plates.json', 'r')
urls = []
for line in file:
	urls.append(json.loads(line)['content'])


## prediction functions for files
mlcrespone = mmlc.predict_for_file(imglist[0])
print("MLC response - single image: ", mlcrespone)
mlcrespmul = mmlc.predict_for_files(imglist[:39])
print("MLC response - multiple images: ", mlcrespmul)

## prediction functions for urls
mlcurlresp = mmlc.predict_for_url(urls[0])
print("MLC response - single URL: ", mlcurlresp)
mlcurlsresp = mmlc.predict_for_urls(urls[:39])
print("MLC response - multiple URLs: ", mlcurlsresp)
