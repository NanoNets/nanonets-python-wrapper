import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from nanonets.multilabel_classification import MultilabelClassification as mlc
import json


key = 'YOUR_API_KEY'
categories = ['car', 'road', 'plate']

midmlc = 'YOUR_MODEL_ID'

mmlc = mlc(key, categories, mmlcel_id=midmlc)

## list of file paths of several test images
imglist = os.listdir('data/images')
imglist = ['data/images/' + x for x in imglist]

## urls of several test images
file = open('data/number_plates.json', 'r')
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
