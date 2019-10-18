from nanonets import MultilabelClassification as mlc
import json
import os


key = 'YOUR_API_KEY'
categories = ['car', 'road', 'plate']

midmlc = 'YOUR_MODEL_ID'

modmlc = mlc(key, categories, model_id=midmlc)

## list of file paths of several test images
imglist = os.listdir('../data/images')
imglist = ['../data/images/' + x for x in imglist]

## urls of several test images
file = open('../data/number_plates.json', 'r')
urls = []
for line in file:
	urls.append(json.loads(line)['content'])


## prediction functions for files
mlcrespone = modmlc.predict_for_file(imglist[0])
print("MLC response - single image: ", mlcrespone)
mlcrespmul = modmlc.predict_for_files(imglist[:39])
print("MLC response - multiple images: ", mlcrespmul)

## prediction functions for urls
mlcurlresp = modmlc.predict_for_url(urls[0])
print("MLC response - single URL: ", mlcurlresp)
mlcurlsresp = modmlc.predict_for_urls(urls[:39])
print("MLC response - multiple URLs: ", mlcurlsresp)
