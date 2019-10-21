from nanonets import ObjectDetection as od
import json
import os


key = 'YOUR_API_KEY'
categories = ['number_plate']

midod = 'YOUR_MODEL_ID'

modod = od(key, categories, model_id=midod)

## list of file paths of several test images
imglist = os.listdir('../data/images')
imglist = ['../data/images/' + x for x in imglist]

## urls of several test images
file = open('../data/number_plates.json', 'r')
urls = []
for line in file:
	urls.append(json.loads(line)['content'])


## prediction functions for files
odrespone = modod.predict_for_file(imglist[0])
print("OD response - single image: ", odrespone)
odrespmul = modod.predict_for_files(imglist[:5])
print("OD response - multiple images: ", odrespmul)

## prediction functions for urls
odurlresp = modod.predict_for_url(urls[0])
print("OD response - single URL: ", odurlresp)
odurlsresp = modod.predict_for_urls(urls[:5])
print("OD response - multiple URLs: ", odurlsresp)
