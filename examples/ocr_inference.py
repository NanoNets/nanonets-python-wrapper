import sys
sys.path.append('/Users/anuj/Desktop/nanonets-github/nanonets-python-wrapper')

from source.ocr import OCR as ocr
import json
import os

key = 'YOUR_API_KEY'
categories = ['number_plate']

midocr = 'YOUR_MODEL_ID'

modocr = ocr(key, categories, model_id=midocr)

## list of file paths of several test images
imglist = os.listdir('sample_data/images')
imglist = ['sample_data/images/' + x for x in imglist]

## urls of several test images
file = open('sample_data/Indian_Number_plates.json', 'r')
urls = []
for line in file:
	urls.append(json.loads(line)['content'])


## prediction functions for files
ocrrespone = modocr.predict_for_file(imglist[0])
print("OCR response - single image: ", ocrrespone)
ocrrespmul = modocr.predict_for_files(imglist[:39])
print("OCR response - multiple images: ", ocrrespmul)

## prediction functions for urls
ocrurlresp = modocr.predict_for_url(urls[0])
print("OCR response - single URL: ", ocrurlresp)
ocrurlsresp = modocr.predict_for_urls(urls[:39])
print("OCR response - multiple URLs: ", ocrurlsresp)
