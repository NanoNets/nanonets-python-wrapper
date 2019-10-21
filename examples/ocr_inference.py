from nanonets import OCR as ocr
import json
import os


key = 'YOUR_API_KEY'
categories = ['number_plate']

midocr = 'YOUR_MODEL_ID'

modocr = ocr(key, categories, model_id=midocr)

## list of file paths of several test images
imglist = os.listdir('../data/images')
imglist = ['../data/images/' + x for x in imglist]

## urls of several test images
file = open('../data/number_plates.json', 'r')
urls = []
for line in file:
	urls.append(json.loads(line)['content'])


## prediction functions for files
ocrrespone = modocr.predict_for_file(imglist[0])
print("OCR response - single image: ", ocrrespone)
ocrrespmul = modocr.predict_for_files(imglist[:5])
print("OCR response - multiple images: ", ocrrespmul)

## prediction functions for urls
ocrurlresp = modocr.predict_for_url(urls[0])
print("OCR response - single URL: ", ocrurlresp)
ocrurlsresp = modocr.predict_for_urls(urls[:5])
print("OCR response - multiple URLs: ", ocrurlsresp)
