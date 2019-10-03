from nanonets.classification import ImageClassification as ic
import json
import os


key = 'YOUR_API_KEY'
categories = ['number_plate', 'no_plate']

midic = 'YOUR_MODEL_ID'

modic = ic(key, categories, model_id=midic)

## list of file paths of several test images
imglist = os.listdir('data/images')
imglist = ['data/images/' + x for x in imglist]

## urls of several test images
file = open('data/number_plates.json', 'r')
urls = []
for line in file:
	urls.append(json.loads(line)['content'])


## prediction functions for files
icrespone = modic.predict_for_file(imglist[0])
print("IC response - single image: ", icrespone)
icrespmul = modic.predict_for_files(imglist[:39])
print("IC response - multiple images: ", icrespmul)

## prediction functions for urls
icurlresp = modic.predict_for_url(urls[0])
print("IC response - single URL: ", icurlresp)
icurlsresp = modic.predict_for_urls(urls[:39])
print("IC response - multiple URLs: ", icurlsresp)
