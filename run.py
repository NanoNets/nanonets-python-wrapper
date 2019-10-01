from source.multilabel_classification import MultilabelClassification as mlc
from source.classification import ImageClassification as ic
from source.object_detection import ObjectDetection as od
from source.ocr import OCR as ocr
import json
import os


key = 'Dr8oVpQ78jXj6SlmmSPzS2QpfvyssWE5'
categories = ['number_plate']
iccategories = ['number_plate', 'no_plate']
mlccategories = ['car', 'road', 'plate']

midocr = '2220cc90-fc90-48c7-b985-211c656f4be6'
midod = '741b4f18-8c2a-4f57-b677-9895e51b136b'
midic = 'd45e41db-91aa-4844-8294-6aa446407ca9'
midmlc = 'f24c6171-d815-4a05-abb8-19fc03bc1599'

modocr = ocr(key, categories, model_id=midocr)
modod = od(key, categories, model_id=midod)
modic = ic(key, iccategories, model_id=midic)
modmlc = mlc(key, mlccategories, model_id=midmlc)

imgpath = 'sample_data/images/10.jpg'
imglist = os.listdir('tests/sample_data/images')[:39]
imgbatch = ['sample_data/images/' + x for x in imglist]

file = open('sample_data/Indian_Number_plates.json', 'r')
urls = []
for line in file:
	urls.append(json.loads(line)['content'])
url = urls[0]

icrespone = modic.predict_for_file(imgpath)
print("IC response - single image: ", icrespone)
icrespmul = modic.predict_for_files(imgbatch)
print("IC response length - multiple images: ", len(icrespmul))
icurlresp = modic.predict_for_url(url)
print("IC response - single URL: ", icurlresp)
icurlsresp = modic.predict_for_urls(urls[:39])
print("IC response length - multiple URLs: ", len(icurlsresp))

mlcrespone = modmlc.predict_for_file(imgpath)
print("MLC response - single image: ", mlcrespone)
mlcrespmul = modmlc.predict_for_files(imgbatch)
print("MLC response length - multiple images: ", len(mlcrespmul))
mlcurlresp = modmlc.predict_for_url(url)
print("MLC response - single URL: ", mlcurlresp)
mlcurlsresp = modmlc.predict_for_urls(urls[:39])
print("MLC response length - multiple URLs: ", len(mlcurlsresp))

ocrrespone = modocr.predict_for_file(imgpath)
print("OCR response - single image: ", ocrrespone)
ocrrespmul = modocr.predict_for_files(imgbatch)
print("OCR response length - multiple images: ", len(ocrrespmul))
ocrurlresp = modocr.predict_for_url(url)
print("OCR response - single URL: ", ocrurlresp)
ocrurlsresp = modocr.predict_for_urls(urls[:39])
print("OCR response length - multiple URLs: ", len(ocrurlsresp))

odrespone = modod.predict_for_file(imgpath)
print("OD response - single image: ", odrespone)
odrespmul = modod.predict_for_files(imgbatch)
print("OD response length - multiple images: ", len(odrespmul))
odurlresp = modod.predict_for_url(url)
print("OD response - single URL: ", odurlresp)
odurlsresp = modod.predict_for_urls(urls[:39])
print("OD response length - multiple URLs: ", len(odurlsresp))
