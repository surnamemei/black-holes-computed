import os
W = os.path.dirname(os.path.abspath(__file__))
import json
d = json.load(open(os.path.join(W, '..', 'out', 'page_data.json')))
d['chisq'] = json.load(open(os.path.join(W, '..', 'out', 'chisq.json')))
data = json.dumps(d, separators=(',', ':'))
page = open(os.path.join(W, 'src', 'index.src.html')).read().replace('/*DATA*/null', data)
open(os.path.join(W, 'dist', 'index.html'), 'w').write(page)
print('index.html KB', round(len(page.encode()) / 1024, 1))
