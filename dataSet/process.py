import os
import json
from collections import OrderedDict

data_json = json.load(open('train.json', encoding='utf8'), object_pairs_hook=OrderedDict)
data = {}
for line in data_json:
	if line['domain'] not in data:
		data[line['domain']] = {}
	if line['intent'] not in data[line['domain']]:
		if(line['intent'] != line['intent']):
			line['intent'] = 'NaN'
		data[line['domain']][line['intent']] = []
	data[line['domain']][line['intent']].append([line['text'], line['slots']]) 

data_pro = []
for k, v in data.items():
	for kk, vv in v.items():
		for vvv in vv:
			d = OrderedDict()
			d['text'] = vvv[0]
			d['domain'] = k
			d['intent'] = kk
			d['slots'] = vvv[1]
			for i in range(215//len(vv)):
				data_pro.append(d)

os.makedirs('data/domain')
for domain in data:
	os.makedirs('data/domain/' + domain)
	for intent in data[domain]:
		dic = []
		for ele in data[domain][intent]:
			d = collections.OrderedDict()
		    d["text"] = ele[0]
		    d["domain"] = domain
		    d["intent"] = intent
		    d["slots"] = ele[1]
			dic.append(d)
		json.dump(dic, open('data/'+domain+'/'+intent+'.json', 'w'), ensure_ascii = False, indent = 2)

os.makedirs('data/intent')
for intent in data:
    os.makedirs('data/intent/' + intent)
    for domain in data[intent]:
        dic = []
        for ele in data[intent][domain]:
            d = collections.OrderedDict()
            d["text"] = ele[0]
            d["domain"] = domain
            d["intent"] = intent
            d["slots"] = ele[1]
            dic.append(d)
        json.dump(dic, open('data/intent/'+intent+'/'+domain+'.json', 'w'), ensure_ascii = False, indent = 2)

slots = {}
for data in data_json:
	for slot, val in data['slots'].items():
		if slot not in slots:
			slots[slot] = set()
		slots[slot].add(val)
			

os.makedirs('data/slots')
for slot in slots:
	with open("data/slots/" + slot + ".txt", 'w') as f:
		for s in slots[slot]:
			f.write(s + '\n')

# *************************************************************************************** #			
train_json = json.load(open('train_local.json'), encoding='utf8')
text = []
for line in train_json:
	text.append(line['text'])

data = []
for line in data_json:
	if line['text'] in text:
		continue
	data.append(line)

json.dump(data, open('test_local.json', 'w'), ensure_ascii = False, indent = 2)


# *************************************************************************************** #			
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

data_json = json.load(open('train.json'), encoding='utf8')
text = np.array([data['text'] for data in data_json])
domain = np.array([data['domain'] for data in data_json])
intent = np.array([data['intent'] for data in data_json])
label = [d+i for d, i in zip(domain, intent)]
kfold = list(StratifiedKFold(n_splits = 2, random_state = 2019, shuffle = True).split(text, label))
train_index, val_index = kfold[0]
text = text[train_index]

train_json, test_json = [], []
for data in data_json:
	if data['text'] in text:
		train_json.append(data)
	else:
		test_json.append(data)

json.dump(train_json, open('train_eval.json', 'w'), ensure_ascii = False, indent = 2)
json.dump(test_json, open('test_eval.json', 'w'), ensure_ascii = False, indent = 2)

# *************************************************************************************** #		
import json
from collections import OrderedDict
data_json1 = json.load(open('result1/test_result.json', encoding='utf8'), object_pairs_hook=OrderedDict)
data_json2 = json.load(open('result2/test_result.json', encoding='utf8'), object_pairs_hook=OrderedDict)
data_json3 = json.load(open('result3/test_result.json', encoding='utf8'), object_pairs_hook=OrderedDict)
data_json = []
for data1, data2, data3 in zip(data_json1, data_json2, data_json3):
    d = OrderedDict()
    d['text'] = data1['text']
    d['domain'] = data1['domain']
    d['intent'] = data2['intent']
    d['slots'] = data3['slots']
    data_json.append(d)

json.dump(data_json, open('result/test_result.json', 'w'), ensure_ascii = False, indent = 2)

# *************************************************************************************** #	
import json
from collections import OrderedDict

data_json = json.load(open('train_eval.json', encoding='utf8'), object_pairs_hook=OrderedDict)
dic = {}
for data in data_json:
	if data['domain'] not in dic:
		dic[data['domain']] = []
	dic[data['domain']].append(data)

length = max([len(v) for k,v in dic.items()])
new_data = []
for k,v in dic.items():
	times = length // len(v)
	for i in range(times):
		new_data.extend(v)

json.dump(new_data, open('result/train1.json', 'w'), ensure_ascii = False, indent = 2)

# *************************************************************************************** #	
dic = {}
for data in data_json:
	if data['domain'] not in dic:
		dic[data['domain']] = [set(), set()]
	dic[data['domain']][0].add(data['intent'])
	for slot in data['slots']:
		dic[data['domain']][1].add(slot)

for k, v in dic.items():
    print('\''+k+'\':' ,v, ',')