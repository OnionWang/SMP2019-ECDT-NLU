import json
import os
import re
from collections import OrderedDict
from domain_rule import domain_rule

def cookbook(text, pred, dishName):
	slots = {}
	for k, v in pred['slots'].items():
		if k == 'dishName' and v not in dishName:
			slots['ingredient'] = v
		elif k == 'ingredient' and v in dishName and k not in slots: 
			slots['dishName'] = v
		else:
			slots[k] = v

	return slots


def bus(text, pred):
	station = []
	for k, v in pred['slots'].items():
		station.append((v, re.search(v, text).span()[0]))
	slots = {}
	if len(station) == 1:
		slots = {'Dest': station[0][0]}
	else:
		station.sort(key = lambda x:x[1])
		slots = {'Src':station[0][0], 'Dest':station[1][0]}

	return slots


def train(text, pred, province, city, railway_station):
	location = set({'startLoc_area', 'endLoc_area', 'startLoc_city', 'endLoc_city', 'startLoc_province', 'endLoc_province', 'startLoc_poi', 'endLoc_poi'})
	slots = {}
	for k, v in pred['slots'].items():
		name = k
		if k in location:
			prefix = k.split('_')[0]
			if v in province:
				name = prefix + '_province'
			elif v in city:
				name = prefix + '_city'
			elif v in railway_station:
				name = prefix + '_area'
			else:
				name = prefix + '_poi'
		slots[name] = v

	return slots


def map_pos(text, pred, province, city, railway_station):
	location = set({'location_province', 'location_city', 'location_poi', 'location_area'})
	slots = {}
	for k, v in pred['slots'].items():
		name = k
		if k in location:
			prefix = 'location'
			if v in province:
				name = prefix + '_province'
			elif v in city:
				name = prefix + '_city'
			elif v in railway_station:
				name = prefix + '_area'
			else:
				name = prefix + '_poi'
		slots[name] = v

	return slots


def process(result, dic_dir):
	result = domain_rule(result,dic_dir)
	table = {
		'music': [{'PLAY', 'SEARCH'}, {'category', 'song', 'artist'}] ,
		'match': [{'QUERY'}, {'datetime_date', 'type', 'homeName', 'name', 'category', 'awayName'}] ,
		'joke': [{'QUERY'}, set()] ,
		'weather': [{'QUERY'}, {'subfocus', 'datetime_date', 'location_city', 'questionWord'}] ,
		'novel': [{'QUERY'}, {'category', 'name', 'popularity', 'author'}] ,
		'flight': [{'QUERY'}, {'startLoc_area', 'startLoc_city', 'startDate_date', 'startDate_date', 'startDate_time', 'endLoc_area', 'endLoc_poi', 'startLoc_poi', 'endLoc_city'}] ,
		'health': [{'QUERY'}, {'keyword'}] ,
		'poetry': [{'QUERY', 'DEFAULT'}, {'keyword', 'queryField', 'dynasty', 'name', 'author'}] ,
		'video': [{'QUERY'}, {'season', 'scoreDescr', 'datetime_date', 'tag', 'popularity', 'name', 'artist', 'category', 'timeDescr', 'date', 'decade', 'resolution', 'artistRole', 'payment', 'area', 'episode'}] ,
		'epg': [{'QUERY', 'LOOK_BACK'}, {'datetime_date', 'tvchannel', 'name', 'category', 'code', 'datetime_time', 'area'}] ,
		'message': [{'SENDCONTACTS', 'VIEW', 'SEND'}, {'content', 'receiver', 'name', 'category', 'teleOperator', 'headNum'}] ,
		'contacts': [{'QUERY', 'CREATE'}, {'name', 'code'}] ,
		'stock': [{'CLOSEPRICE_QUERY', 'QUERY', 'RISERATE_QUERY'}, {'yesterday', 'name', 'code'}] ,
		'radio': [{'LAUNCH'}, {'category', 'location_province', 'name', 'code'}] ,
		'telephone': [{'QUERY', 'DIAL'}, {'teleOperator', 'category', 'name'}] ,
		'map': [{'POSITION', 'ROUTE'}, {'startLoc_area', 'startLoc_city', 'endLoc_province', 'location_city', 'type', 'endLoc_area', 'location_poi', 'endLoc_poi', 'startLoc_poi', 'location_area', 'location_province', 'endLoc_city'}] ,
		'cinemas': [{'QUERY', 'DATE_QUERY'}, {'datetime_date', 'location_city', 'film', 'name', 'theatre', 'timeDescr', 'category', 'datetime_time'}] ,
		'lottery': [{'QUERY', 'NUMBER_QUERY'}, {'category', 'relIssue', 'datetime_date', 'absIssue', 'name'}] ,
		'story': [{'QUERY'}, {'category'}] ,
		'translation': [{'TRANSLATION'}, {'target', 'content'}] ,
		'tvchannel': [{'PLAY'}, {'category', 'resolution', 'name', 'code'}] ,
		'cookbook': [{'QUERY'}, {'keyword', 'dishName', 'ingredient', 'utensil'}] ,
		'bus': [{'QUERY'}, {'Dest', 'Src'}] ,
		'email': [{'REPLY', 'LAUNCH', 'REPLAY_ALL', 'FORWARD', 'CREATE', 'SEND'}, {'content', 'name'}] ,
		'app': [{'QUERY', 'DOWNLOAD', 'LAUNCH'}, {'name'}] ,
		'train': [{'QUERY'}, {'startLoc_area', 'startLoc_city', 'startDate_date', 'endLoc_province', 'startDate_time', 'endLoc_area', 'category', 'startLoc_province', 'startLoc_poi', 'endLoc_city'}] ,
		'website': [{'OPEN'}, {'name'}] ,
		'news': [{'PLAY'}, {'datetime_date', 'location_city', 'location_country', 'keyword', 'category', 'datetime_time', 'media', 'location_province'}] ,
		'riddle': [{'QUERY'}, {'category'}] 
	}


	# app = set([a.strip() for a in open(os.path.join(dic_dir, "app.txt"), 'r').readlines()])
	# website = set([a.strip() for a in open(os.path.join(dic_dir, "website.txt"), 'r').readlines()])
	dishName = set([a.strip() for a in open(os.path.join(dic_dir, "dishName.txt"), 'r', encoding='UTF-8').readlines()])
	province = set([a.strip() for a in open(os.path.join(dic_dir, "province.txt"), 'r', encoding='UTF-8').readlines()])
	city = set([a.strip() for a in open(os.path.join(dic_dir, "city.txt"), 'r', encoding='UTF-8').readlines()])
	railway_station = set([a.strip() for a in open(os.path.join(dic_dir, "railway_station.txt"), 'r', encoding='UTF-8').readlines()])


	for i, pred in enumerate(result):
		text = pred['text']
		if pred['domain'] == 'app':
			name = pred['slots'].get('name', None)
			if re.search('下载', text):
				result[i]['intent'] = 'DOWNLOAD'
			elif re.search('搜索', text) or re.search('找到', text):
				result[i]['intent'] = 'QUERY'
			elif re.search('打开', text) or re.search('开启', text) or re.search('启动', text) or re.search('进入', text):
				result[i]['intent'] = 'LAUNCH'
			elif name and (re.match('.*搜.*' + name + '.*', text) or re.match('.*找.*' + name + '.*', text)):
				result[i]['intent'] = 'QUERY'
			else:
				result[i]['intent'] = 'LAUNCH'
			if name:
				pred['slots']['name'] = name.lower()

		elif pred['domain'] == 'health':
			result[i]['intent'] = 'QUERY'

		elif pred['domain'] == 'joke':
			result[i]['intent'] = 'QUERY'

		elif pred['domain'] == 'cookbook':		
			result[i]['slots'] = cookbook(text, pred, dishName)

		elif pred['domain'] == 'email':
			if re.search('转发', text):
				result[i]['intent'] = 'FORWARD'
			elif re.search('全部', text):
				result[i]['intent'] = 'REPLAY_ALL'
			elif re.search('回复', text) or re.search('答复', text):
				result[i]['intent'] = 'REPLY'
			elif re.search('发邮件', text):
				result[i]['intent'] = 'SEND'
				cnt = re.search('说他', text)
				if not cnt: 
					cnt = re.search('说她', text)
				if not cnt: 
					cnt = re.search('叫他', text)
				if not cnt: 
					cnt = re.search('叫她', text)
				if not cnt: 
					cnt = re.search('说', text)
				if not cnt: 
					cnt = re.search('叫', text)
				if cnt:
					cnt = cnt.span()
					if len(text) > cnt[0]:
						result[i]['slots']['content'] = text[cnt[1]:]
			elif re.search('打开', text) or re.search('开启', text) or re.search('查看', text) or re.search('给我看', text):
				result[i]['intent'] = 'LAUNCH'
			elif re.search('写', text):
				result[i]['intent'] = 'CREATE'

		elif pred['domain'] == 'novel':
			result[i]['intent'] = 'QUERY'

		elif pred['domain'] == 'poetry':
			# if not pred['slots'] == {}:
			# 	result[i]['intent'] = 'QUERY'
			# else:
			# 	result[i]['intent'] = 'DEFAULT'
			pass

		elif pred['domain'] == 'radio':
			result[i]['intent'] = 'LAUNCH'
			result[i]['slots'] = map_pos(text, pred, province, city, railway_station)

		elif pred['domain'] == 'riddle':
			result[i]['intent'] = 'QUERY'

		elif pred['domain'] == 'story':
			result[i]['intent'] = 'QUERY'

		elif pred['domain'] == 'website':
			result[i]['intent'] = 'OPEN'

		elif pred['domain'] == 'weather':
			result[i]['intent'] = 'QUERY'
			result[i]['slots'] = map_pos(text, pred, province, city, railway_station)
			time = set({'datetime_date', 'datetime_time', 'startDate_date', 'startDate_time'})
			slots = {}
			for k, v in pred['slots'].items():
				name = k
				if k in time:
					name = 'datetime_' + k.split('_')[-1]
				slots[name] = v
			result[i]['slots'] = slots
		#************************ bus, train, flight, map, news ************************# 
		elif pred['domain'] == 'bus':
			if re.search('动车', text) or re.search('高铁', text):
				result[i]['domain'] = 'train'
				result[i]['intent'] = 'QUERY'
				result[i]['slots'] = train(text, pred, province, city, railway_station)
				time = set({'datetime_date', 'datetime_time', 'startDate_date', 'startDate_time'})
				slots = {}
				for k, v in pred['slots'].items():
					name = k
					if k in time:
						name = 'startDate_' + k.split('_')[-1]
					slots[name] = v
				result[i]['slots'] = slots	
			elif re.search('飞机', text) or re.search('航班', text) or re.search('机票', text):
				result[i]['domain'] = 'flight'
				result[i]['intent'] = 'QUERY'
				result[i]['slots'] = train(text, pred, province, city, railway_station)
			else:
				result[i]['intent'] = 'QUERY'
				result[i]['slots'] = bus(text, pred)

		elif pred['domain'] == 'train':
			if re.search('汽车', text):
				result[i]['domain'] = 'bus'
				result[i]['intent'] = 'QUERY'
				result[i]['slots'] = bus(text, pred)
			elif re.search('飞机', text) or re.search('航班', text) or re.search('机票', text):
				result[i]['domain'] = 'flight'
				result[i]['intent'] = 'QUERY'
				result[i]['slots'] = train(text, pred, province, city, railway_station)
			else:
				result[i]['intent'] = 'QUERY'
				result[i]['slots'] = train(text, pred, province, city, railway_station)
				time = set({'datetime_date', 'datetime_time', 'startDate_date', 'startDate_time'})
				slots = {}
				for k, v in pred['slots'].items():
					name = k
					if k in time:
						name = 'startDate_' + k.split('_')[-1]
					slots[name] = v
				result[i]['slots'] = slots

		elif pred['domain'] == 'flight':
			if re.search('动车', text) or re.search('高铁', text):
				result[i]['domain'] = 'train'
				result[i]['intent'] = 'QUERY'
				result[i]['slots'] = train(text, pred, province, city, railway_station)
				time = set({'datetime_date', 'datetime_time', 'startDate_date', 'startDate_time'})
				slots = {}
				for k, v in pred['slots'].items():
					name = k
					if k in time:
						name = 'startDate_' + k.split('_')[-1]
					slots[name] = v
				result[i]['slots'] = slots
			elif re.search('汽车', text):
				result[i]['domain'] = 'bus'
				result[i]['intent'] = 'QUERY'
				result[i]['slots'] = bus(text, pred)
			else:
				result[i]['intent'] = 'QUERY'
				result[i]['slots'] = train(text, pred, province, city, railway_station)

		elif pred['domain'] == 'map':
			if pred['intent'] == 'POSITION':
				result[i]['slots'] = map_pos(text, pred, province, city, railway_station)
			elif pred['intent'] == 'ROUTE':
				result[i]['slots'] = train(text, pred, province, city, railway_station)

		elif pred['domain'] == 'news':
			result[i]['intent'] = 'PLAY'
			result[i]['slots'] = map_pos(text, pred, province, city, railway_station)
			time = set({'datetime_date', 'datetime_time', 'startDate_date', 'startDate_time'})
			slots = {}
			for k, v in pred['slots'].items():
				name = k
				if k in time:
					name = 'datetime_' + k.split('_')[-1]
				slots[name] = v
			result[i]['slots'] = slots
			
		#***********************************************************************# 

		elif pred['domain'] == 'translation':
			result[i]['intent'] = 'TRANSLATION'

		elif pred['domain'] == 'cinemas':
			if re.search('什么时候', text) or re.search('何时', text):
				result[i]['intent'] = 'DATE_QUERY'
			else:
				result[i]['intent'] = 'QUERY'
				time = set({'datetime_date', 'datetime_time', 'startDate_date', 'startDate_time'})
				slots = {}
				for k, v in pred['slots'].items():
					name = k
					if k in time:
						name = 'datetime_' + k.split('_')[-1]
					slots[name] = v
				result[i]['slots'] = slots

		elif pred['domain'] == 'video':
			result[i]['intent'] = 'QUERY'

		elif pred['domain'] == 'contacts':
			if re.search('新建', text) or re.search('添加', text):
				result[i]['intent'] = 'CREATE'
			else:
				result[i]['intent'] = 'QUERY'

		elif pred['domain'] == 'telephone':
			telep = ['移动', '联通', '电信']
			# if re.search('呼叫', text) or re.search('打', text) or re.search('拨', text):
			# 	result[i]['intent'] = 'DIAL'
			# else:
			# 	result[i]['intent'] = 'QUERY'
			if re.search('hello word', text.lower()):
				span = re.search('hello word', text.lower()).span()
				result[i]['slots']['name'] = text[span[0]:span[1]]
			for t in telep:
				if 'teleOperator' in result[i]['slots']:
					break
				if re.search(t, text):
					span = re.search(t, text).span()
					result[i]['slots']['teleOperator'] = text[span[0]:span[1]]
					

		elif pred['domain'] == 'message':
			# if re.search('查看', text):
			# 	result[i]['intent'] = 'VIEW'
			# elif re.search('电话', text) or re.search('号码', text):
			# 	result[i]['intent'] = 'SENDCONTACTS'
			# elif re.search('短信', text) or re.search('消息', text) or re.search('简讯', text) or re.search('信息', text) or re.search('短讯', text):
			# 	result[i]['intent'] = 'SEND'
			# 	cnt = re.search('说他', text)
			# 	if not cnt: 
			# 		cnt = re.search('说她', text)
			# 	if not cnt: 
			# 		cnt = re.search('叫他', text)
			# 	if not cnt: 
			# 		cnt = re.search('叫她', text)
			# 	if not cnt: 
			# 		cnt = re.search('说', text)
			# 	if not cnt: 
			# 		cnt = re.search('叫', text)
			# 	if cnt and 'content' not in result[i]['slots']:
			# 		cnt = cnt.span()
			# 		if len(text) > cnt[0]:
			# 			result[i]['slots']['content'] = text[cnt[1]:]
			pass

		elif pred['domain'] == 'tvchannel':
			result[i]['intent'] = 'PLAY'

		elif pred['domain'] == 'epg':
			if re.search('回放', text) or re.search('回看', text):
				result[i]['intent'] = 'LOOK_BACK'
			else:
				result[i]['intent'] = 'QUERY'

		elif pred['domain'] == 'lottery':
			if pred['slots'] == {}:
				result[i]['intent'] = 'QUERY'
			else:
				result[i]['intent'] = 'NUMBER_QUERY'

		elif pred['domain'] == 'music':
			# if re.search('有什么', text) or re.search('搜索', text) or re.search('找一首', text) or re.search('查一下', text):
			# 	result[i]['intent'] = 'SEARCH'
			# else:
			# 	result[i]['intent'] = 'PLAY'
			pass

		elif pred['domain'] == 'stock':
			# yesterday = ['昨天', '昨日']
			# if re.search('收盘价', text):
			# 	result[i]['intent'] = 'CLOSEPRICE_QUERY'
			# elif re.search('涨', text) and re.search('跌', text): 
			# 	result[i]['intent'] = 'RISERATE_QUERY'
			# elif pred['intent'] not in table[pred['domain']][0]:
			# 	result[i]['intent'] = 'QUERY'
			
			# for t in yesterday:
			# 	if 'yesterday' in result[i]['slots']:
			# 		break
			# 	if re.search(t, text):
			# 		span = re.search(t, text).span()
			# 		result[i]['slots']['yesterday'] = text[span[0]:span[1]]
			pass

		elif pred['domain'] == 'match':
			result[i]['intent'] = 'QUERY'

		slots = OrderedDict()
		for k, v in pred['slots'].items():
			if k in table[pred['domain']][1]:
				slots[k] = v

		result[i]['slots'] = slots

	return result

if __name__ == '__main__':
	result = json.load(open('result/test_result.json', encoding = 'utf8'), object_pairs_hook = OrderedDict)
	result = process(result, 'dataSet/dic')
	json.dump(result, open('result/test_result.json', 'w'), ensure_ascii = False, indent = 2)
