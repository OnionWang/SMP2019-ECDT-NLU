#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/5/19 4:12 PM
# @Author  : zchai

# -*- coding: utf-8 -*-
import sys

import random

from extract_features import get_features



'''
    Guess a label in random as a base line
'''
def random_guess():
    data_random = {}

    data_random['domain'] = random.sample(domain_value_list, 1)[0]
    data_random['intent'] = random.sample(intent_value_list, 1)[0]
    slot = {}
    slot[random.sample(slots_key_list, 1)[0]] = random.sample(slots_value_list, 1)[0]
    data_random['slots'] = slot

    return data_random


if __name__ == '__main__':
    import json
    dev_dct = json.load(open(sys.argv[1]), encoding='utf8')

    domain_value_list, intent_value_list, slots_key_list, slots_value_list = get_features(sys.argv[1])

    rguess_dct = []
    for dev_data in dev_dct:
        text_dic = {"text": dev_data['text']}
        rguess_dct.append(dict(text_dic, **random_guess()))
    json.dump(rguess_dct, open(sys.argv[2], 'w', encoding='utf8'), ensure_ascii=False)