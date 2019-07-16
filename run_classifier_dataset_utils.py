# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import json
import collections
import logging
import os
import sys
import numpy as np

from rule import process

logger = logging.getLogger(__name__)

#run 'sample.json:0x9ba8a2' 'config.json:0x190174' 'vocab.txt:0xfec2ea' 'optimization.py:0xddfbb4' 'tokenization.py:0xd6a857' 'modeling.py:0x513a98' 'pytorch_model.bin:0x796f6e' 'pytorch_model2.bin:0xb7667b' 'pytorch_model3.bin:0x99124c' 'run_classifier_dataset_utils.py:0x09918f' 'run_classifier.py:0x45023b' 'rule.py:0x4e1774' 'city.txt:0x50dc61' 'railway_station.txt:0x2acf89' 'dishName.txt:0xa346b6' 'province.txt:0x2cbaf9' 'python3 run_classifier.py --test_data sample.json --result_file pred.json' --request-docker-image pytorch/pytorch:latest --request-memory 6g


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, domain=None, intent=None, slots=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.domain = domain
    self.intent = intent
    self.slots = slots


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               domain_id,
               intent_id,
               slots_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.domain_id = domain_id
    self.intent_id = intent_id
    self.slots_id = slots_id
    self.is_real_example = is_real_example


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

  @classmethod
  def _read_json(cls, input_file, quotechar=None):
    """Reads pandas csv file."""
    data_json = json.load(open(input_file, encoding='utf8'), object_pairs_hook=collections.OrderedDict)

    return data_json


class NLUProcessor(DataProcessor):
  """Processor for nlu data set."""
  def __init__(self):
    pass

  def get_train_examples(self, train_data_path):
    """See base class."""
    set_type = "train"
    data = self._read_json(train_data_path)
    examples = []
    for (i, line) in enumerate(data):
      guid = "%s-%s" % (set_type, i)
      text_a, domain, intent, slots = line['text'], line['domain'], line['intent'], line['slots']
      examples.append(InputExample(guid = guid, text_a = text_a, domain = domain, intent = intent, slots = slots))

    return examples

  def get_dev_examples(self, eval_data_path):
    """See base class."""
    pass

  def get_test_examples(self, test_data_path):
    """See base class."""
    set_type = "test"
    data = self._read_json(test_data_path)
    examples = []
    for (i, line) in enumerate(data):
      guid = "%s-%s" % (set_type, i)
      text_a = line['text']
      examples.append(InputExample(guid = guid, text_a = text_a))

    return examples

  def get_labels(self):
    """See base class."""
    domain = ['app', 'bus', 'cinemas', 'contacts', 'cookbook', 'email', 'epg', 'flight', 'health', 'joke', 'lottery', 'map', 'match', 'message', 'music', 'news', 'novel', 'poetry', 'radio', 'riddle', 'stock', 'story', 'telephone', 'train', 'translation', 'tvchannel', 'video', 'weather', 'website']
    intent = ['CLOSEPRICE_QUERY', 'CREATE', 'DATE_QUERY', 'DEFAULT', 'DIAL', 'DOWNLOAD', 'FORWARD', 'LAUNCH', 'LOOK_BACK', 'NUMBER_QUERY', 'OPEN', 'PLAY', 'POSITION', 'QUERY', 'REPLAY_ALL', 'REPLY', 'RISERATE_QUERY', 'ROUTE', 'SEARCH', 'SEND', 'SENDCONTACTS', 'TRANSLATION', 'VIEW']
    slotsOri = ['Dest', 'Src', 'absIssue', 'area', 'artist', 'artistRole', 'author', 'awayName', 'category', 'code', 'content', 'datetime_date', 'datetime_time', 'decade', 'dishName', 'dynasty', 'endLoc_area', 'endLoc_city', 'endLoc_poi', 'endLoc_province', 'episode', 'film', 'headNum', 'homeName', 'ingredient', 'keyword', 'location_area', 'location_city', 'location_country', 'location_poi', 'location_province', 'media', 'name', 'payment', 'popularity', 'queryField', 'questionWord', 'receiver', 'relIssue', 'resolution', 'scoreDescr', 'season', 'song', 'startDate_date', 'startDate_time', 'startLoc_area', 'startLoc_city', 'startLoc_poi', 'startLoc_province', 'subfocus', 'tag', 'target', 'teleOperator', 'theatre', 'timeDescr', 'tvchannel', 'type', 'utensil', 'yesterday']
    
    slots = []
    slots.append("O")
    for slot in slotsOri:
      slots.append("B-"+slot)
      slots.append("I-"+slot)

    return {'domain':list(domain), 'intent':list(intent), 'slots':slots}


def slots_convert(text, slots):
  """Convert slots to B-I-O form"""
  tokens = ["O"] * len(text)
  if slots:
    for slot, value in slots.items():
      index = text.find(value)
      for i in range(len(value)):
        if i == 0:
          slot_token = "B-"+slot
        else:
          slot_token = "I-"+slot
        tokens[index+i] = slot_token

  return tokens


def convert_examples_to_features(examples, domain_map, intent_map, slots_map, 
                           max_seq_length, tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  features = []
  for (ex_index, example) in enumerate(examples):
    ori_slots = slots_convert(example.text_a, example.slots)
    # tokens_a = tokenizer.tokenize(example.text_a)
    tokens_a = []
    tokens_slots = []
    for i, word in enumerate(example.text_a):
      token = tokenizer.tokenize(word)
      tokens_a.extend(token)
      if len(token) > 0:
        tokens_slots.append(ori_slots[i])
    if not len(tokens_a) == len(tokens_slots):
      logger.info("********** Take Care! ***********")
      print(tokens_a)
      print(tokens_slots)
    assert len(tokens_a) == len(tokens_slots)

    # tokens_b = None
    # if example.text_b:
    #   tokens_b = tokenizer.tokenize(example.text_b)

    # if tokens_b:
    #   # Modifies `tokens_a` and `tokens_b` in place so that the total
    #   # length is less than the specified length.
    #   # Account for [CLS], [SEP], [SEP] with "- 3"
    #   _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    # else:
      # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]
      tokens_slots = tokens_slots[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    slots_id = []
    segment_ids = []
    tokens.append("[CLS]")
    slots_id.append(slots_map["O"])
    segment_ids.append(0)
    for token, slots in zip(tokens_a, tokens_slots):
      tokens.append(token)
      slots_id.append(slots_map[slots])
      segment_ids.append(0)
    tokens.append("[SEP]")
    slots_id.append(slots_map["O"])
    segment_ids.append(0)

    # if tokens_b:
    #   for token in tokens_b:
    #     tokens.append(token)
    #     segment_ids.append(1)
    #   tokens.append("[SEP]")
    #   segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)
      slots_id.append(slots_map["O"])

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(slots_id) == max_seq_length

    domain_id, intent_id = 0, 0
    if example.domain:
      domain_id = domain_map[example.domain]
    if example.intent:
      intent_id = intent_map[example.intent]

    # if ex_index < 1:
    #   tf.logging.info("*** Example ***")
    #   tf.logging.info("guid: %s" % (example.guid))
    #   tf.logging.info("tokens: %s" % " ".join(
    #       [tokenization.printable_text(x) for x in tokens]))
    #   tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    #   tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    #   tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    #   tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    features.append(InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        domain_id=domain_id,
        intent_id=intent_id,
        slots_id=slots_id,
        is_real_example=True))

  return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def get_slots(slots_id, slots_map, text):
  slots = {}
  tokens_slots = []
  for i in range(1, min(len(text)+1, len(slots_id))):
    tokens_slots.append(slots_map[slots_id[i]])

  i = 0
  while i < len(text):
    if not tokens_slots[i] == "O" and tokens_slots[i][:2] == "B-":
      slot = tokens_slots[i][2:]
      value = [text[i]]
      i += 1
      while i < len(text) and not tokens_slots[i] == "O" and not tokens_slots[i][:2] == "B-":
        value.append(text[i])
        i += 1
      slots[slot] = "".join(value)
      i -= 1
    i += 1

  return slots


def write_result(output_predict_file, dic_dir, result, predict_examples, domain_map, intent_map, slots_map):
  """ Write result to json file"""
  result_json = []

  domain_map = {v:k for k, v in domain_map.items()}
  intent_map = {v:k for k, v in intent_map.items()}
  slots_map = {v:k for k, v in slots_map.items()}

  for i, (pred, example) in enumerate(zip(result, predict_examples)):
    text = example.text_a

    domain_id = np.argmax(pred["domain"])
    intent_id = np.argmax(pred["intent"])
    slots_id = np.argmax(pred["slots"], axis = -1)
    slots = get_slots(slots_id, slots_map, text)

    d = collections.OrderedDict()
    d["text"] = text
    d["domain"] = domain_map[domain_id]
    d["intent"] = intent_map[intent_id]
    d["slots"] = slots
    result_json.append(d)
  
  # json.dump(result_json, open(output_predict_file, 'w', encoding = 'utf-8'), ensure_ascii = False, indent = 2)
  json.dump(process(result_json, dic_dir), open(output_predict_file, 'w', encoding = 'utf-8'), ensure_ascii = False, indent = 2)


processors = {
  "nlu": NLUProcessor
}
