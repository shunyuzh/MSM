import os
import csv
import random
import shutil
import jsonlines
from tqdm import tqdm

source = [
    'data/mmarco/chinese', 
    'data/mmarco/french', 
    'data/mmarco/german', 
    'data/mmarco/indonesian', 
    'data/mmarco/italian', 
    'data/mmarco/portuguese', 
    'data/mmarco/russian', 
    'data/mmarco/spanish', 
    'data/msmarco'
]

target = [
    'data/mmarco/zh', 
    'data/mmarco/fr', 
    'data/mmarco/de', 
    'data/mmarco/id', 
    'data/mmarco/it', 
    'data/mmarco/pt', 
    'data/mmarco/ru', 
    'data/mmarco/es', 
    'data/mmarco/en'
]

random.seed(42)

for src_path, tgt_path in zip(source, target):

    os.makedirs(tgt_path, exist_ok=True)
    os.makedirs(tgt_path + '/collection', exist_ok=True)

    with jsonlines.open('%s/corpus.jsonl' % src_path) as reader:
        with jsonlines.open('%s/collection/docs.jsonl' % tgt_path, mode='w') as writer:
            for obj in tqdm(reader, src_path):
                w = {
                    'id': obj['_id'],
                    'contents': '{}\n\n{}'.format(obj['title'], obj['text'])
                }
                writer.write(w)

    queries = {}
    with jsonlines.open('%s/queries.jsonl' % src_path) as reader:
        for obj in reader:
            queries[obj['_id']] = obj['text']
    
    for split in ['dev', 'train']:
        qrels, topic_id, topic = [], [], []
        with open('%s/qrels/%s.tsv' % (src_path, split)) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                if row[0] != 'query-id':
                    topic_id.append(row[0])
                    qrels.append('%s Q0 %s %s\n' % (row[0], row[1], row[2]))
        
        topic_id = list(set(topic_id))
        for _id in topic_id:
            if len(queries[_id]) > 0:
                topic.append('%s\t%s\n' % (_id, queries[_id]))
            else:
                topic.append('%s\t%s\n' % (_id, _id))
        
        print(tgt_path, split, "total", len(topic), "samples")
        
        with open('%s/qrels.%s.txt' % (tgt_path, split), 'w') as fw:
            for items in qrels:
                fw.write(items)
        
        with open('%s/topic.%s.tsv' % (tgt_path, split), 'w') as fw:
            for items in topic:
                fw.write(items)
        
        if split == 'train':
            small_items = random.sample(topic, 50000)
            with open('%s/topic.train.small.tsv' % (tgt_path), 'w') as fw:
                for items in small_items:
                    fw.write(items)

    shutil.rmtree(src_path)