# -*- coding: utf-8 -*-
import time
import os
from model.lfm import LFM, Corpus


def run(input_user_id=1, input_top_n=10):
    assert os.path.exists('data/ratings.csv'), \
        'File not exists in path, run preprocess.py before this.'
    print('Start..')
    print('用户ID：%s' % input_user_id)

    start = time.time()
    if not os.path.exists('data/lfm_items.dict'):
        Corpus.pre_process()
    if not os.path.exists('data/lfm.model'):
        LFM().train()
    movies = LFM().predict(user_id=input_user_id, top_n=input_top_n)
    for movie in movies:
        print(movie)
    print('用时: %f' % (time.time() - start))
