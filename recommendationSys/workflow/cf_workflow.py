# -*- coding: utf-8 -*-
import time
import os
from model.cf import UserCf


def run(input_user_id=100):
    assert os.path.exists('data/ratings.csv'), \
        'File not exists in path, run preprocess.py before this.'
    print('Start..')
    print("用户ID：", input_user_id)
    start = time.time()
    movies = UserCf().calculate(target_user_id=input_user_id)
    for movie in movies:
        print(movie)
    print('Cost time: %f' % (time.time() - start))
