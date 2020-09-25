from hparams import *

import random
import pandas as pd
import os

# 데이터 불러오기
original_data_df = pd.read_csv('./data/original_data.csv', encoding = 'ANSI')
original_data_df.shape

# 결측치 제거
original_data_df = original_data_df.dropna(axis = 0)

# 데이터 저장
original_data_df['document'].to_csv('./data/original_data_txt.txt', index = False, header = None, sep = "\t")

# spacing 진행 후 데이터 불러오기
with open('./data/fixed_original_data_txt.txt', 'r') as text_file:
    sentence_list = text_file.readlines()

# 데이터 덮어씌우기
original_data_df['document'] = pd.DataFrame({'document': sentence_list})
original_data_df = original_data_df.dropna(axis = 0)

# id 행 삭제
original_data_df = original_data_df.drop('id', axis = 1)

# 정규 표현식 사용
original_data_df['document'] = original_data_df['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")


# 데이터 저장
original_data_df[30001:].to_csv('./data/train.txt', index = False)
original_data_df[10001:30000].to_csv('./data/validation.txt', index = False)
original_data_df[:10000].to_csv('./data/test.txt', index = False)