import pandas as pd
import numpy as np

# 파일 읽기


df = pd.read_csv('/Users/seunghyunsung/PycharmProjects/chatbot/models/entity/entity_data.csv')

#label 값확인
labels = df['label'].unique()
list = labels.tolist()
print(list)
