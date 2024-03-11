import pandas as pd
import numpy as np

# 파일 읽기


df = pd.read_csv('/Users/seunghyunsung/PycharmProjects/chatbot/models/intent/added_data.csv')

# dust tag 에서 '공기' 포함한 태그 'dust' 에서 먼지 포함한 문장은 제외
condition = (df.question.str.contains('공기')) | (df.question.str.contains('대기')) & (df.label.str.contains('dust')) & ~(df.question.str.contains('먼지'))

df.loc[condition,'label'] = 'air'



#label 값확인
labels = df['label'].unique()
list = labels.tolist()
print(list)

# index 만들기
indexList = []

for label in labels:
    index = str(list.index(label))
    indexList.append(index)

for index in indexList:
    print(type(index))

#  lable,index dict 로 값 변경
my_dict = dict(zip(list,indexList))
df.replace(my_dict, inplace=True)

#print(df)

#csv 파일로 저장
df.to_csv(r"/Users/seunghyunsung/PycharmProjects/chatbot/models/intent/data.csv", mode="a", header=True, index=False)
