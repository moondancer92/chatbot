import pandas as pd

#불러오기
df = pd.read_csv('/Users/seunghyunsung/PycharmProjects/chatbot/models/entity/entity_data.csv')


condition = (df.question.str.contains('공기')) | (df.question.str.contains('대기')) | (df.question.str.contains('먼지'))



# 원본에서 레이블이 0/1인 것만 복사.


df2 = df.loc[condition].copy()



# 원본에서 먼지/공기를 대기질로 변경
df2['question'] = df2.question.str.replace('먼지', '대기질').replace('공기', '대기질')
df2['question'] = df2.question.str.replace('미세', '')



result = pd.concat([df, df2])

print(result)

result.to_csv(r"/Users/seunghyunsung/PycharmProjects/chatbot/models/entity/added_data.csv", mode="a", header=True, index=False)


