from utils.preprocess import Preprocess
from models.intent.IntentModel import IntentModel

p = Preprocess(word2index_dic='../train_tools/dict/chatbot_dict.bin',
               userdic='../utils/user_dic.tsv')

intent = IntentModel(model_name='/Users/seunghyunsung/PycharmProjects/chatbot/models/intent/intent_model.h5', proprocess=p)
query = "오늘 오사카 대기질 어때 "
predict = intent.predict_class(query)
predict_label = intent.labels[predict]

print(query)
print("의도 예측 클래스 : ", predict)
print("의도 예측 레이블 : ", predict_label)
