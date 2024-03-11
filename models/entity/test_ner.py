from keras.models import Model, load_model
from keras import preprocessing
import numpy as np
from utils.preprocess import Preprocess

p = Preprocess(word2index_dic='../../train_tools/dict/chatbot_dict.bin',
               userdic='../../utils/user_dic.tsv')


new_sentence = '안성시 대기 어때'
pos = p.pos(new_sentence)
keywords = p.get_keywords(pos, without_tag=True)
new_seq = p.get_wordix_sequence(keywords)

max_len = 40
new_padded_seqs = preprocessing.sequence.pad_sequences([new_seq], padding="post", value=0, maxlen=max_len)
print("새로운 유형의 시퀀스 : ", new_seq)
print("새로운 유형의 시퀀스 : ", new_padded_seqs)

# NER 예측
model = load_model('ner_model.h5')
p = model.predict(np.array([new_padded_seqs[0]]))
p = np.argmax(p, axis=-1)  # 예측된 NER 인덱스 값 추출

print(p)


print("{:10} {:5}".format("단어", "예측된 NER"))
print("-" * 50)

index = {1: 'O', 2: 'S-LOCATION', 3: 'S-PLACE', 4: 'S-RESTAURANT', 5: 'S-DATE', 6: 'B-LOCATION', 7: 'E-LOCATION', 8: 'B-RESTAURANT', 9: 'E-RESTAURANT', 10: 'B-DATE', 11: 'E-DATE', 12: 'B-PLACE', 13: 'E-PLACE', 14: 'I-RESTAURANT', 15: 'I-DATE', 0: 'PAD'}


for w, pred in zip(keywords, p[0]):
    print("{:10} {:5}".format(w, index[pred]))



# 새로운 유형의 시퀀스 :  [39, 214, 117, 194, 404, 3, 2, 9]
# 새로운 유형의 시퀀스 :  [[ 39 214 117 194 404   3   2   9   0   0   0   0   0   0   0   0   0   0
#     0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#     0   0   0   0]]


