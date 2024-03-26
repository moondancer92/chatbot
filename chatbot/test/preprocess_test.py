from utils.preprocess import Preprocess

sent = "알려줄래요"

p = Preprocess(userdic='../utils/user_dic.tsv')

pos = p.pos(sent)

ret = p.get_keywords(pos, without_tag=False)
print(ret)

ret = p.get_keywords(pos, without_tag=True)
print(ret)