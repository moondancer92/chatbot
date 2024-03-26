import spacy
nlp = spacy.load("ko_core_news_sm")
doc = nlp("내일 서울 미세먼지 알려 줘.")
print([(w.text, w.pos) for w in doc])