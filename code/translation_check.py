import json
from reldi.lemmatiser import Lemmatiser

squad2_ENG_train = open('./data/squad2_SLO_train.json', "r", encoding='UTF-8')
data_string = squad2_ENG_train.read()
squad2_ENG_train.close()

data = json.loads(data_string)

countAll = 0
countIncorrect = 0

lemmy = Lemmatiser('si')
lemmy.authorize('AjdaM', '76613226')

for q in data['data']:
    if len(q['answers']) != 0:
        for answer in q['answers']:
            brake = 0
            for word in answer['text']:
                if (word not in q['context']) and (brake == 0):
                    countIncorrect = countIncorrect + 1
                    brake = 1
            if (brake == 1):
                print(answer['text'])
                #newWord = lemmy.lemmatise(answer['text'])
            countAll = countAll + 1

print(countIncorrect)
print(countAll)