import nltk
from nltk.corpus import stopwords
from transformers import pipeline
import json
import editdistance
import string

def izracunajF1(prediction, realAnswer, translator, stop_words):
    #odstranimo locila obema stringoma
    predictionBrezLocil=prediction[0].translate(translator)
    realAnswerBrezLocil=realAnswer[0].translate(translator)
    #tokeniziramo in odstranimo stopwords
    predictionTokeni=nltk.word_tokenize(predictionBrezLocil) #Pridobi tokene za napoved in napoved modela
    normaliziraniPredictionTokeni=[w for w in predictionTokeni if not w.lower() in stop_words] #te tokene normalizira, kar pomeni, da jim odstrani ločila in stop words za vsako besedo
    realAnswerTokeni=nltk.word_tokenize(realAnswerBrezLocil)
    normaliziraniRealAnswerTokeni=[w for w in realAnswerTokeni if not w.lower() in stop_words]
    #nastavitev spremenljivk in preverjanje vrednosti
    truePositive=0
    falsePositive=0
    falseNegative=0
    for token in normaliziraniPredictionTokeni:
        if str(token) in normaliziraniRealAnswerTokeni:
            truePositive=truePositive+1
    for token in normaliziraniPredictionTokeni:
        if str(token) not in normaliziraniRealAnswerTokeni:
            falsePositive+=1
    for token in normaliziraniRealAnswerTokeni:
        if str(token) not in normaliziraniPredictionTokeni:
            falseNegative+=1   
    if truePositive==0:
        return 0, 0, falsePositive, falseNegative
    #izracun precision recall ter nato f1 score
    precision=truePositive/(truePositive+falsePositive)             #izracun precision (pravi najdeni/(pravi najdeni+narobni najdeni))
    recall=truePositive/(truePositive+falseNegative)                #izracun recall (pravi najdeni/(pravi najdeni+pravi nenajdeni))
    return 2*(precision*recall)/(precision+recall), truePositive, falsePositive, falseNegative                      #izracun F1 score+vrne true positive, false positive in false negative

def pridobiPrediction(question, paragraph, qamodel):
    result=qamodel(question = question, context = paragraph)
    return result['answer']
#import podatkov
testImport = open('D:/Faks/ONJ/NLP-6/code/data/cleaned_ENG_test.json', "r", encoding='UTF-8')
data_string = testImport.read()
testImport.close()
testData = json.loads(data_string)
stop_words=set(stopwords.words('english')).union(set(stopwords.words('slovene')))
#inicializacija spremenljivk
i=0
stPravilnih=0
stNeprevilnih=0
stPredolgih=0
stPerfectMatch=0
f1Vsi=[]
accuracyVsi=[]
levenstheinSimilarities=[]
tabelaTruePositive=[]
tabelaFalsePositive=[]
tabelaFalseNegative=[]
#inicializacija pipeline za pridobivanje odgovorov
#qamodel=pipeline('question-answering', model="D:/Faks/ONJ/NLP-6/code/Primerjava modelov/ROBERTA", tokenizer=("roberta-base"))
qamodel=pipeline('question-answering', model="D:/Faks/ONJ/NLP-6/code/Modeli/ANG", tokenizer=("roberta-base"))
translator=str.maketrans('', '', string.punctuation)
#zanka za preverjanje vseh vprašanj iz test podatkov
for data in testData['data']:   #iz test data vzame primer po primer
    prediction=pridobiPrediction(data['question'], data['context'], qamodel)    #v pipeline vstavi vprasanje in kontekst
    realAnswer=data['answers']                                                  #pridobi pravi odgovor
    f1, truePositive, falsePositive, falseNegative=izracunajF1([prediction], [realAnswer[0]['text']], translator, stop_words)   #izracuna f1 score za primer kot tudi vrne druge podatke, ki bodo uporabljeni za izracun micro povprecja F1 score
    tabelaTruePositive.append(truePositive)                                                             #doda true positive primere v tabelo
    tabelaFalsePositive.append(falsePositive)                                                           #doda false positive primere v tabelo
    tabelaFalseNegative.append(falseNegative)                                                           #doda false negative primere v tabelo
    editDistance=editdistance.eval(realAnswer[0]['text'], prediction)                                   #izracuna edit distance
    levenstheinSimilarities.append(1-(editDistance/max([len(realAnswer[0]['text']), len(prediction)])))         #iz edit distance pridobimo 
    if editDistance==0:                                                                                         #ce se odgovora popolnoma ujemata(f1 score==1) se doda stevec popolnoma enakih odgovorov
        stPerfectMatch+=1
    if str(realAnswer) in str(prediction):
        stPredolgih=stPredolgih+1
    if str(prediction) in str(realAnswer):
        stPravilnih=stPravilnih+1
    else:
        stNeprevilnih=stNeprevilnih+1
    f1Vsi.append(f1)                                                                                    #dodamo F1 score v tabelo, iz katere se bo nato izracunal macro f1 povprecje
    i=i+1
    print(i)
    #if i==100:
        #break
sumTruePositive=sum(tabelaTruePositive)                                                                 #sestevek true positive
sumFalsePositive=sum(tabelaFalsePositive)                                                               #sestevek false positive
sumFalseNegative=sum(tabelaFalseNegative)                                                               #sestevek false negative
#izracun F1 micro
skupnPrecision=sumTruePositive/(sumTruePositive+sumFalsePositive)
skupnRecall=sumTruePositive/(sumTruePositive+sumFalseNegative)
F1micro=2*(skupnPrecision*skupnRecall)/(skupnPrecision+skupnRecall)
#izpis
print(str(stPravilnih)+" "+str(stNeprevilnih)+" "+str(stPredolgih))
print("F1 macro: " + str(sum(f1Vsi)/len(f1Vsi)))                                                                            #izracun F1 macro
print("F1 micro: "+str(F1micro))
print("Average levenstein similaritiy: "+str(sum(levenstheinSimilarities)/len(levenstheinSimilarities)))
print("Perfect match: "+str(stPerfectMatch))