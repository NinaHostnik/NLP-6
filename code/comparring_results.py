import nltk
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, TFBertForQuestionAnswering, pipeline
import tensorflow as tf
import json

def izracunajF1(prediction, realAnswer):
    predictionTokeni=nltk.word_tokenize(prediction) #Pridobi tokene za napoved in napoved modela
    normaliziraniPredictionTokeni=[word for word in predictionTokeni if word.isalpha()] #te tokene normalizira, kar pomeni, da jim odstrani ločila in stop words za vsako besedo
    realAnswerTokeni=nltk.word_tokenize(realAnswer)
    normaliziraniRealAnswerTokeni=[word for word in realAnswerTokeni if word.isalpha()]
    truePositive=0
    for token in predictionTokeni:
        if token in realAnswerTokeni:
            truePositive=truePositive+1
    precision = truePositive / len(normaliziraniPredictionTokeni)  #izracun precision (pravi najdeni/vsi najdeni)
    recall = truePositive / len(normaliziraniRealAnswerTokeni)     #izracun recall (pravi najdeni/vsi pravilni odgovori)
    f1=0
    #f1=f1_score(normaliziraniPredictionTokeni, normaliziraniRealAnswerTokeni, average='micro')
    #accuracy=accuracy_score(normaliziraniPredictionTokeni, normaliziraniRealAnswerTokeni)
    return 2*(precision*recall)/(precision+recall) #f1, accuracy#2*(precision*recall)/(precision+recall)                      #izracun F1 score

def pridobiPrediction(question, paragraph, qamodel):
    result=qamodel(question = question, context = paragraph)
    return result['answer']
#question="In what country is Normandy located?"
#context="The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ('Norman' comes from 'Norseman') raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries."
#groundTruths="France"
#prediction=pridobiPrediction(question, context)
#answer=["three dimensional index"]
#print("Question: "+question+"\n\nContext: "+context+"\n\nGround truths: "+groundTruths+"\n\nprediction: "+prediction)
#import podatkov
testImport = open('./code/data/cleaned_ENG_test.json', "r", encoding='UTF-8')
data_string = testImport.read()
testImport.close()
testData = json.loads(data_string)
stPravilnih=0
stNeprevilnih=0
qamodel=pipeline('question-answering', model=('./code/model'), tokenizer=("bert-base-multilingual-cased"))
qamodel2=pipeline('question-answering', model="keras-io/transformers-qa")
i=0
f1Vsi=[]
accuracyVsi=[]
for data in testData['data']:
    prediction=pridobiPrediction(data['question'], data['context'], qamodel2)
    realAnswer=data['answers']
    #f1=izracunajF1(prediction, realAnswer[0]['text'])
    if str(prediction) in str(realAnswer):
        stPravilnih=stPravilnih+1
        print(prediction)
        print(realAnswer[0]['text'])
    else:
        stNeprevilnih=stNeprevilnih+1
    #f1Vsi.append(f1)
    #accuracyVsi.append(accuracy)
    #print(i)
    i=i+1
    if i==100:
        break
print(str(stPravilnih)+" "+str(stNeprevilnih))
print(f1Vsi)
print(accuracyVsi)