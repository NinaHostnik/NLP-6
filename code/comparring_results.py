import nltk
from transformers import pipeline
import tensorflow as tf
import json
import editdistance

def izracunajF1(prediction, realAnswer):
    predictionTokeni=nltk.word_tokenize(prediction[0]) #Pridobi tokene za napoved in napoved modela
    normaliziraniPredictionTokeni=[word for word in predictionTokeni if word.isalpha()] #te tokene normalizira, kar pomeni, da jim odstrani ločila in stop words za vsako besedo
    realAnswerTokeni=nltk.word_tokenize(realAnswer[0])
    normaliziraniRealAnswerTokeni=[word for word in realAnswerTokeni if word.isalpha()]
    #print(predictionTokeni)
    #print(realAnswerTokeni)
    truePositive=0
    falsePositive=0
    falseNegative=0
    for token in predictionTokeni:
        if str(token) in realAnswerTokeni:
            truePositive=truePositive+1
    if truePositive==0:
        return 0
    for token in predictionTokeni:
        if str(token) not in realAnswerTokeni:
            falsePositive+=1
    for token in realAnswerTokeni:
        if str(token) not in predictionTokeni:
            falseNegative+=1   
    precision=truePositive/(truePositive+falsePositive)             #izracun precision (pravi najdeni/(pravi najdeni+narobni najdeni))
    recall=truePositive/(truePositive+falseNegative)                #izracun recall (pravi najdeni/(pravi najdeni+pravi nenajdeni))
    return 2*(precision*recall)/(precision+recall)                      #izracun F1 score

def pridobiPrediction(question, paragraph, qamodel):
    result=qamodel(question = question, context = paragraph)
    return result['answer']
question="In what country is Normandy located?"
context="The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ('Norman' comes from 'Norseman') raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries."
groundTruths=["about 10,000 years ago"]
prediction=[]
prediction.append("10,000")#pridobiPrediction(question, context))
f1=izracunajF1(prediction, groundTruths)
#print(f1)
#answer=["three dimensional index"]
#print("Question: "+question+"\n\nContext: "+context+"\n\nGround truths: "+groundTruths+"\n\nprediction: "+prediction)
#import podatkov
testImport = open('D:/Faks/ONJ/NLP-6/code/data/cleaned_ENG_test.json', "r", encoding='UTF-8')
data_string = testImport.read()
testImport.close()
testData = json.loads(data_string)
stPravilnih=0
stNeprevilnih=0
stPredolgih=0
qamodel=pipeline('question-answering', model="D:/Faks/ONJ/NLP-6/code/modelShuffleOn", tokenizer=("roberta-base"))
#qamodel2=pipeline('question-answering', model="keras-io/transformers-qa")
i=0
f1Vsi=[]
accuracyVsi=[]
editDistances=[]
for data in testData['data']:
    prediction=pridobiPrediction(data['question'], data['context'], qamodel)
    realAnswer=data['answers']
    print("Prediction:")
    print(prediction)
    print(len(prediction))
    print("Ground Truth")
    print(realAnswer[0]['text'])
    print(len(realAnswer[0]['text']))
    f1=izracunajF1([prediction], [realAnswer[0]['text']])
    editDistance=editdistance.eval(realAnswer[0]['text'], prediction)
    editDistances.append(editDistance)
    print("F1 score")
    print(f1)
    print("edit distance")
    print(editDistance)
    #print(f1)
    if str(realAnswer) in str(prediction):
        stPredolgih=stPredolgih+1
    if str(prediction) in str(realAnswer):
        stPravilnih=stPravilnih+1
    else:
        stNeprevilnih=stNeprevilnih+1
    f1Vsi.append(f1)
    #print(i)
    i=i+1
    if i==100:
        break
print(str(stPravilnih)+" "+str(stNeprevilnih)+" "+str(stPredolgih))
print(sum(f1Vsi)/len(f1Vsi))
print(sum(editDistances)/len(editDistances))
print(editDistances)

#print(accuracyVsi)