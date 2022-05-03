import nltk

def izracunajF1(prediction, realAnswer):
    predictionTokeni=nltk.word_tokenize(prediction) #Pridobi tokene za pravi odgovor in napoved modela
    normaliziraniPredictionTokeni=[word for word in predictionTokeni if word.isalpha()] #te tokene normalizira, kar pomeni, da jim odstrani ločila in stop words za vsako besedo
    realAnswerTokeni=nltk.word_tokenize(realAnswer)
    normaliziraniRealAnswerTokeni=[word for word in realAnswerTokeni if word.isalpha()]
    truePositive=0
    for token in predictionTokeni:
        if token in realAnswerTokeni:
            truePositive=truePositive+1
    precision = truePositive / len(normaliziraniPredictionTokeni)  #izracun precision (pravi najdeni/vsi najdeni)
    recall = truePositive / len(normaliziraniRealAnswerTokeni)     #izracun recall (pravi najdeni/vsi pravilni odgovori)
    return 2*(precision*recall)/(precision+recall)                      #izracun F1 score

def pridobiPrediction(question, paragraph):
    return