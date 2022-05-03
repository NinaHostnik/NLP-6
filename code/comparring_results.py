import nltk
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, TFBertForQuestionAnswering
import tensorflow as tf

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
    f1=f1_score(normaliziraniPredictionTokeni, normaliziraniRealAnswerTokeni, average='micro')
    accuracy=accuracy_score(normaliziraniPredictionTokeni, normaliziraniRealAnswerTokeni)
    return f1, accuracy#2*(precision*recall)/(precision+recall)                      #izracun F1 score

def pridobiPrediction(question, paragraph):
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    bert_model = TFBertForQuestionAnswering.from_pretrained('./code/modelSlo')
    #bert_model = TFBertForQuestionAnswering.from_pretrained("bert-base-cased")
    question1, text = "Kakšna energija naredi žarilno žarnico?", "Žarnica z žarilno nitko, žarnica ali žarnica z žarilno nitko je električna svetloba z žično nitko, ki se segreje na visoko temperaturo, tako da skozi njo prehaja električni tok, dokler ne sveti z vidno svetlobo (incandescence). Vroč filament je zaščiten pred oksidacijo s stekleno ali kremenčevo žarnico, ki je napolnjena z inertnim plinom ali evakuirana. V halogenski žarnici se izhlapevanje žarilne nitke prepreči s kemičnim postopkom, ki ponovno nalaga kovinsko paro na žarilno nitko in podaljša njeno življenjsko dobo. Žarnica se napaja z električnim tokom s priključki ali žicami, vgrajenimi v steklo. Večina žarnic se uporablja v vtičnici, ki zagotavlja mehansko podporo in električne povezave."
    inputs = tokenizer(text, question1, return_tensors="tf")
    outputs = bert_model(inputs)
    answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
    answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])
    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    print(bert_model)
    return tokenizer.decode(predict_answer_tokens)
print(pridobiPrediction("", ""))