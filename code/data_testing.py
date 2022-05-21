import json
import classla

# squad = load_dataset('json', data_files={'train': './data/squad2_SLO_train.json', 'validation': './data/squad2_SLO_validation.json'}, field='data')
#classla.download('sl')
nlp = classla.Pipeline('sl', processors='tokenize, pos, lemma')

squad2_SLO_train = open('./data/squad2_SLO_train.json', "r", encoding='UTF-8')
data_string = squad2_SLO_train.read()
squad2_SLO_train.close()

data = json.loads(data_string)

out_data = {'data': []}

# remove unanswerable
for row in data['data']:
    if len(row['answers']) != 0:
        out_data['data'].append(row)

# fix answers
for row in out_data['data']:
    if row['answers'][0]['text'] not in row['context']:
        # lemmatize context
        list_context = row['context'].split()
        lem_context = []
        for l in list_context:
            lemma = nlp(l).to_conll().split()
            lem_context.append(lemma[15])

        # get answers
        answers = row['answers']
        for answer in answers:
            # get individual words in an answer
            words = answer['text'].split()
            yes = 0
            new_ans = []
            for word in words:
                wordlem = nlp(word).to_conll().split()[15]
                if wordlem in lem_context:
                    yes += 1
                    idx = lem_context.index(wordlem)
                    new_ans.append(list_context[idx])
            if yes == len(words):  # if all the lemmas in an answer are also in context
                separator = ' '
                row['answers'][0]['text'] = separator.join(new_ans)

#d = out_data['data'][1]
#list_context = d['context'].split()
#lemma_context = []
#for l in list_context:
#    temp = nlp(l).to_conll()
#    lemma = temp.split()
#    lemma_context.append(lemma[15])
#
#lemma = ''
#for answer in d['answers'][0]['text'].split():
#    lemma = nlp(answer).to_conll().split()
#
#if lemma[15] in lemma_context:
#    idx = lemma_context.index(lemma[15])
#    print(list_context[idx])

data_train = {'data': out_data['data'][:76463]}
data_test = {'data': out_data['data'][76463:86463]}

out_train = open('./data/cleaned_SLO_train.json', 'w', encoding='UTF-8')
json.dump(data_train, out_train, ensure_ascii=False)
out_train.close()

out_test = open('./data/cleaned_SLO_test.json', 'w', encoding='UTF-8')
json.dump(data_test, out_test, ensure_ascii=False)
out_test.close()

#out_validation = open('./data/cleaned_ENG_validation.json', 'w', encoding='UTF-8')
#json.dump(out_data, out_validation, ensure_ascii=False)
#out_validation.close()
