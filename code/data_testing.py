import json
import classla
from xml.etree import cElementTree as et

nlp = classla.Pipeline('sl', processors='tokenize, pos, lemma')

# get synonym lib
synonymLib = {}
xmlstr = open('./data/synonyms.xml', "r", encoding='UTF-8').read()
root = et.fromstring(xmlstr)
entry_root = root.findall('entry')
for child in entry_root:
    headword = child.find('headword').text
    core = child.find('groups_core')
    if core is None:
        core = child.find('groups_near')
    group = core.find('group')
    synonyms = ''
    for candidate in group:
        word = candidate.find('s').text
        synonyms = synonyms + word + ' '
    synonymLib[headword] = synonyms

squad2_SLO_train = open('./data/squad2_SLO_train.json', "r", encoding='UTF-8')
lemmaContextJson = open('./data/lemmaContext.json', "r", encoding='UTF-8')
data_string = squad2_SLO_train.read()
context_string = lemmaContextJson.read()
squad2_SLO_train.close()
lemmaContextJson.close()

data = json.loads(data_string)
lemmaContext = json.loads(context_string)

out_data = {'data': []}
final_data = {'data': []}
contextCount = 0

countOriginalAnswers = 0
countLemmadAnswers = 0
countSynonymAnswers = 0
countRemovedAnswers = 0
countRemovedQuestions = 0

# remove unanswerable
for row in data['data']:
    if len(row['answers']) != 0:
        out_data['data'].append(row)
    else:
        countRemovedAnswers = countRemovedAnswers + 1

# fix answers
for row in out_data['data']:
    if row['answers'][0]['text'] not in row['context']:
        list_context = row['context'].split()
        # get context
        if lemmaContext[str(contextCount)]['original'] != row['context']:
            contextCount = contextCount + 1
            if lemmaContext[str(contextCount)]['original'] != row['context']:
                contextCount = 0
                while lemmaContext[str(contextCount)]['original'] != row['context']:
                    contextCount = contextCount + 1
        lem_context = lemmaContext[str(contextCount)]['lemma'].split()

        # get question
        question = row['question']
        words = question.split()
        yes = 0
        for word in words:
            if word not in row['context']:
                wordlem = nlp(word).to_conll().split()[15]
                if wordlem in lem_context:
                    yes = 1
            else:
                yes = 1
        if yes == 0:
            row['answers'] = []
            countRemovedQuestions = countRemovedQuestions + 1

        # get answers
        answers = row['answers']
        for answer in answers:
            # get individual words in an answer
            words = answer['text'].split()
            yes = 0
            new_ans = []
            usedSymbol = False
            for word in words:
                wordlem = nlp(word).to_conll().split()[15]
                if wordlem in lem_context:
                    yes += 1
                    idx = lem_context.index(wordlem)
                    new_ans.append(list_context[idx])
                elif wordlem in synonymLib.keys():
                    synonyms = synonymLib[wordlem].split()
                    for synonym in synonyms:
                        if synonym in lem_context:
                            yes += 1
                            idx = lem_context.index(synonym)
                            new_ans.append(list_context[idx])
                            synonyms = []
                            usedSymbol = True
            if yes == len(words):  # if all the lemmas in an answer are also in context
                separator = ' '
                countLemmadAnswers = countLemmadAnswers + 1
                row['answers'][0]['text'] = separator.join(new_ans)
                if usedSymbol:
                    countSynonymAnswers = countSynonymAnswers + 1
    else:
        countOriginalAnswers = countOriginalAnswers + 1

# remove unanswerable again
for row in out_data['data']:
    if len(row['answers']) != 0:
        final_data['data'].append(row)
    else:
        countRemovedAnswers = countRemovedAnswers + 1
cut = round(len(final_data)*9/10)

print('število vseh končnih vprašanj/odgovorov: ' + str(len(final_data['data'])))
print('število vseh nespremenjenih odgovorov: ' + str(countOriginalAnswers))
print('število vseh odstranjenih odgovorov (neodgovorljivi): ' + str(countRemovedAnswers))
print('število vseh odstranjenih vprašanj (neodgovorljivi): ' + str(countRemovedQuestions))
print('število vseh popravljenih vprašanj z lemanizacijo: ' + str(countLemmadAnswers))
print('število vseh popravljenih vprašanj s sinonimi: ' + str(countSynonymAnswers))

data_train = {'data': final_data['data'][:cut]}
data_test = {'data': final_data['data'][cut:len(final_data)]}

out_train = open('./data/cleaned_SLO_train.json', 'w', encoding='UTF-8')
json.dump(data_train, out_train, ensure_ascii=False)
out_train.close()

out_test = open('./data/cleaned_SLO_test.json', 'w', encoding='UTF-8')
json.dump(data_test, out_test, ensure_ascii=False)
out_test.close()