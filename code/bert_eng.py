from datasets import load_dataset
import tensorflow as tf

from transformers import AutoTokenizer
from transformers import DefaultDataCollator, Adafactor
from transformers import create_optimizer
from transformers import BertConfig, TFBertForQuestionAnswering, TFAutoModelForQuestionAnswering
from transformers import get_constant_schedule_with_warmup
from tensorflow import keras

squad_eng = load_dataset('json', data_files={'train': './code/data/cleaned_ENG_train.json', 'validation': './code/data/cleaned_ENG_validation.json'}, field='data')
#squad_eng = load_dataset('squad_v2')
tokenizer = AutoTokenizer.from_pretrained("roberta-base")


def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]
    inputs = tokenizer(
        questions,
        #examples["context"],
        contexts,
        max_length=384,
        stride=128,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    context = examples["context"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        #print(answer['text'][0])
        ctx = context[i]
        start_char = len(ctx) + 1
        end_char = len(ctx) + 1     #comment these 2 cause no answerable za proper squad v2
        #print(answer["text"])
        if len(answer[0]["text"]) > 0:
            start_char = ctx.find(answer[0]['text']) if ctx.find(answer[0]['text']) > -1 else len(ctx) + 1         #ns dataset ma mjckn drgac za klic
            end_char = start_char + len(answer[0]['text']) if ctx.find(answer[0]['text']) > -1 else len(ctx) + 1
            #start_char = ctx.find(answer['text'][0]) if ctx.find(answer['text'][0]) > -1 else len(ctx) + 1
            #end_char = start_char + len(answer['text'][0]) if ctx.find(answer['text'][0]) > -1 else len(ctx) + 1
        sequence_ids = inputs.sequence_ids(i)
        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


tokenized_squad = squad_eng.map(preprocess_function, batched=True, remove_columns=squad_eng["train"].column_names)
data_collator = DefaultDataCollator(return_tensors="tf")

batch_size = 8
# define training hyperparameters for training set
tf_train_set = tokenized_squad["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "start_positions", "end_positions"],
    dummy_labels=True,
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
    drop_remainder=True,
)
#tf_train_set = tokenized_squad["train"].with_format("numpy")[:]
# define training hyperparameters for validation set
tf_validation_set = tokenized_squad["validation"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "start_positions", "end_positions"],
    dummy_labels=True,
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
    drop_remainder=True,
)
#tf_validation_set =tokenized_squad["validation"].with_format("numpy")[:]

train_data_size=len(tokenized_squad["train"])
num_epochs = 1
num_warmup_steps=int(num_epochs * train_data_size * 0.1 / batch_size)
total_train_steps = (train_data_size // batch_size) * num_epochs
steps_per_epoch = int(train_data_size / batch_size)
num_train_steps = steps_per_epoch * num_epochs
optimizer, schedule = create_optimizer(
    init_lr=5e-5,
    num_warmup_steps=0,#num_warmup_steps,
    num_train_steps=num_train_steps,
    #min_lr_ratio=0.99,
)


# get pretrained model
config = BertConfig.from_pretrained("roberta-base")
#model_eng = TFBertForQuestionAnswering.from_config(config)
model_eng = TFAutoModelForQuestionAnswering.from_config(config)

model_eng.compile(optimizer=optimizer)
# fine-tune model
model_eng.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=num_epochs, batch_size=batch_size)

# save model
model_eng.save_pretrained('./code/modelShuffleOn/')


