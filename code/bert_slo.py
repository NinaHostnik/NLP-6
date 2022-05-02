from datasets import load_dataset
import tensorflow as tf

from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import create_optimizer
from transformers import BertConfig, TFBertForQuestionAnswering

squad_slo = load_dataset('json', data_files={'train': './data/squad2_SLO_train.json', 'validation': './data/squad2_SLO_validation.json'}, field='data')

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")


def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
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
        ctx = context[i]
        start_char = len(ctx) + 1
        end_char = len(ctx) + 1
        if len(answer) > 0:
            start_char = ctx.find(answer[0]['text']) if ctx.find(answer[0]['text']) > -1 else len(ctx) + 1
            end_char = start_char + len(answer[0]['text']) if ctx.find(answer[0]['text']) > -1 else len(ctx) + 1
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


tokenized_squad = squad_slo.map(preprocess_function, batched=True, remove_columns=squad_slo["train"].column_names)
data_collator = DefaultDataCollator(return_tensors="tf")

# define training hyperparameters for training set
tf_train_set = tokenized_squad["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "start_positions", "end_positions"],
    dummy_labels=True,
    shuffle=True,
    batch_size=8,
    collate_fn=data_collator,
)

# define training hyperparameters for validation set
tf_validation_set = tokenized_squad["validation"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "start_positions", "end_positions"],
    dummy_labels=True,
    shuffle=False,
    batch_size=8,
    collate_fn=data_collator,
)

batch_size = 8
num_epochs = 1
total_train_steps = (len(tokenized_squad["train"]) // batch_size) * num_epochs
optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=0,
    num_train_steps=total_train_steps,
)

# get pretrained model
config = BertConfig.from_pretrained("bert-base-multilingual-cased")
model_slo = TFBertForQuestionAnswering.from_config(config)

model_slo.compile(optimizer=optimizer)

# fine-tune model
model_slo.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=1)

# save model
model_slo.save_pretrained('./code/model')
