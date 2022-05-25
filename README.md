# NLP-6
Due to the size of the JSON data files and BERT model files, they cannot be added to the git repository and are accessible at the following link:
https://drive.google.com/drive/folders/1Heg5ACPz-Fncsu73Cp9tkOxsnpz8DQZY?usp=sharing

Instructions to start the fine tuning process:
You must make sure you have installed the following libraries: Pytorch, transformers, datasets

Run the following command in the folder where run_qa_modified.py is located:

python run_qa_modified.py --model_name_or_path MODEL_NAME --train_file "TRAIN_DATA_PATH" --do_train --per_device_train_batch_size 8 --learning_rate 3e-5 --num_train_epochs 6 --max_seq_length 384 --doc_stride 128 --output_dir "OUTPUT_DIR"

Where you replace: MODEL_NAME with a model from huggingface(in our case roberta-base), TRAIN_DATA_PATH with the directory of the train data and OUTPUT_DIR with the path to the folder where you want the model to be outputted. Other parameters should be adjusted as needed, mainly the num_train_epoch and per_device_train_batch_size to the limitations of your hardware.

Instructions to run compare_results_torch:
Make sure you have installed the following dependencies: Pytorch, transformers, nltk, editdistance. 
Also make sure to download stopwords and punkt via this simple script:

import nltk
nltk.download('stopwords')
nltk.download('punkt')

In the file make sure to change the paths for: in line 41: to the path of your test data, in line 59 model to the path to the folder in which you have your model and the tokenizer to the appropriate tokenizer for the model(the exact string that needs to be input can be found on www.huggingface.co)
