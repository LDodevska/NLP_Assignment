

# Named Entity Recognition project for NLP course @ FRI.


For this project we developed different methods for name entity recognition on two datasets:
- [CoNLL2003](https://github.com/patverga/torch-ner-nlp-from-scratch/tree/master/data/conll2003/)
- GMB (Groningen Meaning Bank)  


You can find the weights for each model and the GMB dataset [here](https://drive.google.com/drive/folders/1RwjQe5-VEaFRXwt6E2A1B5GCF8BDdjXX?usp=sharing.). 

There are three files on drive. `ner_csv.` is the original dataset. `ner_first_preprocessing.csv` is the file with removed duplicates and NaNs. `sentense_dict.json` is the dictionary with sentences (there is an example in `initial_dataset_analysis.ipynb`). For each sentence we keep the words and lemmas with their corresponding POS and NE tags.

## Requirements

### BERT on PyTorch
We exported the [conda environment](requirements/nlp-pytorch.yml) for ease of use. To import the file execute `conda env create -f nlp-pytorch.yml` and `conda activate nlp-pytorch`. 

If you want to retrain the model you can execute \
`python parser.py --dataset conll --file_name ./eng.train` or \
`python parser.py --dataset gmb --file_name ./ner_first_preprocessing.csv` and it will generate json file for that dataset. 

Download the weights from the google drive link above. They are located in the *BERT models* folder. The file **conll_ner_bert_pt** are weights trained on CoNLL2003  and **gmb_ner_bert_pt** are trained on GMB dataset.
After the installations are complete, you can run the **BERT PyTorch.ipynb**. You need to execute the first cell (to import the needed dependencies) and depending on which model you are going to test, execute the cell with that specific tags (they have titles above them to differentiate the two models). Then scroll to the bottom where you will find further instructions. 

### ELMo + Bi-LSTM

- tensorflow==1.15 `conda install -c conda-forge tensorflow=1.15.0`
- tensorflow-gpu==1.15.0 `conda install -c anaconda tensorflow-gpu=1.15.0`
- tensorflow_hub=0.6 `conda install -c conda-forge tensorflow-hub=0.6.0`
- PyYAML==3.13 `conda install -c anaconda pyyaml=3.13.0`
- keras==2.3.1 `conda install -c conda-forge keras=2.3.1`
- sklearn
- sklearn_crfsuite `conda install -c conda-forge sklearn-crfsuite`
- numpy, pandas, json

If you are using **pip** you can use the [requirements.txt](requirements/elmo-requirements.txt) `pip install elmo-requirements.txt`

The evaluation of the model can be done by executing the `run_ner.py`. This function accepts 3 arguments:
- sentence:  the sentence you want to evaluate - (string)
- type: the model name: CONLL or GMB - (string)
- show_example: if set to True, it will evaluate 3 predefined sentences - (boolean)\
Example: \
`run_ner.py --sentence "I just had breakfast in London in Blue Cafe" --type GMB`\
`run_ner.py --type GMB --show_example True`


### Bi-LSTM + CRF

- tensorflow==2.0.0
- keras==2.3.1
- keras_contrib==0.0.2
- sklearn_crfsuite
- numpy, pandas, json

You can find the weights for the best Bi-LSTM + CRF model (trained on GMB) in the folder **BI-LSTM + CRF** on drive. Just upload the model in the notebook [BILSTM_CRF_v2.ipynb](https://github.com/LDodevska/NLP_Assignment/blob/master/BILSTM_CRF_v2.ipynb) and continue with your analysis.
