
# Named Entity Recognition project for NLP course @ FRI.


For this project we developed different methods for name entity recognition on two datasets:
- [CoNLL2003](https://github.com/patverga/torch-ner-nlp-from-scratch/tree/master/data/conll2003/)
- GMB (Groningen Meaning Bank)  


You can find the weights for each model and the GMB dataset [here](https://drive.google.com/drive/folders/1RwjQe5-VEaFRXwt6E2A1B5GCF8BDdjXX?usp=sharing.). 

There are three files on drive. `ner_csv.` is the original dataset. `ner_first_preprocessing.csv` is the file with removed duplicates and NaNs. `sentense_dict.json` is the dictionary with sentences (there is an example in `initial_dataset_analysis.ipynb`). For each sentence we keep the words and lemmas with their corresponding POS and NE tags.

## Requirements

### BERT on PyTorch
We exported the [conda environment](env/nlp-pytorch.yml) for ease of use. To import the file execute `conda env create -f nlp-pytorch.yml` and `conda activate nlp-pytorch`. 

If you want to retrain the model you can execute 

`python parser.py --dataset conll --file_name ./eng.train` or 

`python parser.py --dataset gmb --file_name ./ner_first_preprocessing.csv` and it will generate json file for that dataset. 

Download the weights from the google drive link above. They are located in the *weights* folder. The file **conll_ner_bert_pt** are weights trained on CoNLL2003  and **gmb_ner_bert_pt** are trained on GMB dataset.
After the installations are complete, you can run the **BERT PyTorch.ipynb**. You need to execute the first cell (to import the needed dependencies) and depending on which model you are going to test, execute the cell with that specific tags (they have titles above them to differentiate the two models). Then scroll to the bottom where you will find further instructions. 

### Bi-LSTM + ELMo



