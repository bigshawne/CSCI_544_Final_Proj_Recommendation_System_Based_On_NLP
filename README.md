# Given Users Recommandation Based on Reviews on Yelp

## Abstract
This project is the final project for USC CSCI-544 Applied Natural Language Proccessing. This project aims to provide 
users more accuracte recommandation for resturants based on performance semantic analysis over user"s previous reviews
left on Yelp. Due to the limation in size, all the data set cannot be uploaded. To access the origial data, please visit
the [Yelp Data Set](https://www.yelp.com/dataset). The content below will provide a workflow about how to recur the build
of our system.

## Data
Please refer to the following Google Drive link for all the data used and generated in this project. <br>
[Google Drive link](https://drive.google.com/drive/folders/1CXQdXnkDwgQBnfxw2PqHGGriX9qVOiIO?usp=sharing)

## Data Selection
This project mainly works with the three data sets: business, reviews, and users. These data sets are in .json format 
can be accessed in 
[Yelp Data Set](https://www.yelp.com/dataset). To perform the data selection, please run the following .py file with 
command line and the jupyter notebook file.

### Data Selection on businesses and users data set.
Firstly, please the run the following CLI to preform the data selection over businesses and users data set.

```commandline
 python data_preprare.py {business_path} {user_path}
```

### Data Selection on reviews data set.

Next, please run the "**review_json.ipynb**" jupyter notebook file. This file will select the reviews from the original reviews 
data set. Notice that this module applied the Hadoop for MapReduce, and it applied the python implementation 
pyspark package. This process will do the initial data cleaning, including lower all the characters, removing the stop 
words, and removing all the non-alphabetic characters.

## Data Preprocessing and Lemmatization

Then, please run the preprocess.py file by the following CLI. This module will try to perform the misspelled word 
correction to reduce the vocabulary size. It will also perform lemmatization over the reviews to check the impact of 
lemmatization on performance. Notice that this module mainly works with the "**nltk**" library for lemmatizaiton, and
"**pyspellchecker**" library for misspell word detection and correction. It also uses the "**multiprocess**" library
for parallel computing, so please adjust the number of cpu used in the <code>Pool()</code> function.

## Train Test Split

Please use the following CLI to perform the train test split. In this project, we split the whole data set into train 
and test partitions with ration 4:1.

```commandline
python train_test_split.py
```

## Word Embedding
In this project, we use word2vec model to do word level embedding, and BERT for sentence level embedding.

### Word2Vec Embedding
For the Word2Vec Embedding, we mainly work with the "gensim" package from python. We applied two pretrained word embedding
models: "glove-wiki-gigaword-300" and "word2vec-google-news-300"."gensim" also allows us to train our own word2vec 
embedding with our data set.

By running the following CLI, it will generate the word2vec embedding based on the "glove-wiki-gigaword-300".
```commandline
python word_embedding.py
```

By running the following jupyter notebook, it will generate the word2vec embedding based on "word2vec-google-news-300"

"**google_embedding.ipynb**"

By running the following files, it trains a word2vec model based on our data set, and generate the corresponding embedding.

"**myword2vec_lemma_train.ipynb**"

"**myword2vec_no_lemma_train.ipynb**"

### BERT Embedding
The "**bert_prep.ipynb**" jupyter notebook perform the BERT embedding. The inputs of "**bert_prep.ipynb**" are two revised data files: "**lemma_train.json**", "**lemma_test.json**", "**review_info.json**", 
**"review_text.json**" (or no_lemma_train.json, no_lemma_test.json)

## Item CF

### Convert word embedding from json to txt for Word_vec2Sent_vec.ipynb
Run the following files to prepare the inputs for next steps
"**GloVe/input/reformat.ipynb**"
"**Google/input/reformat.ipynb**"
"**Own_Model/input/reformat.ipynb**"

### Convert Word embedding to Sentence embedding
Run the following file to embed sentences to vectors.
Remark: Should have word embedding files in corresponding model/input folder ready.
Remark: Should have models.py downloaded in the same folder. (models.py is the FaceBook InferSent class)
"**Word_vec2Sent_vec.ipynb**"

### Generate similarity dictionary between reviews
Run the following CLI to generate similarity dictionary.
```commandline
python gen_test_sim2.py {mode} {model}
```
mode choice: 1 for with lemma, 2 for without lemma
model choice: Bert, GloVe, Google, Own, Own200
Remark: Should have sentence embedding json in corresponding folder

### Original Item CF model
Run the following CLI to train the original item CF model
```commandline
 python item_CF.py
```
Remark: Should have review_info.json in revised_data folder ready.
Remark: RMSE is directly printed

### Item CF with reviews involved
Run the following CLI to train the item CF with review model
```commandline
 python item_CF_w_review.py {mode} {model}
```
mode choice: 1 for with lemma, 2 for without lemma
model choice: Bert, GloVe, Google, Own, Own200
Remark: mode and model specify which model embedding will be used for item_CF_w_review.py and should have corresponding files in embed_sim folder. If there is no such file error, run gen_test_sim2.py first.
Remark: RMSE is directly printed
