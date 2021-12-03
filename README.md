# Given Users Recommandation Based on Reviews on Yelp

## Abstract
This project is the final project for USC CSCI-544 Applied Natural Language Proccessing. This project aims to provide 
users more accuracte recommandation for resturants based on performance semantic analysis over user's previous reviews
left on Yelp. Due to the limation in size, all the data set cannot be uploaded. To access the origial data, please visit
the [Yelp Data Set](https://www.yelp.com/dataset). The content below will provide a workflow about how to recur the build
of our system.

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

Next, please run the '**review_json.ipynb**' jupyter notebook file. This file will select the reviews from the original 
reviews' data set. Notice that this module applied the Hadoop for MapReduce, and it applied the python implementation 
pyspark package. This process will do the initial data cleaning, including lower all the characters, removing the stop 
words, and removing all the non-alphabetic characters.

## Data Preprocessing and Lemmatization

Then, please run the preprocess.py file by the following CLI. This module will try to perform the misspelled word 
correction to reduce the vocabulary size. It will also perform lemmatization over the reviews to check the impact of 
lemmatization on performance. Notice that this module mainly works with the '**nltk**' library for lemmatizaiton, and
'**pyspellchecker**' library for misspell word detection and correction. It also uses the '**multiprocess**' library
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

