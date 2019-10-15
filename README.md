
## Task Description

- Receives as input the title of a English Wikipedia article.
- Retrieves the text of that article from the MediaWiki API. If using Python, consider using python-mwapi for this.
- Identifies individual sentences within that text, along with the corresponding section titles. If using Python, mwparserfromhell can help you work with wiki markup.
- Runs those sentences through the model to classify them.
- Outputs the sentences, one per line, sorted by score given by the model.



The model given runs on python 2 and the Wikimedia API only runs on python 3 because of this, the script can only be run using python 3.


When *extract_and_classify.py* is run, it gives output a "prediction" score of each sentence.

To run this app, you need to have python 3 installed and create virtual environment using `virtualenv env`

- activate virtual environment using `source env/bin/activate`
- Install necessary libraries using `pip install -r requirements.txt`
-run the script, you can use the following command:

	```
	python extract_and_classify.py -m models/fa_en_model_rnn_attention_section.h5 -v embeddings/word_dict.pck -s embeddings/section_dict.pck -o output_folder
	```

The script output the following: 

- "title", i.e the title of the sentence to be classified
- "statement", i.e. the text of the sentence to be classified
- "prediction", i.e. the prediction is the prediction score
  
Keys:
- **'-o', '--out_dir'**, is the output directory where the result is stored
- **'-m', '--model'**, is the path to the model which is used for classifying the statements.
- **'-v', '--vocab'**, is the path to the vocabulary which is used for classifying the statements.
- **'-s', '--sections'**, is the path to the vocabulary of section with which the model was trained on.


### System Requirements
There are some requirements for this script to run smoothly. Below are the versions which they'd need to have in order to run this script.  

Python 3

- mwapi==0.5.1
- pandas==0.23.0
- nltk==3.3
- keras==2.1.5
- tensorflow==1.7.0
- sklearn==0.18.1
