# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 15:56:14 2018

@author: alexa
"""

import numpy as np
import pandas as pd 
import re, string, unicodedata, random, os, itertools, time, contractions,inflect
from string import punctuation as punc
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import treebank, stopwords
from nltk.tag.sequential import ClassifierBasedPOSTagger
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
nltk.download('treebank')
train_sents = treebank.tagged_sents()
stemmer = SnowballStemmer('english')
tagger = ClassifierBasedPOSTagger(train=train_sents)

import plotly.tools as tls
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
punc = string.punctuation
#========================== DATA ======================


train = pd.read_csv('C:/Users/alexa/OneDrive/Documents/Dataset/train.csv', sep =",")
test = pd.read_csv('C:/Users/alexa/OneDrive/Documents/Dataset/test.csv', sep=",")

print("Le corpus d'entrainement contient {0} commentaires et {1} colonnes \n, le corpus  \
contients {2} mots".format(len(train), len(train.columns), len(" ".join(train["comment_text"]).split())))

print("Le corpus de validation contient {0} commentaires et {1} colonnes \n, le corpus  \
contients {2} mots".format(len(test), len(test.columns), len(" ".join(test["comment_text"]).split())))

"""Le corpus d'entrainement contient 159 571 commentaires 
et 21 colonnes  le corpus  contients 10 734 904 mots

Le corpus de validation contient 153 164 commentaires 
et 15 colonnes , le corpus  contients 9 436 549 mots
"""

### PAR CORPUS :
toxic = train[train["toxic"] ==1]
svr_tox = train[train["severe_toxic"] ==1]
identity_hate = train[train["identity_hate"] ==1]
obscene = train[train["obscene"] ==1]
threat = train[train["threat"] ==1]
insult = train[train["insult"] ==1]
clean_com = train.loc[(train['toxic'] == (0)) 
                      & (train['severe_toxic'] == (0))
                      & (train['obscene'] == (0)) 
                      & (train['threat'] == (0))
                      & (train['insult'] == (0))]


text_severe_tox = " ".join(svr_tox.comment_text)
text_tox = " ".join(toxic.comment_text)
txxt_identity_hate = " ".join(identity_hate.comment_text)
txxt_obscene = " ".join(obscene.comment_text)
txxt_threat = " ".join(threat.comment_text)
txxt_insult = " ".join(insult.comment_text)
clean_text = " ".join(clean_com.comment_text)

def views(list_) :
    return print("Le corpus à une taille de {0} lignes et est composé de {1} mots".format(len(list_), len(" ".join(list_).split())))

for corpus in [clean_text, txxt_insult, txxt_threat,
 txxt_obscene, txxt_identity_hate, text_tox, text_severe_tox] :
    views(corpus)

""" Le corpus clear à une taille de 58 132 963 lignes et est composé de 47 790 598 mots
Le corpus d'insulte à une taille de 2 192 048 lignes et est composé de 1 792 692 mots
Le corpus de menace à une taille de 147 577 lignes et est composé de 120 014 mots
Le corpus d'obscénité à une taille de 2 431 493 lignes et est composé de 1 990 632 mots
Le corpus d'haine raciale à une taille de 434 904 lignes et est composé de 358 022 mots
Le corpus de commentaires toxiques à une taille de 4 530 786 lignes et est composé de 3 703 988 mots
Le corpus de commentaires sévèrement toxique à une taille de 725 142 lignes et est composé de 597 676 mots"""

#==================== BASIC EXPLORATION ======================
#Multilabel plot 

colors_list = ["brownish green", "pine green", "ugly purple",
               "blood", "deep blue", "brown", "azure"]

palette= sns.xkcd_palette(colors_list)

x=train.iloc[:,2:8].sum()

plt.figure(figsize=(9,6))
ax= sns.barplot(x.index, x.values,palette=palette)
plt.title("Nombres d'occurences par label")
plt.ylabel('Occurrences', fontsize=12)
plt.xlabel('Label ')
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 10, label, 
            ha='center', va='bottom')

plt.show()


#tags per comment 

rowsums=train.iloc[:,2:8].sum(axis=1)

x=rowsums.value_counts()

#plot
plt.figure(figsize=(8,4))
ax = sns.barplot(x.index, x.values, alpha=0.8)
plt.title("Plusieurs labels par commentaires")
plt.ylabel('# occurences', fontsize=12)
plt.xlabel('# tags ', fontsize=12)

#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()


"""
 ================================I) 1) stats descriptive =============================================
 =====================================================================================================
=======================================================================================================
=======================================================================================================
=======================================================================================================
=======================================================================================================
=======================================================================================================
"""

def feature(df) :
    df['word_count'] = df['comment_text'].apply(lambda x : len(x.split()))
    df['char_count'] = df['comment_text'].apply(lambda x : len(x.replace(" ","")))
    df['word_density'] = df['word_count'] / (df['char_count'] + 1)
    df['punc_count'] = df['comment_text'].apply(lambda x : len([a for a in x if a in punc]))
    df['total_length'] = df['comment_text'].apply(len)
    df['capitals'] = df['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['caps_vs_length'] = df.apply(lambda row: float(row['capitals'])/float(row['total_length']),axis=1)
    df['num_exclamation_marks'] =df['comment_text'].apply(lambda x: x.count('!'))
    df['num_question_marks'] = df['comment_text'].apply(lambda x: x.count('?'))
    df['num_punctuation'] = df['comment_text'].apply(lambda x: sum(x.count(w) for w in '.,;:'))
    df['num_symbols'] = df['comment_text'].apply(lambda x: sum(x.count(w) for w in '*&$%'))
    df['num_unique_words'] = df['comment_text'].apply(lambda x: len(set(w for w in x.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['word_count']
    df["word_unique_percent"] =  df["num_unique_words"]*100/df['word_count']
    return df

for dframe in tqdm([train, test]) : 
    feature(dframe)

""" Scale data for radar plot on the same scale"""

for col in train.select_dtypes(exclude = ["O"]).columns : 
    X_std = (train[col] - train[col].min(axis=0)) / (train[col].max(axis=0) - train[col].min(axis=0))
    X_scaled = X_std * (max(X_std) - min(X_std)) + min(X_std)
    train[col] = X_scaled

clean_com = train.loc[(train['toxic'] == (0)) 
                      & (train['severe_toxic'] == (0))
                      & (train['obscene'] == (0)) 
                      & (train['threat'] == (0))
                      & (train['insult'] == (0))]


# RADAR PLOT
Data = [
    go.Scatterpolar(
        r = [train.loc[train["toxic"] ==1]['total_length'].median(),
                train.loc[train["toxic"] ==1]['word_count'].median(), 
                train.loc[train["toxic"] ==1]['num_unique_words'].median(),
                train.loc[train["toxic"] ==1]['caps_vs_length'].median(), 
                 train.loc[train["toxic"] ==1]['char_count'].median()],
                theta = ['Total_Lenght','WordCount',
                         'Count_unique_words', "Capitals_VS_Length", "Charcount"],
                fill = 'toself', 
                line = dict( color = 'brown'), 
                name=  "Toxic Statistiques", subplot = "polar"),
        
        go.Scatterpolar(
            r = [train.loc[train["obscene"] ==1]['total_length'].median(),
                train.loc[train["obscene"] ==1]['word_count'].median(), 
                train.loc[train["obscene"] ==1]['num_unique_words'].median(),
                train.loc[train["obscene"] ==1]['caps_vs_length'].median(), 
                 train.loc[train["obscene"] ==1]['char_count'].median()],
                theta = ['Total_Lenght','WordCount',
                         'Count_unique_words', "Capitals_VS_Length", "Charcount"],
                fill = 'toself', 
                line = dict( color = 'magenta'), 
                name=  "obscene Statistiques", subplot = "polar2"),
        
        go.Scatterpolar(
            r = [train.loc[train["severe_toxic"] ==1]['total_length'].median(),
                train.loc[train["severe_toxic"] ==1]['word_count'].median(), 
                train.loc[train["severe_toxic"] ==1]['num_unique_words'].median(),
                train.loc[train["severe_toxic"] ==1]['caps_vs_length'].median(), 
                 train.loc[train["severe_toxic"] ==1]['char_count'].median()],
                theta = ['Total_Lenght','WordCount',
                         'Count_unique_words', "Capitals_VS_Length", "Charcount"],
                fill = 'toself', 
                line = dict( color = 'orange'), 
                name=  "severe_toxic Statistiques", subplot = "polar3"),
        
        go.Scatterpolar(
            r = [train.loc[train["threat"] ==1]['total_length'].median(),
                train.loc[train["threat"] ==1]['word_count'].median(), 
                train.loc[train["threat"] ==1]['num_unique_words'].median(),
                train.loc[train["threat"] ==1]['caps_vs_length'].median(), 
                 train.loc[train["threat"] ==1]['char_count'].median()],
                theta = ['Total_Lenght','WordCount',
                         'Count_unique_words', "Capitals_VS_Length", "Charcount"],
                fill = 'toself', 
                line = dict( color = 'green'), 
                name=  "threat Statistiques", subplot = "polar4"),
        
        go.Scatterpolar(
            r = [train.loc[train["insult"] ==1]['total_length'].median(),
                train.loc[train["insult"] ==1]['word_count'].median(), 
                train.loc[train["insult"] ==1]['num_unique_words'].median(),
                train.loc[train["insult"] ==1]['caps_vs_length'].median(), 
                 train.loc[train["insult"] ==1]['char_count'].median()],
                theta = ['Total_Lenght','WordCount',
                         'Count_unique_words', "Capitals_VS_Length", "Charcount"],
                fill = 'toself', 
                line = dict( color = 'blue'), 
                name=  "insult Statistiques", subplot = "polar5"),
        
        go.Scatterpolar(
            r = [clean_com['total_length'].median(),
            clean_com['word_count'].median(), clean_com['num_unique_words'].median(),
            clean_com['caps_vs_length'].median(), clean_com['char_count'].median()],
            theta = ['Total_Lenght','WordCount',
                     'Count_unique_words', "Capitals_VS_Length", "Charcount"],
            fill = 'toself', 
            line =  dict(color = "red"),
            name = 'Commentaires jugée normale', subplot = "polar6")
       ]

layout = go.Layout(
    polar3 = dict(
      domain = dict(
        x = [0, 0.3],
        y = [0.55, 1]
      ),
      radialaxis = dict(visible = True,)),

    polar2 = dict(
      domain = dict(
        x = [0, 0.3],
        y = [0, 0.45]
      ),
      radialaxis = dict(visible = True,)),
        
    polar = dict(
      domain = dict(
        x = [0.33, 0.6525],
        y = [0, 0.45]
      ),
      radialaxis = dict(visible = True,)),
        
    polar4 = dict(
      domain = dict(
        x = [0.33, 0.6525],
        y = [0.55, 1]
      ),
      radialaxis = dict(visible = True,)),
        
    polar5 = dict(
        domain = dict(
        x = [0.6775, 1],
        y = [0, 0.45]
      ),
      radialaxis = dict(visible = True,)),

    polar6= dict(
        domain = dict(
        x = [0.6775, 1],
        y = [0.55, 1]
      ),
      radialaxis = dict(visible = True,)), 
    
    title =  "Statistiques textuelles médianne comparées")

fig = go.Figure(data=Data, layout=layout)
iplot(fig)  


#spammer : word unique ??
train['num_unique_words'].loc[train['num_unique_words']>200] = 200
train["word_unique_percent"] =  train["num_unique_words"]*100/train['word_count']
spammers=train[train['word_unique_percent']<30]

x=spammers.iloc[:,2:8].sum()
plt.figure(figsize=(18,7))
plt.title("Compte du nombre de commentaires < 30% de mots unique",fontsize=15)
ax=sns.barplot(x=x.index, y=x.values,color='crimson')

#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.xlabel('Labels', fontsize=12)
plt.ylabel("Nombre de commentaires", fontsize=12)
plt.show()

#spammer exemple
spammer = train[train["word_unique_percent"] < 2]
print(spammer[spammer["toxic"] == 1]["comment_text"].iloc[5])
"""Go fuck yourself! Go fuck yourself! Go fuck yourself! Go fuck yourself! 
Go fuck yourself! Go fuck yourself! Go fuck yourself! Go fuck yourself! 
Go fuck yourself! Go fuck yourself! Go fuck yourself! 
Go fuck yourself! Go fuck yourself! Go fuck yourself! ... ETC """



#points d'exclamation 

trace0= go.Histogram(
    x= clean_com["num_exclamation_marks"],autobinx=True, 
    showlegend=False
)

trace1= go.Histogram(
    x= train[train["toxic"] ==1]["num_exclamation_marks"],autobinx=True,
    showlegend=False
)
trace2 = go.Histogram(
    x= train[train["severe_toxic"] ==1]["num_exclamation_marks"],autobinx=True,
    showlegend=False
)

trace3 = go.Histogram(
    x= train[train["identity_hate"] ==1]["num_exclamation_marks"],autobinx=True,
    showlegend=False
)

trace4 = go.Histogram(
    x= train[train["insult"] ==1]["num_exclamation_marks"],autobinx=True,
    showlegend=False
)

trace5 = go.Histogram(
    x= train[train["threat"] ==1]["num_exclamation_marks"],autobinx=True,
    showlegend=False
)

trace6 = go.Histogram(
    x= train[train["obscene"] ==1]["num_exclamation_marks"],autobinx=True,
    showlegend=False
)

#Creating the grid
fig = tls.make_subplots(rows=3, cols=3, specs=[[{'colspan': 1}, None, None], [{}, {}, {}], [{}, {}, {}]],
                          subplot_titles=("Commentaires normale",
                                          "Commentaires toxique", 
                                          "Commentaires sévèrement toxique",
                                          "Commentaires de haine identitaire", 
                                          "Commentaires d'insultes", 
                                          "Commentaires de menaces", 
                                          "Commentaires obscènes"))
#setting the figs
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 2, 1)
fig.append_trace(trace2, 2, 2)
fig.append_trace(trace3, 2, 3)
fig.append_trace(trace4, 3, 1)
fig.append_trace(trace5, 3, 2)
fig.append_trace(trace6, 3, 3)

fig['layout'].update(showlegend=True, title="Nombre de points d'exclamation entre les topics", xaxis=dict(range=[0, 10]), xaxis2=dict(range=[0, 70]),
                     xaxis3=dict(range=[0, 50]), xaxis4=dict(range=[0, 50]), 
                     xaxis5=dict(range=[0, 50]), xaxis6=dict(range=[0, 60]), xaxis7=dict(range=[0, 50]))
iplot(fig)

""" 
 =====================================================================================================
=======================================================================================================
=======================================================================================================
=======================================================================================================
============================================== I) 2) POS TAGGING ======================================
=======================================================================================================
 =====================================================================================================
=======================================================================================================
=======================================================================================================
=======================================================================================================
=======================================================================================================
=======================================================================================================
"""

# subset 
     
# https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html   
""" Une première partie sera pour avoir un détail des tags sous forme de liste
et la deuxième partie le nombre de tag en générale """

""" create class for POS TAG and lemmatisation """
class Splitter(object):
    """
    split the document into sentences and tokenize each sentence
    """
    def __init__(self):
        self.splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def split(self,text):
        # split into single sentence
        sentences = self.splitter.tokenize(text)
        # tokenization in each sentences
        tokens = [self.tokenizer.tokenize(sent) for sent in sentences]
        return tokens

class LemmatizationWithPOSTagger(object):
    def __init__(self):
        pass
    def get_wordnet_pos(self,treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def pos_tag(self,tokens):
        # find the pos tagging for each tokens [('What', 'WP'), ('can', 'MD'), ('I', 'PRP') ....
        pos_tokens = [nltk.pos_tag(token) for token in tokens]

        # lemmatization using pos tagg   
        #  [('What', 'What', ['WP']), ('can', 'can', ['MD']) --> WORD/LEMMA/POSTAG
        pos_tokens = [ [(word, lemmatizer.lemmatize(word,self.get_wordnet_pos(pos_tag)), pos_tag) for (word, pos_tag) in pos] for pos in pos_tokens]
        return pos_tokens

lemmatizer = WordNetLemmatizer()
splitter = Splitter()
lemmatization_using_pos_tagger = LemmatizationWithPOSTagger()

tokens = splitter.split(text_severe_tox) 
svr_tox = lemmatization_using_pos_tagger.pos_tag(tokens)

tokens = splitter.split(text_tox) 
tox = lemmatization_using_pos_tagger.pos_tag(tokens)

tokens = splitter.split(txxt_obscene) 
obsc = lemmatization_using_pos_tagger.pos_tag(tokens)

tokens = splitter.split(txxt_insult) 
insult = lemmatization_using_pos_tagger.pos_tag(tokens)

tokens = splitter.split(txxt_identity_hate) 
hate = lemmatization_using_pos_tagger.pos_tag(tokens)

tokens = splitter.split(clean_text) 
clean_lemma = lemmatization_using_pos_tagger.pos_tag(tokens)

import pickle

with open('C:/Users/alexa/Desktop/M2_EA/cleantext.pkl', 'wb') as f:
    pickle.dump(clean_lemma, f)

with open('C:/Users/alexa/Desktop/M2_EA/svr_tox_text.pkl', 'wb') as f: 
	pickle.dump(svr_tox, f) 

with open('C:/Users/alexa/Desktop/M2_EA/hate.pkl', 'wb') as f:
	pickle.dump(hate, f)

with open('C:/Users/alexa/Desktop/M2_EA/obscene.pkl', 'wb') as f: 
	pickle.dump(obsc, f) 

with open('C:/Users/alexa/Desktop/M2_EA/tox_text.pkl', 'wb') as f:
	pickle.dump(tox, f) 

with open('C:/Users/alexa/Desktop/M2_EA/insult_text.pkl', 'wb') as f:
	pickle.dump(insult, f)

#use pickle to save every list with pos tagging coz it takes more than 4hours to compute

"""========= =GENERATE POS TAGGING FEATURES =================
================================================================================
================================================================================
================================================================================
================================================================================
================================================================================
================================================================================
================================================================================
================================================================================"""


""" PEN TREE BANK TAG : list des besoins
CC | Coordinating conjunction |
CD | Cardinal number |
DT | Determiner |
EX | Existential there |
FW | Foreign word |
IN | Preposition or subordinating conjunction |
JJ | Adjective |
JJR | Adjective, comparative |
JJS | Adjective, superlative |
LS | List item marker |
MD | Modal |
NN | Noun, singular or mass |
NNS | Noun, plural |
NNP | Proper noun, singular |
NNPS | Proper noun, plural |
PDT | Predeterminer |
POS | Possessive ending |
PRP | Personal pronoun |
PRP$ | Possessive pronoun |
RB | Adverb |
RBR | Adverb, comparative |
RBS | Adverb, superlative |
RP | Particle |
SYM | Symbol |
TO | to |
UH | Interjection |
VB | Verb, base form |
VBD | Verb, past tense |
VBG | Verb, gerund or present participle |
VBN | Verb, past participle |
VBP | Verb, non-3rd person singular present |
VBZ | Verb, 3rd person singular present |
WDT | Wh-determiner |
WP | Wh-pronoun |
WP$ | Possessive wh-pronoun |
WRB | Wh-adverb |"""

with open('C:/Users/alexa/Desktop/M2_EA/insult_text.pkl', 'rb') as f:
    insult_list = pickle.load(f)
with open('C:/Users/alexa/Desktop/M2_EA/obscene.pkl', 'rb') as f:
    obscene_list = pickle.load(f)
with open('C:/Users/alexa/Desktop/M2_EA/tox_text.pkl', 'rb') as f:
    tox_list = pickle.load(f)
with open('C:/Users/alexa/Desktop/M2_EA/threat_text.pkl', 'rb') as f:
    threat_list = pickle.load(f)
with open('C:/Users/alexa/Desktop/M2_EA/hate.pkl', 'rb') as f:
    hate_list = pickle.load(f)
with open('C:/Users/alexa/Desktop/M2_EA/insult_text.pkl', 'rb') as f:
    svr_tox_list = pickle.load(f)
with open('C:/Users/alexa/Desktop/M2_EA/clean_text.pkl', 'rb') as f:
    clean_list = pickle.load(f)
    
def pos_tag_features(list_) :
    list_weight = len(' '.join(str(v[0][0]) for v in list_))
    POS_DICT = {}
    POS_DICT['count_nouns'] = sum([sum(1 for words in sentence if words[2] == 'NN' or words[2] == 'NNS' or words[2] == 'NNP' or words[2] == 'NNP') 
                                   for sentence in list_])/ list_weight
    POS_DICT['Passé'] = sum([sum(1 for words in sentence if words[2] == 'VBD') 
                                   for sentence in list_])/ list_weight
    POS_DICT['Adverbe_superlatif'] = sum([sum(1 for words in sentence if words[2] == 'RBS') 
                                   for sentence in list_])/ list_weight
    POS_DICT['Adjectives'] = sum([sum(1 for words in sentence if words[2] == 'JJ' or words[2] == 'JJR' or words[2] == 'JJS') 
                                   for sentence in list_])/ list_weight
    POS_DICT['Pronom_Possessive'] = sum([sum(1 for words in sentence if words[2] == 'WP$') 
                                   for sentence in list_])/ list_weight
    POS_DICT['interjection'] = sum([sum(1 for words in sentence if words[2] == 'UH') 
                                   for sentence in list_])/ list_weight
    POS_DICT['Symbole'] = sum([sum(1 for words in sentence if words[2] == 'SYM') 
                                   for sentence in list_])/ list_weight
    POS_DICT['Adverbe_comparative'] = sum([sum(1 for words in sentence if words[2] == 'RBR') 
                                   for sentence in list_])/ list_weight
    POS_DICT['pronom_personnel'] = sum([sum(1 for words in sentence if words[2] == 'PRP') 
                                   for sentence in list_])/ list_weight
    POS_DICT['determinant'] = sum([sum(1 for words in sentence if words[2] == 'DT') 
                                   for sentence in list_])/ list_weight
    POS_DICT['terminaison_possesive'] = sum([sum(1 for words in sentence if words[2] == 'POS') 
                                   for sentence in list_])/ list_weight
    POS_DICT['pronom_posseive'] = sum([sum(1 for words in sentence if words[2] == 'PRP$') 
                                   for sentence in list_])/ list_weight
    POS_DICT['ajdective_comparative'] = sum([sum(1 for words in sentence if words[2] == 'JJR') 
                                   for sentence in list_])/ list_weight
    POS_DICT['adjective_superlative'] = sum([sum(1 for words in sentence if words[2] == 'JJS') 
                                   for sentence in list_])/ list_weight
    POS_DICT['forme_verbal_classic'] = sum([sum(1 for words in sentence if words[2] == 'VB') 
                                   for sentence in list_])/ list_weight
    POS_DICT['participe_présent'] = sum([sum(1 for words in sentence if words[2] == 'VBG') 
                                   for sentence in list_])/ list_weight
    POS_DICT['participe_passé'] = sum([sum(1 for words in sentence if words[2] == 'VBN') 
                                   for sentence in list_])/ list_weight
    POS_DICT['count_verb'] = sum([sum(1 for words in sentence if words[2] == 'VB' or words[2] == "VBN" or words[2] == "VB" \
                                      or words[2] == "VBG" or words[2] == "VBP" or words[2] == "VBZ" or words[2] == "VBD") 
                                   for sentence in list_])/ list_weight
    POS_DICT['adjective_superlative'] = sum([sum(1 for words in sentence if words[2] == 'JJS') 
                                   for sentence in list_])/ list_weight
    POS_DICT["NOM_SINGULIER"] = sum([sum(1 for words in sentence if words[2] == 'NN') 
                                     for sentence in list_])/ list_weight
    POS_DICT["NOM_PLURIEL"] = sum([sum(1 for words in sentence if words[2] == 'NNS') 
                                   for sentence in list_])/ list_weight
    POS_DICT["Mots_étranger"] = sum([sum(1 for words in sentence if words[2] == 'FW') 
                                   for sentence in list_])/ list_weight
    
    return pd.DataFrame(POS_DICT, index = range(1))


clean_df = pos_tag_features(clean_list)
all_vices_list = threat_list +  tox_list + svr_tox_list + hate_list + obscene_list +  insult_list
all_vices_df = pos_tag_features(all_vices_list)

""" bar plot avec le nombre de tag general"""
tagged_form = [x for x in clean_df.columns]
tagged_label = [str(x) for x in tagged_form]
tagged_dict = dict(zip(tagged_label, tagged_form))


fig = tls.make_subplots(rows=1, cols=2, specs = [[{}, {}]])

for k,v in tagged_dict.items():
    tag_trace = go.Bar(x=clean_df[v], name = str(k), text=k, textposition = 'auto', 
                       marker=dict( color='rgb(58,202,225)',line=dict(color='rgb(8,48,107)',width=1.5),),
                       opacity=0.6, showlegend=False)
    fig.append_trace(tag_trace, 1, 1)
    
for k,v in tagged_dict.items():
    tag_trace = go.Bar(x=all_vices_df[v], name = str(k), text=k, textposition = 'auto', 
                       marker=dict( color='rgb(239, 243, 198)',line=dict(color='rgb(8,48,107)',width=1.5),),
                       opacity=0.6, showlegend=False)
    fig.append_trace(tag_trace, 1, 2)
    
fig['layout'].update(height=800, width=1000, title="Distribution Gramaticale/Verbale selon le corpus")
#fig['data'] = Data(verbal_data + tagged_data)
iplot(fig)

""" 
================================================================================
================================================================================
================================ FURTHER IN POSTAG EXPLORATION ======================="""

#nested dict basis for exploration without use dataframe for computation efficienty

big_dict = {"Toxique" :  {"FW" : [[words for words in sentence if words[2] == "FW"] for sentence in tox_list], 
                        "ADJ": [[words for words in sentence if words[2] == "JJ"] for sentence in tox_list], 
                       "Noun" : [[words for words in sentence if words[2] == "NN"] for sentence in tox_list], 
                       "Pronom_perso" : [[words for words in sentence if words[2] == "PRP"] for sentence in tox_list], 
                       "Verbe" : [[words for words in sentence if words[2] == "VB"] for sentence in tox_list]}, 
            
            "Sévèrement toxique" :  {"FW" : [[words for words in sentence if words[2] == "FW"] for sentence in svr_tox_list], 
                        "ADJ": [[words for words in sentence if words[2] == "JJ"] for sentence in svr_tox_list], 
                       "Noun" : [[words for words in sentence if words[2] == "NN"] for sentence in svr_tox_list], 
                       "Pronom_perso" : [[words for words in sentence if words[2] == "PRP"] for sentence in svr_tox_list], 
                       "Verbe" : [[words for words in sentence if words[2] == "VB"] for sentence in svr_tox_list]}, 
            
            "Raciste" : {"FW" : [[words for words in sentence if words[2] == "FW"] for sentence in hate_list], 
                        "ADJ": [[words for words in sentence if words[2] == "JJ"] for sentence in hate_list], 
                       "Noun" : [[words for words in sentence if words[2] == "NN"] for sentence in hate_list], 
                       "Pronom_perso" : [[words for words in sentence if words[2] == "PRP"] for sentence in hate_list], 
                       "Verbe" : [[words for words in sentence if words[2] == "VB"] for sentence in hate_list]},
            
            "Insultant" : {"FW" : [[words for words in sentence if words[2] == "FW"] for sentence in insult_list], 
                        "ADJ": [[words for words in sentence if words[2] == "JJ"] for sentence in insult_list], 
                       "Noun" : [[words for words in sentence if words[2] == "NN"] for sentence in insult_list], 
                       "Pronom_perso" : [[words for words in sentence if words[2] == "PRP"] for sentence in insult_list], 
                       "Verbe" : [[words for words in sentence if words[2] == "VB"] for sentence in insult_list]},
            
            "Obscène" : {"FW" : [[words for words in sentence if words[2] == "FW"] for sentence in obscene_list], 
                        "ADJ": [[words for words in sentence if words[2] == "JJ"] for sentence in obscene_list], 
                       "Noun" : [[words for words in sentence if words[2] == "NN"] for sentence in obscene_list], 
                       "Pronom_perso" : [[words for words in sentence if words[2] == "PRP"] for sentence in obscene_list], 
                       "Verbe" : [[words for words in sentence if words[2] == "VB"] for sentence in obscene_list]},
            
            "Menaçant" : {"FW" : [[words for words in sentence if words[2] == "FW"] for sentence in threat_list], 
                        "ADJ": [[words for words in sentence if words[2] == "JJ"] for sentence in threat_list], 
                       "Noun" : [[words for words in sentence if words[2] == "NN"] for sentence in threat_list], 
                       "Pronom_perso" : [[words for words in sentence if words[2] == "PRP"] for sentence in threat_list], 
                       "Verbe" : [[words for words in sentence if words[2] == "VB"] for sentence in threat_list]},
            
            "Canonique" : {"FW" : [[words for words in sentence if words[2] == "FW"] for sentence in clean_list], 
                        "ADJ": [[words for words in sentence if words[2] == "JJ"] for sentence in clean_list], 
                       "Noun" : [[words for words in sentence if words[2] == "NN"] for sentence in clean_list], 
                       "Pronom_perso" : [[words for words in sentence if words[2] == "PRP"] for sentence in clean_list], 
                       "Verbe" : [[words for words in sentence if words[2] == "VB"] for sentence in clean_list]}
           }


def most_common_pos(dict_key, tag, col):
    import itertools
    from collections import Counter
    list_of_list = [[words for words in sentences] for sentences in big_dict.get(dict_key).get(tag) if sentences != []]
    list_of_tuples = list(itertools.chain(*list_of_list))
    CounterList = Counter(list_of_tuples)
    topcommon = [(k[0], v) for k,v in CounterList.most_common(25)]
    xvals = list(reversed([_[0] for _ in topcommon]))
    yvals = list(reversed([_[1] for _ in topcommon]))
    trace = go.Bar(x=yvals, y=xvals, name=dict_key, marker=dict(color=col), xaxis=dict(linecolor='#fff',), opacity=0.7, orientation='h')
    return trace

""" Nous allons observer les tops words pour les tag qui nous interessent cad foreign word, adj, nom, pronom perso, verb"""

#TAGS IS VARIABLE --> WITH KEY OF DICT
trace0 = most_common_pos("Canonique", 'FW', '#4286f4')
trace1 = most_common_pos("Sévèrement toxique", 'FW', '#f44268')
trace2 = most_common_pos("Toxique", 'FW', '#e0d75e')
trace3 = most_common_pos("Raciste", 'FW', '#3e8441')
trace4 = most_common_pos("Menaçant", 'FW', '#4286f4')
trace5 = most_common_pos("Obscène", 'FW', '#f44268')
trace6 = most_common_pos("Insultant", 'FW',  'rgb(171,217,233)')

fig = tools.make_subplots(rows=3, cols=3, print_grid=False, specs=[[{'colspan': 2}, None, None], [{}, {}, {}], [{}, {}, {}]],
                          subplot_titles=("Conforme" ,
                                          "Sévèrement toxique",
                                          "Toxique", 
                                          "Raciste",
                                          "Menaçant", 
                                          "Obscène", 
                                          "Insulte"));

fig.append_trace(trace0, 1, 1);
fig.append_trace(trace1, 2, 1);
fig.append_trace(trace2, 2, 2);
fig.append_trace(trace3, 2, 3);
fig.append_trace(trace4, 3, 1);
fig.append_trace(trace5, 3, 2);
fig.append_trace(trace6, 3, 3);

fig['layout'].update(height=1000, title='Hiérarchisation des mots étrangers selon leurs degré de récurrence', legend=dict(orientation="v"));
iplot(fig);


"""
================================================
================================================================================
================================================================================
================================================================================
================================================================================
================================================================================
I) 3) Analyse UNI/BI/TRI - GRAM """ 

lab = ['toxic', 'severe_toxic', 'obscene', 'threat',
       'insult', 'identity_hate']

X = train["comment_text"].values
y = train[lab].values

from sklearn.preprocessing import StandardScaler

def get_feature_importances(model, analyzer, ngram, ngram2, lowercase, min_df=10, sampsize=40000):
    tfv = TfidfVectorizer(min_df=min_df,
                          strip_accents='unicode',
                          analyzer=analyzer,
                          ngram_range=(ngram, ngram2),
                          lowercase=lowercase)
    df_sample = train.sample(sampsize, random_state=123)
    X = tfv.fit_transform(df_sample.comment_text)
    scaler = StandardScaler(with_mean=False)
    scaler.fit(X)
    terms = tfv.get_feature_names()
    var_imp = pd.DataFrame(index=terms)
    #multiclassif
    for category in lab:
        y = df_sample[category].values
        model.fit(X, y)
        var_imp[category] =  np.sqrt(scaler.var_) * model.coef_[0]
    var_imp = var_imp.sort_values('toxic', ascending=False)
    return var_imp

model = LogisticRegression()
var_imp = get_feature_importances(model, analyzer='word', ngram=1, ngram2=1, lowercase=True)
var_imp.head(10)

var_imp = get_feature_importances(model, analyzer='word', ngram=2, ngram2 =2, lowercase=True)
var_imp.head(10)

var_imp = get_feature_importances(model, analyzer='word', ngram= 3, ngram2=3, lowercase=True)
var_imp.head(10)

""" plot top word """

stopword=set(STOPWORDS)
stopwords = [x.replace("\r","") for x in stopword]
$
def clean_textt(txt):    
    txt = txt.lower() #minuscule
    txt = "".join(x for x in txt if x not in punc).split() #split
    words = [wrd for wrd in txt if wrd not in stopwords] #remove stopwords
    words = [wrd for wrd in words if len(wrd) > 1] #longueur du mot >1
    txt = " ".join(words)
    return txt

def ngrams(txt, n):
    txt = txt.split()
    output = []
    for i in range(len(txt)-n+1):
        output.append(" ".join(txt[i:i+n]))
    return output

def get_unigrams_data(txt, tag, col):
    cleaned_text = clean_textt(txt)
    all_bigrams = ngrams(cleaned_text, 1)
    topbigrams = Counter(all_bigrams).most_common(25)
    xvals = list(reversed([_[0] for _ in topbigrams]))
    yvals = list(reversed([_[1] for _ in topbigrams]))
    trace = go.Bar(x=yvals, y=xvals, name=tag, marker=dict(color=col), xaxis=dict(linecolor='#fff',), opacity=0.7, orientation='h')
    return trace


text_severe_tox = " ".join(svr_tox.comment_text) 
text_tox = " ".join(toxic.comment_text)
txxt_identity_hate = " ".join(identity_hate.comment_text) 
txxt_obscene = " ".join(obscene.comment_text)
txxt_threat = " ".join(threat.comment_text) 
txxt_insult = " ".join(insult.comment_text)
clean_text =  " ".join(clean_com.comment_text)


trace0 = get_unigrams_data(clean_text, 'Canonique', '#e0d75e')
trace1 = get_unigrams_data(text_severe_tox, 'Sévèrement toxique', '#4286f4')
trace2 = get_unigrams_data(text_tox, 'Toxique', '#f44268')
trace3 = get_unigrams_data(txxt_identity_hate, 'Raciste', '#e0d75e')
trace4 = get_unigrams_data(txxt_threat, 'Menaçant', '#3e8441')
trace5 = get_unigrams_data(txxt_obscene, 'Obscène', '#4286f4')
trace6 = get_unigrams_data(txxt_insult, 'Insultant', '#f44268')

fig = tools.make_subplots(rows=3, cols=3, print_grid=False, specs=[[{'colspan': 2}, None, None], [{}, {}, {}], [{}, {}, {}]],
                          subplot_titles=("Conforme" ,
                                          "Sévèrement toxique",
                                          "Toxique", 
                                          "Raciste",
                                          "Menaçant", 
                                          "Obscène", 
                                          "Insulte"));

fig.append_trace(trace0, 1, 1);
fig.append_trace(trace1, 2, 1);
fig.append_trace(trace2, 2, 2);
fig.append_trace(trace3, 2, 3);
fig.append_trace(trace4, 3, 1);
fig.append_trace(trace5, 3, 2);
fig.append_trace(trace6, 3, 3);

fig['layout'].update(height=800, title='Hiérarchisation des allocutions selon leurs degré de récurrence', legend=dict(orientation="v"));
iplot(fig, filename='unigram_Plot');

""" 
=======================================================================
=======================================================================
=======================================================================
======================== II) modelisation comparative =================
=======================================================================
=======================================================================
=======================================================================
=======================================================================
"""

#feature engineering 
def count_regexp_occ(regexp="", text=None):
    return len(re.findall(regexp, text))

"""L'analyse du pos tagging nous a permis de démontrer que les commentaires haineux ne se différencié pas sur un nombre de temps, des pronoms, adjectives mais sur 
l'utilisation des pronom_personnel en masse pour s'addresser à leur victime on peut donc dégager des bad key word et des postag keyword"""

def regex_features(train) :
    train["nb_fuck"] = train["comment_text"].apply(lambda x: count_regexp_occ(r"[dD]ick", x))
    train["nb_shit"] = train["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Ss]hit\W", x))
    train["nb_nigga"] = train["comment_text"].apply(lambda x: count_regexp_occ(r"\Wnigga\W", x))
    train["nb_ass"] = train["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Aa]ss\W", x))
    train["nb_nigger"] = train["comment_text"].apply(lambda x: count_regexp_occ(r"\Wnigger\W", x))
    train["nb_fuck"] = train["comment_text"].apply(lambda x: count_regexp_occ(r"\W[fF]uck\W", x))
    train["nb_bitch"] = train["comment_text"].apply(lambda x: count_regexp_occ(r"\W[bB]itch\W", x))
    train["nb_die"] = train["comment_text"].apply(lambda x: count_regexp_occ(r"\W[dD]ie\W", x))
    train["nb_suck"] = train["comment_text"].apply(lambda x: count_regexp_occ(r"\W[sS]uck\W", x))
    train["nb_kill"] = train["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Kk]il\W", x))
    train["nb_vagina"] = train["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Vv]agina\W", x))
    train["nb_You"] = train["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Yy]ou\W", x))
    train["nb_gay"] = train["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Gg]ay\W", x))
    train["nb_puta"] = train["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Pp]uta\W", x))
    train["nb_nazi"] = train["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Nn]azi\W", x))
    train["nb_faggot"] = train["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Ff]aggot\W", x))
    train["nb_morron"] = train["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Mm]or[r]on\W", x))
    train["nb_mother"] = train["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Mm]other\W", x))
    train["nb_cunt"] = train["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Cc]unt\W", x))
    train["nb_son_of"] = train["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Ss]on of\W", x))
    train["yourself"] = train["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Yy]ourself\W", x))
    train["nb_u"] = train["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Uu]\W", x))
    train["nb_licker"] = train["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Ll]icker\W", x))
    train["nb_fucksex"] = train["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Ff]uck[Ss]ex\W", x))
    train["nb_penis"] = train["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Pp]enis\W", x))                                                                           
    train["is_spammer"] = np.where(train['word_unique_percent']<30, 1, 0)
    return train

for dframe in tqdm([train, test]) : 
    regex_features(dframe)

#stats
clear_com = train.loc[(train['toxic'] == (0))  & (train['severe_toxic'] == (0))& (train['obscene'] == (0)) & (train['threat'] == (0)) 
                      & (train['insult'] == (0))]

not_clear_com = train.loc[(train['toxic'] == (1))  | (train['severe_toxic'] == (1)) | (train['obscene'] == (1))  | (train['threat'] == (1)) 
                       | (train['insult'] == (1))]

not_clear_agg = pd.DataFrame(not_clear_com.select_dtypes(exclude = ["object"]).agg(np.mean)).T
clear_agg= pd.DataFrame(clear_com.select_dtypes(exclude = ["object"]).agg(np.mean)).T
notag = clear_agg[not_clear_com.select_dtypes(exclude = ["object"]).columns].T.rename(columns = {0 : "valeurs_corpus_normal"})
tag = not_clear_agg[not_clear_com.select_dtypes(exclude = ["object"]).columns].T.rename(columns = {0 : "valeurs_corpus_taggé"})

tag.join(notag)

# ========= MODELISATION ==================

test_labelled = pd.merge(test, y_true, on = "id", how = "inner")
print(test_labelled.shape)
test_labelled.head()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler 

sample_train =  train.sample(30000)
sample_test =  test_labelled.sample(30000)

lab = ['toxic', 'severe_toxic', 'obscene', 'threat',
       'insult', 'identity_hate']

num_features = [f_ for f_ in sample_train.columns
                if f_ not in ["comment_text", "id"] + lab]

train_num_features = sample_train[num_features].values
test_num_features = sample_test[num_features].values

train_text = sample_train['comment_text'].fillna("")
test_text =  sample_test['comment_text'].fillna("")

vectorizer = CountVectorizer()
vectorizer.fit(train['comment_text'])
train_text_matrix =vectorizer.transform(train_text)
test_test_matrix =vectorizer.transform(test_text)
scaler = StandardScaler(with_mean=False)
#scaler.fit(train_text_matrix)
scaler.fit(test_test_matrix)
terms = vectorizer.get_feature_names()

#split avec et sans analyse textuel
analyzed_train_features = hstack([train_text_matrix, train_num_features]).tocsr() #horizontal stack of matrix + sparse representation (compressed row format)
analyzed_test_features = hstack([test_test_matrix, test_num_features]).tocsr() 

terms = vectorizer.get_feature_names()
var_imp = pd.DataFrame(index=terms)
for label in lab:
    print('... Processing {}'.format(label))
    y = sample_train[label].values
    classifier = LogisticRegression()
    classifier.fit(train_text_matrix, y)
    y_pred_X = classifier.predict(test_test_matrix)
    print('Valid Accuracy is {}'.format(accuracy_score(y, y_pred_X)))
    var_imp[label] =  np.sqrt(scaler.var_) * classifier.coef_[0]
    
var_imp = var_imp.sort_values('toxic', ascending=False)

#avec
scaler.fit(analyzed_test_features)
var_imp2 = pd.DataFrame(index=terms + num_features)
for label in lab:
    print('... Processing {}'.format(label))
    y = sample_train[label].values
    classifier = LogisticRegression()
    classifier.fit(analyzed_train_features, y)
    y_pred_X = classifier.predict(analyzed_test_features)
    print('Valid Accuracy is {}'.format(accuracy_score(y, y_pred_X)))
    var_imp2[label] =  np.sqrt(scaler.var_) * classifier.coef_[0]
    
var_imp2.sort_values('toxic', ascending=False)