#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#get_ipython().system('pip install datasets nltk transformers torch PyTorch scipy tabulate')

#clear_output()


# In[1]:


from IPython.display import clear_output
import numpy as np
import pandas as pd
from tabulate import tabulate
import nltk
from nltk.tokenize import RegexpTokenizer
import warnings
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
import torch
from scipy.spatial.distance import cdist

# Suppress all warnings in cells printouts for clear output
warnings.filterwarnings('ignore')


# # Load dataset

# In[2]:


dataset = load_dataset("medical_questions_pairs")

for split in dataset:
    print(split)



# # Parse and clean dataset
# 

# In[3]:


df = pd.DataFrame(dataset['train'])
df.head()


# In[4]:


question_1 = 'question_1'
question_2 = 'question_2' # column names as vars for convenienct


# In[5]:


df.drop('dr_id', axis=1, inplace=True)  # keep only question pairs and labels. Label 1 means match


# In[6]:


df.shape


# Properties of a good set:
# 1. No paired duplicates (no rows with same q1 and q2 values as a pair). Ensure removal of inverse duplication:
#    q1,q2 and q2,q1
# 2. Its ok to have duplicates in question_1 as they can have multiple matches
# 3. No NaN
# 4. Many to many relation is possible

# In[7]:


# Check for NaN and remove
nan_cols = df.isnull().any()
print(nan_cols)


# In[8]:


# Remove paired duplicates 
df['isometric_pair'] = df.apply(lambda x: tuple(sorted([x[question_1], x[question_2]])), axis=1)

# Remove duplicates based on the normalized pairs
df = df.drop_duplicates(subset=['isometric_pair'])

# Drop the auxiliary column
df = df.drop(columns=['isometric_pair'])


# In[9]:


df.shape  # see if something was removed


# Outcome: clean dataset without pairs of duplicates

# In[10]:


# TEST LABLES CORRECTNESS

matches_and_labels = df[df[question_1] == df.loc[100, question_1]]
print('question: ', df.loc[3, question_1])
print(matches_and_labels)


# # Embedding: transform each question into vector
# 
#     1. Tokenize questions to words
#     2. Embed words to vectors with BERT
#     3. Make an average vectors to represent the sentences (or weighted average: like TF-IDF scores, giving more importance to certain words.)    
#     4. Collect vectorized Q1 and Q2 questions sets into two dataframes
#     5. Precalculate table of distances for each q1 to each q2
#     6. For specific q1, get its distances vector from all Q2 and select 5 closest
#     7. Based on indices, return the original Q2 questions
# 

# In[11]:


def tokenize(question):
    """ Break question sentence to word tokens without punctuation """
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(question) # split
    tokens = [token.lower() for token in tokens] # decapitalize
    
    return tokens


# In[12]:


def embed(tokens):
    """ Embed tokenized sentence to one vector """
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Convert custom tokens into a string for BERT special format
    custom_text = ' '.join(tokens)
    
    # Tokenize this text using BERT's tokenizer
    bert_inputs = bert_tokenizer(custom_text, return_tensors="pt", padding=True, truncation=True)
    
    # Pass the tokenized inputs through the BERT model
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    with torch.no_grad():
        bert_outputs = bert_model(**bert_inputs)
    
    # Extracting embeddings
    embeddings = bert_outputs.last_hidden_state
    embedded_question = embeddings.mean(dim=1)[0]  # single vector representing the entire question.
    return embedded_question.detach().numpy()
    


# In[13]:


def calculate_distances_matrix(v1, v2):
   
    """ Calculate cosine distance between a vector v1 and each vector in df_v2 """
    return cdist(vectors_Q1, vectors_Q2, 'cosine')

    #if the cosine similarity is high (vectors are similar), the cosine distance will be low, and vice versa.
    


# In[14]:


def tokenize_embed(x):
    return embed(tokenize(x))


# In[139]:


def return_x_matches(df, question_index, table_of_distances, x):
    # sort KEEPING the original index
    list_of_distances = table_of_distances.loc[question_index].sort_values(ignore_index=False).head(x)
    indices = list_of_distances.index    
    #print(list_of_distances)
    question_str = df.loc[question_index][question_1]
    print('ORIGINAL DATAFRAME:')
    print(df.loc[question_index])
    print('distances for that question:')
    print(list_of_distances)
    matches = df.loc[indices]
    print('matches for that question with all labels:')
    print(matches)
    matches = matches[(matches['label'] == 1) & (df[question_1] == question_str)]
   # print(question_str)
    #print('INDEX BASED:')
    #print(matches)
    return matches[question_2].to_list()[:x]  


# In[16]:


pd.set_option('display.max_colwidth', None)


# In[132]:


# Test out on a slice of the original dataframe
SLICE_SIZE_q1 = 10
SLICE_SIZE_q2 = 50
# Since question 1 repeats, we cannot use the same slice size for question 2, it does not take all of the matches

embedded_Q1_df=pd.DataFrame(df.head(SLICE_SIZE_q1)[question_1].apply(lambda x: tokenize_embed(str(x))))
embedded_Q2_df=pd.DataFrame(df.head(SLICE_SIZE_q2)[question_2].apply(lambda x: tokenize_embed(str(x))))


vectors_Q1 = np.stack(embedded_Q1_df.iloc[:, 0].values)
vectors_Q2 = np.stack(embedded_Q2_df.iloc[:, 0].values)

# DataFrame of 1D DataFrames : each entry is a vector of distances of Q1_i question from Q2 questions

distances_matrix = calculate_distances_matrix(vectors_Q1, vectors_Q2)

distances_Q1_Q2_df = pd.DataFrame(distances_matrix, index = embedded_Q1_df.index, columns = embedded_Q2_df.index)

distances_Q1_Q2_df.head(SLICE_SIZE_q1)



# In[114]:


def print_results(question_index):
    question1 = df.loc[question_index, question_1]
    matches = return_x_matches(df, question_index, distances_Q1_Q2_df, 5)
    print(f'Matches for "{question1}" are: \n')
    print('\n'.join(matches))
    return '\n'.join(matches)


# In[140]:


print_results(0)  #very strange for index 0 and 1 I do not get the same question_1 in indices-based.


# In[ ]:




