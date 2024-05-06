#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#get_ipython().system('pip install datasets nltk transformers torch PyTorch scipy tabulate')

#clear_output()


# In[1]:


from IPython.display import clear_output
import numpy as np
import os
import pandas as pd
from tabulate import tabulate
import psycopg
import nltk
from nltk.tokenize import RegexpTokenizer
import warnings
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
import torch
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from sqlalchemy import create_engine, Table, Column, BigInteger, String, MetaData, UniqueConstraint, inspect
from sqlalchemy.dialects.postgresql import BYTEA
import pickle


load_dotenv()

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


def tokenize_embed(x):
    return embed(tokenize(x))


# In[38]:


def return_x_matches_labels(df, question_index, table_of_distances, x):
    list_of_distances = table_of_distances.loc[question_index].sort_values(ignore_index=False, ascending=False) # max similarity
    indices = list_of_distances.index    
    matches = df.loc[indices][question_2].head(x)
    matches_labels = {}
    matches_list = matches.to_list()
    matches_labels['matches'] = '\n'.join(matches_list)
    matches_labels['labels'] = df.loc[indices]['label'].head(x)
    return matches_labels


# In[15]:


pd.set_option('display.max_colwidth', None)


# In[22]:


def cosine_similarity_df(df1, df2):
    cosine_sim = cosine_similarity(df1, df2)
    cosine_sim_df = pd.DataFrame(cosine_sim, index=df1.index, columns=df2.index)
    return cosine_sim_df


# In[49]:


# Test out on a slice of the original dataframe
SLICE_SIZE_q1 = 10


# In[50]:


# pre-embed target questions for both scenarios: looking for existing question and for a completely new
SLICE_SIZE_q2 = 100
embedded_Q2_df = pd.DataFrame(df.head(SLICE_SIZE_q2)[question_2].apply(tokenize_embed).tolist(), index=df.head(SLICE_SIZE_q2).index)


# In[51]:


# Since question 1 repeats, we cannot use the same slice size for question 2, it does not take all of the matches

embedded_Q1_df = pd.DataFrame(df.head(SLICE_SIZE_q1)[question_1].apply(tokenize_embed).tolist(), index=df.head(SLICE_SIZE_q1).index)

distances_q1_q2 = cosine_similarity_df(embedded_Q1_df, embedded_Q2_df)


# In[52]:



TABLE_NAME = 'embedded_medical_questions'

def get_connection_string():
    user = os.getenv('DB_DESTINATION_USER')
    password = os.getenv('DB_DESTINATION_PASSWORD')
    host = os.getenv('DB_DESTINATION_HOST')
    port = os.getenv('DB_DESTINATION_PORT')
    dbname = os.getenv('DB_DESTINATION_NAME')
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


def read_data(engine):
    query = f"SELECT question_id, question, embedding FROM {TABLE_NAME}"
    df = pd.read_sql(query, con=engine)
    # Deserialize the 'embedding' column directly after loading it
    # This step converts the binary data (stored in BYTEA) back into numpy arrays
    df['embedding'] = df['embedding'].apply(lambda x: pickle.loads(x))
    return df

def create_or_check_table(engine):
    metadata = MetaData(engine)
    table = Table(TABLE_NAME, metadata,
                  Column('question_id', BigInteger, primary_key=True),
                  Column('question', String),
                  Column('embedding', BYTEA),
                  UniqueConstraint('question_id', name=TABLE_NAME))
    if not inspect(engine).has_table(TABLE_NAME):
        metadata.create_all(engine)
        return False
    return True

def write_data(df, engine):
    df.to_sql(TABLE_NAME, con=engine, if_exists='replace', index_label='question_id', dtype={'embedding': BYTEA})


engine = create_engine(get_connection_string())
db_exists = create_or_check_table(engine)



if not db_exists:
    df = pd.DataFrame({
        'question': df.head(SLICE_SIZE_q2)[question_2],
        'embedding': embedded_Q2_df.apply(lambda x: pickle.dumps(x))
    }, index=df.head(SLICE_SIZE_q2).index)
    write_data(df, engine)
else:
    df_embedded = read_data(engine)
    #print(df)



def return_x_matches_labels_database(df, question_index, table_of_distances, x):
    list_of_distances = table_of_distances.loc[question_index].sort_values(ignore_index=False, ascending=False) # max similarity
    indices = list_of_distances.index
    matches = df.loc[indices]['question'].head(x)
    matches_labels = {}
    matches_list = matches.to_list()
    matches_labels['matches'] = '\n'.join(matches_list)
    return matches_labels

# WORK WITH THE FIXED QUESTION 
SLICE_SIZE_FIXED = 1

df_fixed=df.head(1).copy()

def return_to_streamlit(question):
    df_fixed[question_1] = "How to treat my anxiety?"
    embedded_Q1_df_fixed = pd.DataFrame(df_fixed.head(SLICE_SIZE_FIXED)[question_1].apply(tokenize_embed).tolist(), index=df_fixed.head(SLICE_SIZE_FIXED).index)

    distances_q1_q2_fixed = cosine_similarity_df(embedded_Q1_df_fixed, df_embedded['embedding'])
    question_index_fixed = 0
   # question1 = df_fixed.loc[question_index_fixed, question_1]
   # print("Your question:", question1)
    matches_labels = return_x_matches_labels_database(df_embedded, question_index_fixed, distances_q1_q2_fixed, 5) # search in original df for matches
  #  print("Similar questions: ", matches_labels['matches'])
    return f"Similar questions: {matches_labels['matches']}"



# In[29]:





# In[42]:


question_index = 9
question1 = df.loc[question_index, question_1]
print(question1)
matches_labels = return_x_matches_labels(df, question_index, distances_q1_q2, 5)
print(matches_labels['matches'])
print(matches_labels['labels'])
accuracy_at_5 = matches_labels['labels'].sum() / len(matches_labels['labels'])
print('Accuracy@5: ', accuracy_at_5)


# In[ ]:




