# jupyter nbconvert --to script [YOUR_NOTEBOOK].ipynb


from fastapi import FastAPI
import sys
sys.path.insert(0, '../')  # Adjust the path as necessary

from st_nlp_medical_qa import print_results  # Import the desired function

app = FastAPI()

@app.post('/api/post_question')
def post_question(question: int):
    return print_results(question)
