# DocSummarizer

LaMini-LM-Summarization-Streamlit-App.
This is a document summarization app powered by LLM (Language Model). The app allows users to upload PDF files and generate summaries of the document contents using the LLM model.


## Features

- Upload PDF files for summarization
- Automatic text splitting and preprocessing
- LLM-based summarization pipeline
- Display of uploaded PDF file and generated summary

## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/ruxiiii/DocSummarizery.git

2. Install Dependencies:

  pip install -r requirements.txt

3. Run the App:

  streamlit run app.py


## Usage

* Open the app in your web browser (by default, it runs on http://localhost:8501).
* Click on the "Upload PDF" button to select and upload a PDF file for summarization.
* Once the file is uploaded, click the "Summarize" button to generate a summary of the document 
  contents.
* The uploaded PDF file will be displayed on the left side, and the generated summary will be 
  shown on the right side of the app interface.


## Modifications

Modify the llm_pipeline function to make changes in generated summary length:

def llm_pipeline(filepath):
    pipe_sum = pipeline(
        'summarization',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 500, //change the summary length you want here
        min_length = 50
    )

