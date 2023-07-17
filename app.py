import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64
import accelerate


#Load the model and tokenizer
checkpoint = 'LaMini-Flan-T5-248M'
tokenizer = T5Tokenizer.from_pretrained(checkpoint)

base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map = 'auto',  torch_dtype = torch.float32, offload_folder = 'C:/Users/aarus/Documents/Code/NLP/offload')


def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 50)

    texts = text_splitter.split_documents(pages)
    final_texts = ''

    for text in texts:
        print(text)
        final_texts = final_texts + text.page_content
    return final_texts


#LLM Pipeline
def llm_pipeline(filepath):
    pipe_sum = pipeline(
        'summarization',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 500,
        min_length = 50
    )

    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    return result


#cache data and cache resource, resource for numpy and all
@st.cache_data
#function to display the give pdf file


def displayPDF(file):
    #Opening the file from the file path
    with open(file,'rb') as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

        # embedding pdf in html
        pdf_display = F'<iframe src = "data:application/pdf;base64,{base64_pdf}" width = "100%" height = "600" type = "application/pdf"></iframe>'


        #displaying the file
        st.markdown(pdf_display, unsafe_allow_html=True)



#streamlit code
st.set_page_config(layout = 'wide', page_title='Document Summarizer')

def main():

    st.title('Document Summarization App using LLM')
    
    uploaded_file =  st.file_uploader('Upload PDF', type = ['pdf'])

    if uploaded_file is not None:
        if st.button('Summarize'):
            col1, col2 = st.columns(2)

            filePath = 'data/' + uploaded_file.name
            with open(filePath, 'wb') as f:
                f.write(uploaded_file.read())

            with col1:
                st.info('Uploaded PDF File')
                #viewing the PDF
                pdf_view = displayPDF(filePath)


            with col2:
                st.info('Summarization below')

                summary = llm_pipeline(filePath)
                st.success(summary)

if __name__ == '__main__':
    main()
