import os
from azure.identity import DefaultAzureCredential
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import TextLoader
import openai
import textwrap

class Summarizer:
    def __init__(self):
        self.openai_deploymentname = 'DEPLOYMENT_NAME'
        self.azure_endpoint = f'https://{self.openai_deploymentname}.openai.azure.com/'
        self.credential = DefaultAzureCredential()
        
        openai.api_type = os.environ["OPENAI_API_TYPE"] = "azure_ad"
        openai.api_base = os.environ["OPENAI_API_BASE"] = self.azure_endpoint 
        openai.api_version = os.environ["OPENAI_API_VERSION"] = "2023-08-01-preview"
        openai.api_key = os.environ["OPENAI_API_KEY"] = self.credential.get_token("https://cognitiveservices.azure.com/.default").token
            
    def load_document(self):
        cwd = os.getcwd()
        loader = TextLoader(f'{cwd}/text_document.txt')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        return docs
        
    def summarize_mapreduce_summary_version1(self):
        
        llm = AzureChatOpenAI(openai_api_base=self.azure_endpoint,
                openai_api_version="2023-08-01-preview",
                deployment_name='gpt-35-turbo',
                openai_api_key=self.credential.get_token("https://cognitiveservices.azure.com/.default").token,
                openai_api_type = "azure_ad",
                max_tokens=1800)
        self.text = self.load_document()
        
        chain = load_summarize_chain(llm=llm, chain_type="map_reduce")
        output_summary = chain.run(self.text)
        wrapped_text = textwrap.fill(output_summary, width=100)
        print(wrapped_text)
        
if __name__ == '__main__':
    summarizer = Summarizer()
    summary = summarizer.summarize_mapreduce_summary_version1()