from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
import os
from langchain.document_loaders import TextLoader
import openai
from azure.identity import DefaultAzureCredential
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

class Summarizer:
        def __init__(self):
        #self.api_version = '2023-08-01-preview'
            self.openai_deploymentname = 'LearningAzureAI'
            self.azure_endpoint = f'https://{self.openai_deploymentname}.openai.azure.com/openai'
            self.credential = DefaultAzureCredential()
            
            os.environ["OPENAI_API_TYPE"] = "azure"
            os.environ["OPENAI_API_VERSION"] = "2023-08-01-preview"
            os.environ["OPENAI_API_BASE"] = self.azure_endpoint 
            os.environ["OPENAI_API_KEY"] = self.credential.get_token("https://cognitiveservices.azure.com/.default").token

            openai.api_type = "azure"
            openai.api_base = self.azure_endpoint 
            openai.api_version = "2023-08-01-preview"
            openai.api_key = self.credential.get_token("https://cognitiveservices.azure.com/.default").token

        def load_document(self):
            cwd = os.getcwd()
            loader = TextLoader(f'{cwd}/how_to_win.txt')
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs = text_splitter.split_documents(documents)
            return docs[:20]

        def summarize_mapreduce_summary_version1(self):
            # Map
            llm = AzureChatOpenAI(openai_api_base=self.azure_endpoint,
                openai_api_version="2023-08-01-preview",
                deployment_name='gpt-35-turbo',
                openai_api_key=self.credential.get_token("https://cognitiveservices.azure.com/.default").token,
                openai_api_type = "azure")
            docs = self.load_document()
            # Map
            map_template = """The following is a set of documents
            {docs}
            Based on this list of docs, please identify the main themes 
            Helpful Answer:"""
            map_prompt = PromptTemplate.from_template(map_template)
            map_chain = LLMChain(llm=llm, prompt=map_prompt)
            # Reduce
            reduce_template = """The following is set of summaries:
            {docs}
            Take these and distill it into a final, consolidated summary of the main themes. 
            Helpful Answer:"""
            reduce_prompt = PromptTemplate.from_template(reduce_template)
            # Run chain
            reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

            # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
            combine_documents_chain = StuffDocumentsChain(
                llm_chain=reduce_chain, document_variable_name="docs"
            )

            # Combines and iteravely reduces the mapped documents
            reduce_documents_chain = ReduceDocumentsChain(
                # This is final chain that is called.
                combine_documents_chain=combine_documents_chain,
                # If documents exceed context for `StuffDocumentsChain`
                collapse_documents_chain=combine_documents_chain,
                # The maximum number of tokens to group documents into.
                token_max=4000,
            )
            # Combining documents by mapping a chain over them, then combining results
            map_reduce_chain = MapReduceDocumentsChain(
                # Map chain
                llm_chain=map_chain,
                # Reduce chain
                reduce_documents_chain=reduce_documents_chain,
                # The variable name in the llm_chain to put the documents in
                document_variable_name="docs",
                # Return the results of the map steps in the output
                return_intermediate_steps=False,
            )

            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=1000, chunk_overlap=0
            )
            split_docs = text_splitter.split_documents(docs)
            print(map_reduce_chain.run(split_docs))

if __name__ == '__main__':
    summarizer = Summarizer()
    summary = summarizer.summarize_mapreduce_summary_version1()