---
post_title: 'Document Summarization Solution Patterns using Azure Open AI & Langchain'
author1: Komal Grover
post_slug: solution-patterns-for-document-summarization-azureopenai
post_date: 2024-02-02 00:00:00
categories: Cognitive Services, GPT, LLM
tags: azureopenai,langchain
summary: This post lists the various solution patterns that can be applied for document summarization. Document summarization comes with its challenges related to token limitation and chunk sizes. This blog post discusses about the solutions to tackle those challenges.
---

# Introduction

In this blog post we will discuss about the various solutions patterns that can be experimented with to generate document summaries.
Document summarization using Azure Open AI provides the capability to generate concise summaries. <br>

**Use cases** 

 - In our recent engagement, the customer wanted an internal search system and had a requirement of being able to search on the summary of the document based on the user query. <br> 
 - Document summarization can also be used to extract key information which help in generating trends & highlights. <br>
 - To extract relevant information and build tools for question answering or decision support.  <br>

There can be various challenges associated with document summarization which this blog post aims to solve.

## Challenges
1. Input text size and token limitations
2. Summarization of summaries can lead to loss of information
3. Indexing the document summary for each chunk will lead to redundant information in the chunks <br>

## Initial Setup
The below solution patterns are using RBAC access for accessing Azure Open AI resource. To access the AzureOpenAI resource you need to provide `Cognitive Services OpenAI User role` to the `Identity` used for retrieving credentials. When using RBAC access `openai.api_type` is `azure_ad`. <br>
If you don't have the permission rights for assigning the RBAC access, replace the value for `os.environ["OPENAI_API_KEY"]` with the `Azure Open AI API Key`, which can be extracted from Azure portal for the Azure OpenAI resource in the `Keys & Endpoints` section and also replace the value of `os.environ["OPENAI_API_TYPE"]` and `openai_api_type` to `azure` instead of `azure_ad`.

``` python
def __init__(self):
    self.openai_deploymentname = 'DeploymentName'
    self.azure_endpoint = f'https://{self.openai_deploymentname}.openai.azure.com/openai'
    self.credential = DefaultAzureCredential()
    
    os.environ["OPENAI_API_TYPE"] = "azure_ad"
    os.environ["OPENAI_API_VERSION"] = "2023-08-01-preview"
    os.environ["OPENAI_API_BASE"] = self.azure_endpoint 
    os.environ["OPENAI_API_KEY"] = self.credential.get_token("https://cognitiveservices.azure.com/.default").token

    openai.api_type = "azure_ad"
    openai.api_base = self.azure_endpoint 
    openai.api_version = "2023-08-01-preview"
    openai.api_key = self.credential.get_token("https://cognitiveservices.azure.com/.default").token

```

## Solutions
Note - The patterns 2,3,4 below use the text for the book 'The Adventures of Huckleberry Finn' and summarizes using different approaches. <br>

### Pattern 1 - Simple Chunk based mechanism
Using a simple chunk based mechanism where summary of summaries is generated. Using this approach we are chunking only based on document size otherwise the whole text is being sent in a single prompt.

***Import libraries***
```python
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
```

The below code is splitting the text first based on the tokens and then calls the `_summarize_text_chunk` method which calls Azure OpenAI API

``` python
 def summarize_text(self, text: str) -> str:
    """
    Public method to summarize text using the Azure OpenAI service.
    """
    count_tokens = self.num_tokens_from_string_text(text)
    print(count_tokens)
    if count_tokens > 1000:
        output_chunks = []
        chunked_text = self.split_string_with_limit(text, 1000)
        for chunk in chunked_text:
            summary = self._summarize_text_chunk(chunk)
            output_chunks.append(summary)
        summary = self._summarize_text_chunk(str(output_chunks))
    else:
        summary = self._summarize_text_chunk(text)
    
    return summary
```
Below code calls the Azure OpenAI `create` api and returns the summarized response.

``` python
def _summarize_text_chunk(self, text: str) -> str:
    """
    Private method to summarize text using the Azure OpenAI service.
    """
    response = self.client.chat.completions.create(model='gpt-35-turbo',
            messages=[{'role': 'user', 'content': f'Summarize this text: {text}'}],
            temperature=0.5,
            max_tokens=2000,
            n=1,
            stop=None)
    return response.choices[0].message.content.strip()
```
Pros:
- Can be used in cases where a lot of documents will be summarized but majority of those documents are small in size. <br>

Cons:
- Uses a sequential mechanism, hence more time taking.

A full code example of this pattern can be found [here.](https://github.com/komalg1/document-summarization/blob/main/summarizer.py)

### Pattern 2 - Using MapReduce 
This pattern is also chunk based mechanism but the chunk processing is done in parallel instead of sequential using langchain MapReduce. Using `load_summarize_chain` method from langchain framework the document can be summarized.

***Import libraries***
```python
import os
from azure.identity import DefaultAzureCredential
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import TextLoader
import openai
import textwrap
```
Below code snippet loads the document & returns in the `Document` format which is required by langchain map reduce method.

``` python
def load_document(self):
    cwd = os.getcwd()
    loader = TextLoader(f'{cwd}/the-adventures-of-huckleberry-finn.txt')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    return docs[:6]
```

Code to call the `load_summarize_chain` method
``` python
def summarize_mapreduce_summary_version1(self):
    llm = AzureChatOpenAI(openai_api_base=self.azure_endpoint,
            openai_api_version="2023-08-01-preview",
            deployment_name='gpt-35-turbo',
            openai_api_key=self.credential.get_token("https://cognitiveservices.azure.com/.default").token,
            openai_api_type = "azure_ad",
            max_tokens=2500)
    self.text = self.load_document()
    
    chain = load_summarize_chain(llm=llm, chain_type="map_reduce")
    output_summary = chain.run(self.text)
    wrapped_text = textwrap.fill(output_summary, width=100)
    print(wrapped_text)
```
Pros:
- Faster processing time as chunks are processed in parallel.

Cons:
- Loss of information may be possible as the chunks are processed parallely without any connection with previous chunks. <br>

A full code example of this pattern can be found [here.](https://github.com/komalg1/document-summarization/blob/main/summarizer_mapreduce.py)


### Pattern 3 - Map Reduce chain <br>
In this approach first each document is mapped to an individual summary using an LLMChain and then it uses a ReduceDocumentChain to combine the chunk summaries to a common summary. Here we can reuse the chain to combine and then collapse the chain. In case the max_tokens exceeds a given number then it recursively passes the chunks in batches of tokens less than the token-max to StuffDocumentsChain to create chunk summaries. In the end, the batched summaries are then passed to StuffDocumentChain to create one cumulative summary.

***Import libraries***
```python
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
```

Following code snippet demonstrates the map reduce chain.

``` python
def summarize_mapreduce_summary_version1(self):
    # Map
    llm = AzureChatOpenAI(openai_api_base=self.azure_endpoint,
        openai_api_version="2023-08-01-preview",
        deployment_name='gpt-35-turbo',
        openai_api_key=self.credential.get_token("https://cognitiveservices.azure.com/.default").token,
        openai_api_type = "azure_ad")
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
```
Pros:
- It can be scaled for large documents and the processing on chunks is done in parallel.

Cons:
- It requires more calls to the LLM.

A full code example of this pattern can be found [here.](https://github.com/komalg1/document-summarization/blob/main/summarizer_mapreducechain.py)


### Pattern 4 - Refine <br>
In this approach an initial prompt on the first chunk of data is sent to generate the output. For the next document, the previous output along with the document is passed in and LLM is asked to refine the output based on the current document. This approach prevents loss of data that may happen in the Map Reduce approach. Also, the calls are sequential and not independent.

***Import libraries***
```python
import os
from azure.identity import DefaultAzureCredential
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
import openai
import textwrap
```

``` python
def summary_refine(self):
    docs = self.load_document()
    prompt_template = """Write a concise summary of the following extracting the key information:

    {text}

    CONCISE SUMMARY:"""
    PROMPT = PromptTemplate(template=prompt_template, 
                            input_variables=["text"])

    refine_template = (
        "Your responsibility is to craft a final summary. We've presented an existing summary up to a designated point:  {existing_answer}\n"
        "There's an opportunity to refine this summary (if required) with additional context provided below. \n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Considering the new context, revise the original summary. If the context isn't pertinent, maintain the original summary."
    )
    refine_prompt = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )
    chain = load_summarize_chain(AzureChatOpenAI(openai_api_base=self.azure_endpoint,
            openai_api_version="2023-08-01-preview",
            deployment_name='gpt-35-turbo',
            openai_api_key=self.credential.get_token("https://cognitiveservices.azure.com/.default").token,
            openai_api_type = "azure_ad"), 
                                chain_type="refine", 
                                return_intermediate_steps=True, 
                                question_prompt=PROMPT, 
                                refine_prompt=refine_prompt)
    output_summary = chain({"input_documents": docs}, return_only_outputs=True)
    wrapped_text = textwrap.fill(output_summary['output_text'], 
                            width=100,
                            break_long_words=False,
                            replace_whitespace=False)
    print(wrapped_text)
```
Pros:
- Continuity of the context between documents is maintained as documents are summarized sequentially.

Cons:
- Multiple calls to LLM. This might cause an issue when using the Free plan as there is a limit for sending requests per minute.

A full code example of this pattern can be found [here.](https://github.com/komalg1/document-summarization/blob/main/summarizer_refine.py)

## Conclusion
As organizations increasingly seek efficient ways to manage and retrieve information from large documents, the presented solution patterns serve as valuable tools in enhancing search systems and meeting diverse summarization requirements. Whether it's chunking large texts, leveraging parallel processing, or refining summaries based on context, the discussed patterns offer a comprehensive toolkit for tackling document summarization challenges.