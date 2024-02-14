**Document Summarization using Azure Open AI**

Document summarization with Azure Open AI enables to distil concise, insightful summaries. Summarizing large documents can get challenging due to token limitation but using a combination of Azure Open AI with langchain this can be tackled. In this part we will discuss about the first approach that can be taken to have the summary of a document generated and get the system up and running.

**NOTE** - This repo is using RBAC access for accessing Azure Open AI resource. To access to OpenAI resource you need to provide `Cognitive Services OpenAI User` role to the Identity used for retrieving credentials.
**Challenges in document summarization** : <br><br>
	1. Input text size and token limitations <br>
	2. Summarization of summaries can lead to loss of information <br>
	3. Indexing the document summary for each chunk will lead to redundant information in the chunks <br>

**Solutions**: <br><br>
**Pattern 1** - Using a simple chunk based mechanism where summary of summaries is generated. Using this approach we are chunking only based on document size otherwise the whole text is being sent in a single prompt. <br>
Code - https://github.com/komalg1/document-summarization/blob/main/summarizer.py

**Pattern 2** - Using chunk based mechanism and processing chunks in parallel instead of sequential. This pattern uses langchain MapReduce. Using load_summarize_chain method, the document can be summarized. <br>
Code - https://github.com/komalg1/document-summarization/blob/main/summarizer_mapreduce.py

**Pattern3** - Using Map Reduce chain. In this approach first each document is mapped to an individual summary using an LLMChain and then it uses a ReduceDocumentChain to combine the chunk summaries to a common summary. In this approach we can reuse the chain to combine and then collapse the chain. In case the max_tokens exceeds a given number then it recursively passes the chunks in batches of tokens less than the `token-max` to StuffDocumentsChain to create chunk summaries. In the end the batched summaries are then passed to StuffDocumentChain to create one cumulative summary. <br>
Code - https://github.com/komalg1/document-summarization/blob/main/summarizer_mapreducechain.py

**Pattern 4** - Refine. In this approach an intial prompt on the first chunk of data is sent and generate output. For the nect document, the previous output along with the document is passed and LLM is asked to refine the ouput based on the current document. This approach prevents loss of data that may happen in Map Reduce approach. Secondly, the calls are sequential and not independent. <br>
Code - https://github.com/komalg1/document-summarization/blob/main/summarizer_refine.py