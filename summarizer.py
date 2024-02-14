from typing import List
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.core.credentials import AzureKeyCredential
import tiktoken
from langchain.document_loaders import TextLoader
import os

class Summarizer:
    def __init__(self):
        self.api_version = '2023-08-01-preview'
        self.openai_deploymentname = 'DEPLOYMENT_NAME'
        self.azure_endpoint = f'https://{self.openai_deploymentname}.openai.azure.com/'
        self.credential = DefaultAzureCredential()
        self.client = self._get_client()

    def _get_client(self) -> AzureOpenAI:
        """
        Private method to retrieve and return the AzureOpenAI client.
        """
        token_provider = get_bearer_token_provider(self.credential, "https://cognitiveservices.azure.com/.default")
        return AzureOpenAI(
            azure_ad_token_provider=token_provider,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint
        )
    
    def num_tokens_from_string_text(self, string_text: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = len(encoding.encode(string_text))
        return num_tokens
    
    def split_string_with_limit(self, text: str, limit: int) -> List[str]:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        parts = []
        current_part = []
        current_count = 0

        for token in tokens:
            current_part.append(token)
            current_count += 1

            if current_count >= limit:
                parts.append(current_part)
                current_part = []
                current_count = 0

        if current_part:
            parts.append(current_part)

        text_parts = [encoding.decode(part) for part in parts]

        return text_parts
    
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
if __name__ == '__main__':
    summarizer = Summarizer()
    cwd = os.getcwd()
    text = TextLoader(f'{cwd}/text_document.txt')
    document = text.load()[0]
    
    summary = summarizer.summarize_text(document.page_content)
    print(summary)
