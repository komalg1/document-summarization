from typing import List
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.core.credentials import AzureKeyCredential
import tiktoken

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
    text = "The video titled Life in 2323 A.D. by Isaac Arthur provides an illustrative vision of daily life in the year 2323, exploring how technological advancements such as nanobots, life extension, cybernetic augmentation, orbital rings, safe artificial intelligence (AI), and highly automated and climate-controlled greenhouses could shape the lifestyle and civilization of people in the future.The video predicts a population in 2323 A.D. to be over 100 billion but under a trillion, supported primarily by resources supplied by highly automated and climate-controlled greenhouses. These greenhouses, coupled with technological advancements such as nanobots, life extension, and advanced automation, are expected to ensure sufficient resources for the growing population.Nanobots are predicted to become smaller, more durable, and cheaper, able to maintain equipment, homes, and roads, as well as potentially extend human life by managing aging and health conditions. Life extension technologies, coupled with the capabilities of nanobots, could lead to significantly extended lifespans. Cybernetic augmentation and mind uploading might further enhance human capabilities and potentially achieve immortality.AI is expected to be much smarter than current technology, leading to advancements in automation and sustainable energy. The Orbital Ring, a predicted structure around the Earth, will enable fast and cheap access to space for both people and cargo, revolutionizing space travel and resource extraction.The video also delves into the concept of a highly automated and climate-controlled greenhouse system that could provide food and resources for a population estimated to be over 100 billion but under a trillion. The video focuses on the daily life and lifestyle choices of individuals living in this future era. It narrates the lives of seven individuals: Amy, Becky, Cameron, Duncan, Emily, Fido, and Gary Googleson, each representing different aspects of this future society.Amy, a product of artificial DNA, dreams of marrying Steve, a man over 100 years old. They plan to live in the Suburban Enclave of Oskaloosa, a place with self-repairing roads and extensive walkways. Becky, Amys great-grandmother, lives in the Neo-Sears Tower Arcology in North Chicago and has raised multiple children with the help of cybernetic mental augmentation.Cameron, engaged to Beckys 17th daughter, was brought out of deep freeze after his mothers death. Duncan, on the other hand, is involved in funding and overseeing the construction of space habitats, particularly an asteroid colony called Metis. Emily is a crew member on the arkship Francis Baily, traveling to Lacaille 8760, accompanied by her intelligent dog, Fido.Finally, Gary, originally an AI program, had his brain transferred into an organic body and is interested in space colonization and the uplifting movement. The video, therefore, provides a comprehensive view of daily life in 2323, through the experiences and lifestyle choices of these individuals, all influenced by the technological advancements of their era.The video also discusses different life scenarios and civilizations in the year 2323 A.D. Steve and Amy live in a suburban enclave called Oskaloosa, where they desire a classic suburban life. They have a technologically advanced home with self-repairing features and live in a community governed by strict homeowner association rules.Amys great-grandmother, Becky, lives in the Neo-Sears Tower Arcology in North Chicago, a massive building that has expanded underground and is home to 10 million people. Becky and her husband have a large family and maintain a classical human lifestyle, despite advancements in technology. They also engage in cryogenic preservation and mind uploading experiments.Cameron and many others have strong feelings against transhumanism and opt for external devices, like augmented reality contact lenses, rather than implants. They prefer a more minimalistic and less technologically advanced lifestyle.Space habitats, privately funded or supported by governments, are being constructed in various locations, including Cis-Lunar space and the asteroid belt. These habitats range in size and population and serve different purposes, such as mining and farming. The video mentions that there are currently around 300,000 space habitats under construction, with a median size of 25 square kilometers and an intended population of 10,000 people each. Nearly a million people immigrate from Earth to these space habitats every year.Duncan, a character in the video, is involved in the funding and oversight of a mining and farming colony on the Metis asteroid. The colony aims to keep technology minimal and low profile, relying on robots for mining and selling food to neighboring asteroid mines and ships.New Athens is a planned refuge for those seeking uninterrupted contemplation. It is inhabited by post-biological philosophers and monks living a digital existence, as well as some biological administrators.The Francis Baily is a giant arkship en route to Lacaille 8760, a star system 13 light years from Earth. The ship has experienced technical issues with its nanotechnology, resulting in most of the crew and colonists being frozen. The descendants of the crew are now trying to run the ship, facing challenges due to their lack of institutional knowledge and unreliable technology.The video provides a comprehensive view of the potential challenges and solutions in the year 2323, offering a fascinating glimpse into a possible future shaped by technological advancement and societal adaptation.The video titled Life in 2323 A.D. by Isaac Arthur provides an illustrative vision of daily life in the year 2323, exploring how technological advancements such as nanobots, life extension, cybernetic augmentation, orbital rings, safe artificial intelligence (AI), and highly automated and climate-controlled greenhouses could shape the lifestyle and civilization of people in the future.The video predicts a population in 2323 A.D. to be over 100 billion but under a trillion, supported primarily by resources supplied by highly automated and climate-controlled greenhouses. These greenhouses, coupled with technological advancements such as nanobots, life extension, and advanced automation, are expected to ensure sufficient resources for the growing population.Nanobots are predicted to become smaller, more durable, and cheaper, able to maintain equipment, homes, and roads, as well as potentially extend human life by managing aging and health conditions. Life extension technologies, coupled with the capabilities of nanobots, could lead to significantly extended lifespans. Cybernetic augmentation and mind uploading might further enhance human capabilities and potentially achieve immortality.AI is expected to be much smarter than current technology, leading to advancements in automation and sustainable energy. The Orbital Ring, a predicted structure around the Earth, will enable fast and cheap access to space for both people and cargo, revolutionizing space travel and resource extraction.The video also delves into the concept of a highly automated and climate-controlled greenhouse system that could provide food and resources for a population estimated to be over 100 billion but under a trillion. The video focuses on the daily life and lifestyle choices of individuals living in this future era. It narrates the lives of seven individuals: Amy, Becky, Cameron, Duncan, Emily, Fido, and Gary Googleson, each representing different aspects of this future society.Amy, a product of artificial DNA, dreams of marrying Steve, a man over 100 years old. They plan to live in the Suburban Enclave of Oskaloosa, a place with self-repairing roads and extensive walkways. Becky, Amys great-grandmother, lives in the Neo-Sears Tower Arcology in North Chicago and has raised multiple children with the help of cybernetic mental augmentation.Cameron, engaged to Beckys 17th daughter, was brought out of deep freeze after his mothers death. Duncan, on the other hand, is involved in funding and overseeing the construction of space habitats, particularly an asteroid colony called Metis. Emily is a crew member on the arkship Francis Baily, traveling to Lacaille 8760, accompanied by her intelligent dog, Fido.Finally, Gary, originally an AI program, had his brain transferred into an organic body and is interested in space colonization and the uplifting movement. The video, therefore, provides a comprehensive view of daily life in 2323, through the experiences and lifestyle choices of these individuals, all influenced by the technological advancements of their era.The video also discusses different life scenarios and civilizations in the year 2323 A.D. Steve and Amy live in a suburban enclave called Oskaloosa, where they desire a classic suburban life. They have a technologically advanced home with self-repairing features and live in a community governed by strict homeowner association rules.Amys great-grandmother, Becky, lives in the Neo-Sears Tower Arcology in North Chicago, a massive building that has expanded underground and is home to 10 million people. Becky and her husband have a large family and maintain a classical human lifestyle, despite advancements in technology. They also engage in cryogenic preservation and mind uploading experiments.Cameron and many others have strong feelings against transhumanism and opt for external devices, like augmented reality contact lenses, rather than implants. They prefer a more minimalistic and less technologically advanced lifestyle.Space habitats, privately funded or supported by governments, are being constructed in various locations, including Cis-Lunar space and the asteroid belt. These habitats range in size and population and serve different purposes, such as mining and farming. The video mentions that there are currently around 300,000 space habitats under construction, with a median size of 25 square kilometers and an intended population of 10,000 people each. Nearly a million people immigrate from Earth to these space habitats every year.Duncan, a character in the video, is involved in the funding and oversight of a mining and farming colony on the Metis asteroid. The colony aims to keep technology minimal and low profile, relying on robots for mining and selling food to neighboring asteroid mines and ships.New Athens is a planned refuge for those seeking uninterrupted contemplation. It is inhabited by post-biological philosophers and monks living a digital existence, as well as some biological administrators.The Francis Baily is a giant arkship en route to Lacaille 8760, a star system 13 light years from Earth. The ship has experienced technical issues with its nanotechnology, resulting in most of the crew and colonists being frozen. The descendants of the crew are now trying to run the ship, facing challenges due to their lack of institutional knowledge and unreliable technology.The video provides a comprehensive view of the potential challenges and solutions in the year 2323, offering a fascinating glimpse into a possible future shaped by technological advancement and societal adaptation."
    summary = summarizer.summarize_text(text)
    print(summary)
