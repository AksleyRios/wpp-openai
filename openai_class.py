import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd 

class ChatOpenAi:
    df = pd.read_csv('Data/embedded_dictionary.csv')
    def __init__(self):

        openai.organization = "org-zESGRZNdQTvSZkSx4LQN0J6N"
        openai.api_key = "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"
        

    def generate_text(self, prompt, max_tokens=50):
        try:
            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=prompt,
                max_tokens=max_tokens,
            )
            return response.choices[0].text
        except Exception as e:
            return f"Error: {str(e)}"
        
    def get_context(self,query):
        df_res = self.search_reviews(self.df, query)
        

    def search_reviews(self,df, query, n=1, pprint=True):
        embedding = get_embedding(query, model='text-embedding-ada-002')
        df['similarities'] = df.ada_embedding.apply(lambda x: cosine_similarity(x, embedding))
        res = df.sort_values('similarities', ascending=False).head(n)
        return res


