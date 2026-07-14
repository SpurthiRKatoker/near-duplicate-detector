from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

q=pd.read_csv("sample_question.csv")["question"].dropna().tolist()
model=SentenceTransformer('all-MiniLM-L6-v2')
emb=model.encode(q)
print(type(emb))
try:
    import numpy as np
    print('shape', getattr(emb, 'shape', None))
    arr = np.array(emb)
    print('arr shape', arr.shape, 'dtype', arr.dtype)
    s = cosine_similarity(arr)
    print('max sim', s.max(), 'min sim', s.min())
    print('row0 top5', sorted(s[0], reverse=True)[:5])
except Exception as e:
    print('error', e)
