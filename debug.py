import os
import sys
import importlib
from unittest.mock import patch
import uuid

# Mock environment variables
os.environ['DATABASE_PATH'] = f'debug_{uuid.uuid4().hex}.db'
os.environ['CHROMA_PATH'] = 'debug_chroma'
os.environ['UPLOAD_DIR'] = 'debug_uploads'
os.environ['SECRET_KEY'] = 'test-secret-key-for-testing-only'

class FakeEmb:
    def __init__(self, **kwargs):
        pass
    def embed_documents(self, texts):
        return [[0.1]*384 for _ in texts]
    def embed_query(self, text):
        return [0.1]*384

import config
importlib.reload(config)
import database
importlib.reload(database)
import auth
importlib.reload(auth)

with patch('langchain_community.embeddings.HuggingFaceEmbeddings', FakeEmb):
    import main
    importlib.reload(main)
    main.embeddings = FakeEmb()
    
    from starlette.testclient import TestClient
    client = TestClient(main.app)

    # Use valid username
    res = client.post('/api/register', json={'username':'validuser','password':'password123'})
    print('register status:', res.status_code)
    try:
        data = res.json()
        token = data.get('token')
        
        res2 = client.get('/api/me', headers={'Authorization': f'Bearer {token}'})
        print('me status:', res2.status_code)
        if res2.status_code != 200:
            from jose import jwt
            print("\n--- Decoding with test key ---")
            try:
                payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
                print("Decoded with test key:", payload)
            except Exception as e:
                print("Failed:", e)
                
            print("\n--- Decoding with default key ---")
            try:
                # the default secret key from config.py
                default_key = "change-me-in-production-use-a-long-random-string"
                payload = jwt.decode(token, default_key, algorithms=[config.ALGORITHM])
                print("Decoded with default key:", payload)
            except Exception as e:
                print("Failed:", e)
    except Exception as e:
        print("Error during test:", e)
