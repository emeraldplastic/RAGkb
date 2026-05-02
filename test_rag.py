import requests
import time

def run_test():
    # login
    password = 'Password123!'
    login = requests.post('http://127.0.0.1:8000/api/register', json={'username': 'testuser12', 'password': password})
    if login.status_code == 409:
        login = requests.post('http://127.0.0.1:8000/api/login', json={'username': 'testuser12', 'password': password})
    
    t = login.json()['token']
    headers = {'Authorization': 'Bearer ' + t}
    print("Logged in")

    # upload
    with open('synopsis1.pdf', 'rb') as f:
        res = requests.post('http://127.0.0.1:8000/api/upload', files={'file': f}, headers=headers)
        print('Upload:', res.json())
        
    # chat
    print("Sending chat request...")
    start = time.time()
    chat = requests.post('http://127.0.0.1:8000/api/chat', json={'question': 'what is the context of synopsis1 ?'}, headers=headers)
    print(f"Chat status: {chat.status_code}")
    print(f"Chat response in {time.time()-start} seconds:", chat.text)

if __name__ == '__main__':
    run_test()
