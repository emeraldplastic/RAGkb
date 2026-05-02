import requests

PASSWORD = "Password123!"


def main():
    login_res = requests.post(
        "http://127.0.0.1:8000/api/register",
        json={"username": "testuser_chat", "password": PASSWORD},
        timeout=10,
    )
    if login_res.status_code == 409:
        login_res = requests.post(
            "http://127.0.0.1:8000/api/login",
            json={"username": "testuser_chat", "password": PASSWORD},
            timeout=10,
        )
    token = login_res.json()["token"]

    print("Token:", token)
    headers = {"Authorization": f"Bearer {token}"}
    print("Calling chat endpoint...")
    try:
        res = requests.post(
            "http://127.0.0.1:8000/api/chat",
            json={"question": "what is the context of synopsis 1?"},
            headers=headers,
            timeout=10,
        )
        print("Status:", res.status_code)
        print("Body:", res.text)
    except Exception as exc:
        print("Error:", exc)


if __name__ == "__main__":
    main()
