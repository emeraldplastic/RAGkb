import ollama

def chat(message):
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": message}]
    )
    return response["message"]["content"]

print("Local AI ready. Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    reply = chat(user_input)
    print(f"AI: {reply}\n")