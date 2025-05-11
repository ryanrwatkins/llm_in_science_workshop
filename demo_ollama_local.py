import requests
import json

# Send request to Ollama
res = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "deepseek-r1:7b", 
        "prompt": "Explain what p-hacking is in research in 100 words.",
        "stream": False
    }
)

print(res.text)

# Convert to JSON
#data = res.json()
#print(data["response"])


