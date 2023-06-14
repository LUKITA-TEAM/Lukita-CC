import requests

resp = requests.post("https://lukita-model-vrer4llm6a-et.a.run.app/", files={'file': open('test/thu1.jpg', 'rb')})

print(resp.json())