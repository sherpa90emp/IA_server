import requests

url = 
response = request.get(url)
data = response.json()

print(data['current_weather'])