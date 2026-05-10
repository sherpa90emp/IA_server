import requests

url = "open-meteo.com"
response = requests.get(url)
data = response.json()

print(data['current_weather'])