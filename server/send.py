import requests
import json

url = 'http://127.0.0.1:5000/predict'

input_data = {
    "text": "Мне не понравилось, как прошла предыдущая лекция"  
}
response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(input_data))

print(response.json())