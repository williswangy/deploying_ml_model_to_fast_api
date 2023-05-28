import requests
import json

url = "https://fastapiproject3.herokuapp.com/inference"  # Update with your server's address and port
headers = {
    'Content-Type': 'application/json'
}

data = {
    'age': 50,
    'workclass': "Private",
    'fnlgt': 234721,
    'education': "Doctorate",
    'education_num': 16,
    'marital_status': "Separated",
    'occupation': "Exec-managerial",
    'relationship': "Not-in-family",
    'race': "Black",
    'sex': "Female",
    'capital_gain': 0,
    'capital_loss': 0,
    'hours_per_week': 50,
    'native_country': "United-States"
}

response = requests.post(url, headers=headers, data=json.dumps(data))

print("Status Code: ", response.status_code)
print("Response: ", response.json())
