import requests
import pprint

pp = pprint.PrettyPrinter(indent=4)

to_predict_dict = {
	"list_of_smiles": "C1=CC=CC=C1"
}

# url = 'http://127.0.0.1:8000/predict'
url = 'http://35.246.144.115:8000/predict'
r = requests.post(url, params={"list_of_smiles": "C1=CC=CC=C1"})
pp.pprint(r.json())

print(r.json()["mordred_366_random_logreg"])