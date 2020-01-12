# to open the models
import pickle

# to work with arrays and dataframes
import numpy as np
import pandas as pd

# server stuff
import uvicorn
from fastapi import FastAPI

# to pass stuff to the get requests
from starlette.requests import Request
# to serve static html
from starlette.staticfiles import StaticFiles
# to serve html pages
from starlette.responses import HTMLResponse, FileResponse


# to create molecules from SMILES
from rdkit import Chem
from rdkit.Chem import Descriptors # to get the names of all the RDKit descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors # the names of all desc-s get passed here

# this calculates Mordred descriptors
from mordred import Calculator, descriptors

import logging

app = FastAPI()

my_logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG, filename='sample.log')

# to serve static html, see https://fastapi.tiangolo.com/tutorial/templates/
app.mount("/static", StaticFiles(directory="static"), name="static")

# initialize the models

# - - - - - - - - - - - - models trained using 366 random molecules - - - - - - 
# # on RDKit descriptors
with open("./prod_models/for_random_inactive/rdkit/2drdkit_random_366_logreg.pickle", "rb") as file:
	rdkit_366_random_logreg = pickle.load(file)

with open("./prod_models/for_random_inactive/rdkit/2drdkit_random_366_rf.pickle", "rb") as file:
	rdkit_366_random_rf = pickle.load(file)

# with open("./prod_models/for_random_inactive/rdkit/2drdkit_random_366_svm.pickle", "rb") as file:
# 	rdkit_366_random_svm = pickle.load(file)

# # on Mordred descriptors
with open("./prod_models/for_random_inactive/mordred/mordred_random_366_logreg.pickle", "rb") as file:
	mordred_366_random_logreg = pickle.load(file)

with open("./prod_models/for_random_inactive/mordred/mordred_random_366_rf.pickle", "rb") as file:
	mordred_366_random_rf = pickle.load(file)

# with open("./prod_models/for_random_inactive/mordred/mordred_random_366_svm.pickle", "rb") as file:
# 	mordred_366_random_svm = pickle.load(file)

#  - - - - - models trained using 366 autoencoder-selected molecules - - - - - - 
# # on RDKit descriptors
with open("./prod_models/for_autoencoder_inactive/rdkit/2drdkit_autoencoder_366_logreg.pickle", "rb") as file:
	rdkit_autoencoder_logreg = pickle.load(file)

with open("./prod_models/for_autoencoder_inactive/rdkit/2drdkit_autoencoder_366_rf.pickle", "rb") as file:
	rdkit_autoencoder_rf = pickle.load(file)

# with open("./prod_models/for_autoencoder_inactive/rdkit/2drdkit_autoencoder_366_svm.pickle", "rb") as file:
# 	rdkit_autoencoder_svm = pickle.load(file)

# # on Mordred descriptors
with open("./prod_models/for_autoencoder_inactive/mordred/mordred_autoencoder_366_logreg.pickle", "rb") as file:
	mordred_autoencoder_logreg = pickle.load(file)

with open("./prod_models/for_autoencoder_inactive/mordred/mordred_autoencoder_366_rf.pickle", "rb") as file:
	mordred_autoencoder_rf = pickle.load(file)

# with open("./prod_models/for_autoencoder_inactive/mordred/mordred_autoencoder_366_svm.pickle", "rb") as file:
# 	mordred_autoencoder_svm = pickle.load(file)

names_of_all_rdkit_descriptors = [x[0] for x in Descriptors._descList]
rdkit_descriptors_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(names_of_all_rdkit_descriptors)
mordred_desc_calculator = Calculator(descriptors, ignore_3D=True)

@app.get("/")
def root():
	with open("./static/index_2.html", "r") as page_file:
		html_content = page_file.read()
	return HTMLResponse(content=html_content)

@app.get("/tutorial")
def show_tutorial():
	with open("./static/tutorial.html", "r") as page_file:
		html_content = page_file.read()
	return HTMLResponse(content=html_content)

@app.get("/about")
def about():
	with open("./static/about.html", "r") as page_file:
		html_content = page_file.read()
	return HTMLResponse(content=html_content)

@app.get("/api_info")
def api_info():
	with open("./static/api_info.html", "r") as page_file:
		html_content = page_file.read()
	return HTMLResponse(content=html_content)

@app.get("/img1")
def get_image1():
	return FileResponse("./static/img/img1.png")

@app.post("/predict")
def predict(list_of_smiles):
	"""
	Accepts a series of SMILES strings, separated by commas and no spaces
	Example: C1=CC=CC=C1,C1=CC=CC=C1,C1=CC=CC=C1,C1=CC=CC=C1
	Iterates through the list, creates the molecules, calculates the descriptors
	Returns the models' predictions for given descriptors
	"""
	my_logger.debug("Inside the /predict")
	my_logger.debug("Received: ", str(list_of_smiles))
	list_of_smiles = list_of_smiles.split(',')
	my_logger.debug("List of smiles splitted : ", list_of_smiles)
	try:
		my_logger.debug("inside try")
		list_of_molecules = [Chem.MolFromSmiles(x) for x in list_of_smiles]
		my_logger.debug("molecules parsed")
		
		# use this
		rdkit_desc_for_posted_list_of_mols = np.array([rdkit_descriptors_calculator.CalcDescriptors(mol) for mol in list_of_molecules])
		my_logger.debug("rdkit calculated")

		mordred_desc_for_posted_list_of_mols_df = mordred_desc_calculator.pandas(list_of_molecules)
		my_logger.debug("mordred1")
		mordred_desc_for_posted_list_of_mols_df = mordred_desc_for_posted_list_of_mols_df.drop(["SpAbs_Dt", "SpMax_Dt", "SpDiam_Dt", "SpAD_Dt", "SpMAD_Dt", "LogEE_Dt", 
	               "SM1_Dt", "VE1_Dt", "VE2_Dt", "VE3_Dt", "VR1_Dt", "VR2_Dt", "VR3_Dt", 
	               "DetourIndex"], axis = 1)
		my_logger.debug("mordred2")
		mordred_desc_for_posted_list_of_mols_df = mordred_desc_for_posted_list_of_mols_df.apply(pd.to_numeric, errors='coerce').fillna(0)
		my_logger.debug("mordred calculated")

		# and this
		mordred_desc_for_posted_list_of_mols = mordred_desc_for_posted_list_of_mols_df.values
		my_logger.debug("mordred values passed")

		# print("\n\n\nThis happens before return\n\n\n")

		# print(rdkit_366_random_logreg.predict_proba(rdkit_desc_for_posted_list_of_mols)[:,-1]) # being active
		# print("\n\n")
		# print(rdkit_366_random_svm.predict(rdkit_desc_for_posted_list_of_mols)) # 

		return  {
			"rdkit_366_random_logreg": ",".join([str(x) for x in rdkit_366_random_logreg.predict_proba(rdkit_desc_for_posted_list_of_mols)[:,-1]]),
			"rdkit_366_random_rf": ",".join([str(x) for x in rdkit_366_random_rf.predict_proba(rdkit_desc_for_posted_list_of_mols)[:,-1]]),
			"mordred_366_random_logreg": ",".join([str(x) for x in mordred_366_random_logreg.predict_proba(mordred_desc_for_posted_list_of_mols)[:,-1]]),
			"mordred_366_random_rf": ",".join([str(x) for x in mordred_366_random_rf.predict_proba(mordred_desc_for_posted_list_of_mols)[:,-1]]),
			"rdkit_autoencoder_logreg": ",".join([str(x) for x in rdkit_autoencoder_logreg.predict_proba(rdkit_desc_for_posted_list_of_mols)[:,-1]]),
			"rdkit_autoencoder_rf": ",".join([str(x) for x in rdkit_autoencoder_rf.predict_proba(rdkit_desc_for_posted_list_of_mols)[:,-1]]),
			"mordred_autoencoder_logreg": ",".join([str(x) for x in mordred_autoencoder_logreg.predict_proba(mordred_desc_for_posted_list_of_mols)[:,-1]]),
			"mordred_autoencoder_rf": ",".join([str(x) for x in mordred_autoencoder_rf.predict_proba(mordred_desc_for_posted_list_of_mols)[:,-1]]),
		}
	except:
		my_logger.debug("inside except")
		return  {
			"error_code": "1"
		}