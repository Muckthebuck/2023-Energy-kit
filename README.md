

## Installation

Download Conda.

setup conda path in .vscode/settings.json

change the following lines if conda isnt recognised
  "python.defaultInterpreterPath": "C:\\user\\anaconda3\\python.exe",   // here 
  "python.condaPath": "C:\\user\\anaconda3\\Scripts\\conda.exe" // here


Run `conda env create -f SOML.yaml` at root folder.

## File structure

### directories
/data: has source and cleaned data
/alpg-master: code used to generate synthetic load data
/diagrams: some of the graphs and diagrams used in reports
/graphs: some of the graphs generated during exploration process
/saved_model: saved trained ml models

### python classes
- ml_models.py         : loads pre-trained ml models and performs online training
- online_batches.py    : loads pre-made sliding window batches (for ease of use in ml_model.py)

### jupyter notebooks

####  main jupyter notebook to run the project

#### Optimisation 
 - optimization(bidirectional_pricing).ipynb
 - optimization(unidirectional_pricing).ipynb
#### ML training and exploration
- \[Austin\]solar_temp_predictions.ipynb                : Data cleaning and DNN models experiments for solar insolation and temp predictions
- \[BOM\]solar_predictions.ipynb                        : Data cleaning and DNN models experiments for BOM-72 solar insolation predictions.
- \[BOM\]Outdoor_temperature_predictions.ipynb          : Data cleaning and classical ML models experiments for BOM dataset.
- complete_temp.ipynb                                   : Data cleaning and DNN models for BOM-72 temperature predictions.
- load_dnn.ipynb                                        : DNN models experiments for load predictions
- load_prediction.ipynb                                 : Classical ML models for load predictions
- mol_model_testin.ipynb                                : To test teh ml_model.py and online_batches.py classes
- load_data.ipynb





