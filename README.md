# Botnet Prediction - Computer Network Security

## Problem Description and Data

A botnet is defined as a network of private computers infected with malicious software and controlled as a group without the owners' knowledge. 

We will be using the [CTU-13 dataset](https://www.stratosphereips.org/datasets-ctu13), which is large capture of real botnet traffic mixed with normal traffic and background traffic captured from CTU University, Czech Republic, in 2011. The CTU-13 dataset consists in thirteen captures (called scenarios) of different botnet samples. On each scenario, there was a specific malware executed, which used several protocols and performed different actions. More information about this dataset can be found in the link. All CSV data can be found in the *.binetflow file of each respective directory. 

You can get the data by running the following: 
```bash
cd data
wget https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/CTU-13-Dataset.tar.bz2
```
Note: this data file is 1.9GB so it may take some time to download. 

And then unzip it:
```bash
tar -xvjf CTU-13-Dataset.tar.bz2
```

## Setup 

Create a python virtual environment: 
```bash 
python -m venv venv
```

Activate it: 
```bash 
source venv/bin/activate
```

Install all dependencies
```bash 
pip install -r requirements.txt
```

