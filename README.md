# Botnet Prediction - Computer Network Security

## Problem Description and Data

A botnet is defined as a network of private computers infected with malicious software and controlled as a group without the owners' knowledge. 

Computer Network Traffic Data - A ~500K CSV with summary of some real network traffic data from the past. The dataset has ~21K rows and covers 10 local workstation IPs over a three month period. Half of these local IPs were compromised at some point during this period and became members of various botnets. Data is taken from the following [Stanford Public Data Repository](https://chiarasabatti.su.domains/data.html).

The columns of the CSV data are the following:
date: yyyy-mm-dd (from 2006-07-01 through 2006-09-30)
l_ipn: local IP (coded as an integer from 0-9)
r_asn: remote ASN (an integer which identifies the remote ISP)
f: flows (count of connnections for that day)

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

