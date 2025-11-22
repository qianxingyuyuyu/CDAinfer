# CDAinfer

## Environment Dependencies

- Python >= 3.8  
- See `requirements.txt` for details  

Install dependencies:  
```bash  
pip install -r requirements.txt
```

## Data Preparation

- Directory: `data_processing`
- Split the original data into `price.csv`, `transaction.csv` and `user.csv`. (Due to file size limitations for uploads, `transaction.csv` need to be unzipped) Run `data_processing.py` to generate `newnewdata_nn.jsonl`. Due to file size limitations for uploads, only 2.5% of the `newnewdata_nn.jsonl` data volume is displayed here `sampled dataset.jsonl`.

- Additionally, this directory contains files including `bound.jsonl`, `inference_data.jsonl`, and `inference_market.txt`, which are used for boundary calculation and inference respectively.


## Quick Start

### Train Environment Model

```bash
python training/regression_linear_new.py
```
Outputs `saved_model/regression_model.h5` - a universal model for both buyers and sellers in the CDA market environment.

To run market simulations, use `market_simulate_new.py`.
For visualization, use `simulate_data.py`.

### Train the RL Agent
Train a risk-neutral agent:
```bash
python agent-all-market.py
```
Fine-tune a risk-sensitive agent:
```bash
python finetune.py --risk_gamma 0.5
```

## Inference Results
```bash
python mse.py --logfile actor_model_risk_gamma_0.5.txt --modelfile actor_model_risk_gamma_0.5.h5
```

## Other Comparison Methods
`PGM.py`

Under the `compare`directory:
- `blue.py`
- `CrossEntropy.py`
- `DL.py`
- `RandomMedian.py`
- `SL.py`
- `RvS.py`

## Directory Structure

```

CDAinfer/
├── data_processing/
│   ├── price.csv                  # Price data from the original dataset
│   ├── transaction.csv            # Transaction data from the original dataset
│   ├── user.csv                   # User data from the original dataset  
│   ├── newnewdata_nn.jsonl        # Real observations and actions for training the environment model 
│   ├── sampled dataset.jsonl      # Subsampled version of the above data  
│   ├── inference_data.jsonl       # Real observations and actions for inference  
│   ├── data_processing.py         # Processes raw data into a format suitable for model training  
│   ├── bound.jsonl                # Used for boundary computation (can be generated with bound.py)
│   ├── inference_market.txt       # Market IDs for inference 
│   └── ...                        # Other data processing scripts  
├── training/
│   ├── regression_linear.py       # Trains the general model for buyers and sellers in the environment 
│   └── utils.py                   # Utility functions for data formatting, etc. 
├── compare/                       # Comparison methods  
│   ├── blue.py         
│   ├── CrossEntropy.py
│   ├── DL.py 
│   ├── RandomMedian.py         
│   ├── SL.py     
│   ├── RvS.py
│   └── ...                        # Reusable functions       
├── saved_model/                   
│   ├── regression_model_new.h5    # General model for buyers and sellers in the environment 
│   ├── expert_model.h5            # Expert model for PGM 
│   ├── history_mean.npy           # Normalization data for training and model inference  
    ├── history_std.npy            
    └── ...
├── requirements.txt               # List of dependencies  
├── agent-all-market.py            # Trains a risk-neutral agent  
├── finetune.py                    # Fine-tunes a risk-sensitive agent  
├── mse.py                         # Computes inference results
├── PGM.py                         
├── market_simulate_new.py         # Runs market simulation  
└── README.md                      # Project documentation  
```
