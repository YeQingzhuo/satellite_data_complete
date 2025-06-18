# Satellite Data Processing and Training Repository

## Project Overview
This repository is primarily dedicated to satellite data processing and training tasks. It encompasses multiple aspects, including data loading, dataset construction, and the implementation of custom loss functions. The code supports the processing of various datasets, such as satellite data, solar data, ETTh1, and PM2.5. Additionally, it provides corresponding functionalities for different training tasks, such as PriSTI, NTM, NTF, NTC, Costco, and TimesNet.

## Directory Structure
```
satellite_data_complete/
├── data_provider/
│   └── data_loader.py  # Data loading module
├── train/
│   ├── PriSTI/
│   │   ├── sun/
│   │   │   └── dataset_sun_PriSTI.py  # Dataset construction for solar data in PriSTI
│   │   ├── Satellite/
│   │   │   └── dataset_Satellite_PriSTI.py  # Dataset construction for satellite data in PriSTI
│   │   ├── ETTh1/
│   │   │   └── dataset_ETTh1_PriSTI.py  # Dataset construction for ETTh1 data in PriSTI
│   │   └── PM25/
│   │       └── dataset_pm25_PriSTI.py  # Dataset construction for PM2.5 data in PriSTI
│   ├── NTM/
│   │   ├── PM25/
│   │   │   ├── NTM_PM25_1.py  # Custom loss function 1 for PM2.5 data in NTM
│   │   │   └── NTM_PM25_2.py  # Custom loss function 2 for PM2.5 data in NTM
│   │   └── ETTh1/
│   │       ├── NTM_ETTh1_1.py  # Custom loss function 1 for ETTh1 data in NTM
│   │       ├── NTM_ETTh1_2.py  # Custom loss function 2 for ETTh1 data in NTM
│   │       └── NTM_ETTh1_3.py  # Custom loss function 3 for ETTh1 data in NTM
│   ├── NTF/
│   │   └── PM25/
│   │       ├── NTF_PM25_1.py  # Custom loss function 1 for PM2.5 data in NTF
│   │       └── NTF_PM25_2.py  # Custom loss function 2 for PM2.5 data in NTF
│   ├── NTC/
│   │   └── PM25/
│   │       ├── NTC_PM25_1.py  # Custom loss function 1 for PM2.5 data in NTC
│   │       └── NTC_PM25_4.py  # Custom loss function 4 for PM2.5 data in NTC
│   ├── Costco/
│   │   └── PM25/
│   │       └── Costco_PM25_1.py  # Custom loss function for PM2.5 data in Costco
│   └── TimesNet/
│       └── TimesNet_5.py  # Custom loss function for TimesNet
```

## Main Functional Modules

### Data Loading Module (`data_provider/data_loader.py`)
This module implements data reading and processing functions, supporting different types of data (e.g., satellite data, solar data). It filters and processes data according to different flags (`flag`). Additionally, it handles NaN and Inf values in the data and extracts timestamp features.

### Dataset Construction Module (`train/PriSTI/*/dataset_*.py`)
These modules construct corresponding dataset classes for different datasets (solar data, satellite data, ETTh1, PM2.5). They support data missingness handling (random missingness and continuous missingness) and generate corresponding masks. During the training and testing processes, different conditional masks can be generated based on different modes (`mode`).

### Custom Loss Function Module (`train/*/*/*.py`)
These modules implement various custom loss functions, including simple absolute value loss functions (`CustomLoss`) and weighted loss functions (`CustomWeightedLoss`), which are used for different training tasks.

## Usage

### Data Loading
```python
from data_provider.data_loader import DataLoader

# Initialize the data loader
data_loader = DataLoader(root_path='your_root_path', data_path='your_data_path', ...)

# Read data
data_loader.__read_data__()
```

### Dataset Construction
```python
from train.PriSTI.Satellite.dataset_Satellite_PriSTI import Dataset

# Initialize the dataset
dataset = Dataset(current_day=0, num=10, ...)

# Get the length of the dataset
length = len(dataset)

# Get a data sample
sample = dataset[0]
```

### Custom Loss Function
```python
import torch
from train.NTM.PM25.NTM_PM25_1 import CustomLoss

# Initialize the loss function
criterion = CustomLoss()

# Calculate the loss
y_pred = torch.randn(10, 36)
y_true = torch.randn(10, 36)
loss = criterion(y_pred, y_true)
```

## Notes
- Ensure that the paths to the data files are correct and that the data files are in the required format.
- When using custom loss functions, select the appropriate loss function based on the specific task and data type.
- For dataset construction, adjust the parameters (e.g., `current_day`, `num`, `missing_ratio`) according to the actual situation.

## Contribution
If you have any suggestions or improvements for the code in this repository, please feel free to submit a Pull Request or raise an Issue.

## License
The code in this repository is licensed under the [Specific License Name] license. Please comply with the corresponding license terms when using the code.
