Contributors:
Patrick Kahardipraja    
Olena Vyshnevska

Project Structure
------------


    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical datasets for modelling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- Various documents related to the project
    │
    ├── models             <- Trained and serialised models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Mostly for data exploration
    │
    ├── report             <- Final Report
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modelling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualisation  <- Scripts to create exploratory and results oriented visualisations
    │       └── visualise.py


--------
