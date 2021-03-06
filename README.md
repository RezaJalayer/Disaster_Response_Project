## Disaster Response Pipeline
### libraries used: NumPy, Pandas, SkLearn, Nltk, Sqlalchemy, Pickle
### Motivations: Build a machine learning pipeline and launch it in a web app to categorize emergency messages based on the needs communicated by the sender 
### Instructions:
1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

#### acknowledgement: Figure Eight Data
