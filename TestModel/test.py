import pandas as pd
import pickle

model = pickle.load(
    open(
        r"D:\Events\NASA\Code\Nasa-Space-Hackathon-WizCoders-backend copy\TestModel\lgbm_model.pkl",
        "rb",
    )
)

df = pd.read_csv(
    r"D:\Events\NASA\Code\Nasa-Space-Hackathon-WizCoders-backend copy\TestModel\exo.csv"
)
single_sample_data = df.iloc[0].tolist()  # Convert the first row to a list
single_sample_data = single_sample_data[1:]  # Exclude the 'id' column if present
print(single_sample_data)
preds = model.predict([single_sample_data])

label = {0: "CANDIDATE", 1: "CONFIRMED", 2: "FALSE POSITIVE"}
print(label[preds[0]])
