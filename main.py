from src.preprocessing import load_data, preprocess
from src.model import train_model, save_model

df = load_data("data/creditcard.csv")

X, y = preprocess(df)

model = train_model(X, y)

save_model(model)

print("✅ Model trained and saved successfully!")