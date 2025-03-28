import pickle
from joblib import load

# # List of individual model filenames
# model_files = ["models/Boosting/SVMwBoosting-03-24-2025_20-59-16_0.model", "models/Boosting/SVMwBoosting-03-24-2025_20-59-16_1.model",
# "models/Boosting/SVMwBoosting-03-24-2025_20-59-16_2.model", "models/Boosting/SVMwBoosting-03-24-2025_20-59-16_3.model",
# "models/Boosting/SVMwBoosting-03-24-2025_20-59-16_4.model"]

# # Load each model and store in a list
# ensemble_models = []
# for file in model_files:
#     model = load(file)  # Change this if a different method was used for saving
#     ensemble_models.append(model)

# # Save the entire ensemble as a single file
# with open("ensemble_model.pkl", "wb") as f:
#     pickle.dump(ensemble_models, f)

# print("Ensemble saved successfully!")

with open("models/Boosting/ensemble_model_boosting.pkl", "rb") as f:
    content = pickle.load(f)
    print(content)  # This will let you inspect what is inside the pickle file