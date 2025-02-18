import pandas as pd
import numpy as np
import pickle
from pymongo import MongoClient
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import math
import os

# Compatibility function
def blood_type_compatible(recipient_blood, donor_blood):
    compatibility = {
        "O-": ["O-"],
        "O+": ["O-", "O+"],
        "A-": ["O-", "A-"],
        "A+": ["O-", "O+", "A-", "A+"],
        "B-": ["O-", "B-"],
        "B+": ["O-", "O+", "B-", "B+"],
        "AB-": ["O-", "A-", "B-", "AB-"],
        "AB+": ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"]
    }
    return 1 if donor_blood in compatibility.get(recipient_blood, []) else 0

# Distance calculation using Haversine formula
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


client = MongoClient('mongodb://localhost:27017/')
db = client['Database']
donors = pd.DataFrame(list(db['Donor'].find({}, {'_id': 0})))
recipients = pd.DataFrame(list(db['Reciptent'].find({}, {'_id': 0})))
model_collection = db["Model"]
def save_model_to_mongo(model, scaler, model_name="default_model"):
    model_binary = pickle.dumps(model)
    scaler_binary = pickle.dumps(scaler)
    existing_model = model_collection.find_one({"model_name": model_name})
    if existing_model:
        model_collection.update_one(
            {"model_name": model_name},
            {"$set": {"model_binary": model_binary, "scaler_binary": scaler_binary}}
        )
        print(f"Model '{model_name}' updated successfully in MongoDB.")
    else:
        model_collection.insert_one({
            "model_name": model_name,
            "model_binary": model_binary,
            "scaler_binary": scaler_binary
        })
        print(f"Model '{model_name}' saved successfully in MongoDB.")


def load_model_from_mongo(model_name="default_model"):
    try:
        model_document = model_collection.find_one({"model_name": model_name})
        if model_document:
            model = pickle.loads(model_document["model_binary"])
            scaler = pickle.loads(model_document["scaler_binary"])
            print(f"Model '{model_name}' loaded successfully from MongoDB.")
            return model, scaler
        else:
            print(f"No model found with the name '{model_name}'. Please train and save the model first.")
            return None, None
    except Exception as e:
        print(f"Error loading model from MongoDB: {e}")
        return None, None

# Prepare training data
def prepare_training_data(recipients, donors):
    urgency_mapping = {"High": 2, "Moderate": 1, "Low": 0}
    features = []
    labels = []

    for _, recipient in recipients.iterrows():
        for _, donor in donors.iterrows():
            blood_compatibility = blood_type_compatible(recipient["Recipient_Blood_Type"], donor["Blood_Type"])
            distance = calculate_distance(
                recipient["Location_Lat"], recipient["Location_Long"],
                donor["Location_Lat"], donor["Location_Long"]
            )
            urgency_level = urgency_mapping.get(recipient["Urgency_Level"], 0)
            features.append([blood_compatibility, distance, urgency_level])
            labels.append(1 if blood_compatibility == 1 and distance <= 500 else 0)

    return np.array(features), np.array(labels)

def train_and_save_model(recipients, donors, model_name="default_model"):
    """
    Train the model, save it to MongoDB, and return the model and scaler.
    """
    print("🚀 Training the model...")
    features, labels = prepare_training_data(recipients, donors)

    # Feature scaling
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

    # Train Gradient Boosting Classifier
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save the model and scaler to MongoDB
    save_model_to_mongo(model, scaler, model_name=model_name)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Return the model and scaler
    return model, scaler

# Load the model
def load_model(model_path="trained_model.pkl"):
    if os.path.exists(model_path):
        return pickle.load(open(model_path, "rb")), pickle.load(open("scaler.pkl", "rb"))
    else:
        print("No saved model found. Train the model first.")
        return None, None
def retrain_model(recipients, donors, model_name="default_model"):
    print("🔄 Retraining the model...")
    return train_and_save_model(recipients, donors, model_name=model_name)

# Match recipients with donors using the saved model
def get_top_matches(recipient_id, num_matches, recipients, donors, model, scaler):
    recipient = recipients[recipients["Recipient_ID"] == recipient_id]
    if recipient.empty:
        print(f"❌No recipient found with ID: {recipient_id}")
        return
    
    recipient = recipient.iloc[0]
    recipient_coord = (recipient["Location_Lat"], recipient["Location_Long"])
    required_blood = recipient["Recipient_Blood_Type"]
    required_organ = recipient["organ_needed"]

    matching_donors = donors[
        donors["Blood_Type"].apply(lambda x: blood_type_compatible(required_blood, x)) &
        (donors["Organ"] == required_organ)
    ]

    if matching_donors.empty:
        print(f"No matching donors found for Recipient ID: {recipient_id}")
        return

    matching_donors["distance"] = matching_donors.apply(
        lambda donor: calculate_distance(
            recipient_coord[0], recipient_coord[1],
            donor["Location_Lat"], donor["Location_Long"]
        ), axis=1
    )

    urgency_mapping = {"High": 2, "Moderate": 1, "Low": 0}
    recipient_urgency = urgency_mapping.get(recipient["Urgency_Level"], 0)
    matching_donors["blood_compatibility"] = 1
    matching_donors["urgency_level"] = recipient_urgency

    features = matching_donors[["blood_compatibility", "distance", "urgency_level"]].to_numpy()
    features_scaled = scaler.transform(features)

    matching_donors["match_score"] = model.predict_proba(features_scaled)[:, 1]
    top_matches = matching_donors.nlargest(num_matches, "match_score")

    print(f"\nTop {num_matches} Matches for Recipient ID: {recipient_id}:\n")
    print(top_matches[["Donor_ID", "Blood_Type", "Organ", "distance", "match_score"]])

# Match donors with recipients using the saved model
def get_top_matches_for_donor(donor_id, num_matches, recipients, donors, model, scaler):
    donor = donors[donors["Donor_ID"] == donor_id]
    if donor.empty:
        print(f"❌ No donor found with ID: {donor_id}")
        return  # Exit function

    donor = donor.iloc[0]
    donor_coord = (donor["Location_Lat"], donor["Location_Long"])
    donor_blood = donor["Blood_Type"]
    donor_organ = donor["Organ"]

    # Filter recipients who need this donor's organ and have compatible blood type
    matching_recipients = recipients[
        recipients["Recipient_Blood_Type"].apply(lambda x: blood_type_compatible(x, donor_blood)) &
        (recipients["organ_needed"] == donor_organ)
    ].copy()

    if matching_recipients.empty:
        print(f"❌ No matching recipients found for Donor ID: {donor_id}")
        return  # Exit function

    # Calculate distance
    matching_recipients.loc[:, "distance"] = matching_recipients.apply(
        lambda recipient: calculate_distance(
            donor_coord[0], donor_coord[1],
            recipient["Location_Lat"], recipient["Location_Long"]
        ), axis=1
    )

    # Prepare features
    urgency_mapping = {"High": 2, "Moderate": 1, "Low": 0}
    matching_recipients.loc[:, "urgency_level"] = matching_recipients["Urgency_Level"].map(urgency_mapping)
    matching_recipients.loc[:, "blood_compatibility"] = 1  # Already filtered for compatibility

    # Convert to numpy and scale
    features = matching_recipients[["blood_compatibility", "distance", "urgency_level"]].to_numpy()
    features_scaled = np.nan_to_num(scaler.transform(features))  # Fix NaN issues

    # If features are empty, exit gracefully
    if features_scaled.shape[0] == 0:
        print(f"❌ No valid matches found for Donor ID: {donor_id}")
        return  # Exit function

    # Predict match scores
    matching_recipients["match_score"] = model.predict_proba(features_scaled)[:, 1]

    # Sort and get top matches
    top_matches = matching_recipients.nlargest(num_matches, "match_score")

    if top_matches.empty:
        print(f"❌ No suitable recipients found for Donor ID: {donor_id}")
        return  # Exit function

    # Display results
    print(f"\n✅ Top {num_matches} Matches for Donor ID: {donor_id} (Blood Type: {donor_blood}, Organ Available: {donor_organ}):\n")
    print(top_matches[["Recipient_ID", "Recipient_Blood_Type", "organ_needed", "distance", "match_score"]])



# Main loop
model, scaler = load_model_from_mongo("default_model")
while model is None or scaler is None:
    print("⚠️ Model not found. Retraining...")
    model, scaler = retrain_model(recipients, donors, model_name="default_model")

while True:
    print("\nWhat do you want to do?")
    print("1. Fetch top matches for a recipient.")
    print("2. Fetch top matches for a donor.")  # NEW OPTION
    print("3. Retrain the model.")
    print("4. Exit.")
    
    choice = input("Enter your choice: ")

    if choice == "1":
        recipient_id = input("Enter Recipient ID: ")
        num_matches = int(input("Enter number of matches to retrieve: "))
        if model and scaler:
            get_top_matches(recipient_id, num_matches, recipients, donors, model, scaler)
        else:
            print("⚠️ Model not loaded. Please retrain or load a model first.")

    elif choice == "2":
        donor_id = input("Enter Donor ID: ")
        num_matches = int(input("Enter number of matches to retrieve: "))
        if model and scaler:
            get_top_matches_for_donor(donor_id, num_matches, recipients, donors, model, scaler)
        else:
            print("⚠️ Model not loaded. Please retrain or load a model first.")

    elif choice == "3":
        print("🔄 Retraining the model...")
        model, scaler = train_and_save_model(recipients, donors, model_name="default_model")

    elif choice == "4":
        print("👋 Exiting...")
        break

    else:
        print("❌ Invalid choice. Please try again.")






