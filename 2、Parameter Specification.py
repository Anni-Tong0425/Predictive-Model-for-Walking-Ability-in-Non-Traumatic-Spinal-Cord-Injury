import pandas as pd
import numpy as np
import joblib


def predict_walking_ability(new_data):
    '''
    Predict walking ability for new patients.

    Parameters:
    -----------
    new_data : pandas DataFrame
        DataFrame containing the required features.
        Must include all 19 features used during training.

    Returns:
    --------
    predictions : numpy array
        Binary predictions (1 = can walk, 0 = cannot walk)
    probabilities : numpy array
        Prediction probabilities for class 1 (can walk)
    '''

    # Load model components
    model_data = joblib.load('Saved_Models/Bernoulli_Naive_Bayes_Model.pkl')
    model = model_data['model']
    scaler = model_data['scaler']
    binarizer = model_data['binarizer']
    features = model_data['features']

    # Ensure all required features are present
    missing_features = set(features) - set(new_data.columns)
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    # Extract features in correct order
    X_new = new_data[features]

    # Preprocess: Standardize and binarize
    X_scaled = scaler.transform(X_new)
    X_binary = binarizer.transform(X_scaled)

    # Make predictions
    predictions = model.predict(X_binary)
    probabilities = model.predict_proba(X_binary)[:, 1]

    return predictions, probabilities


# Example usage
new_patients = pd.DataFrame({
    '性别_编码': [0, 1, 0],  # 0 = Male, 1 = Female
    '年龄': [45, 32, 67],
    '损伤程度_编码': [2, 1, 3],  # A=0, B=1, C=2, D=3
    '分段_颈段': [1, 0, 0],
    '分段_上胸段': [0, 1, 0],
    '分段_中胸段': [0, 0, 1],
    '分段_下胸段': [0, 0, 0],
    '分段_腰段': [0, 0, 0],
    '入院L2（左）': [3, 4, 2],
    '入院L2（右）': [3, 4, 2],
    '入院L3（左）肌力': [3, 4, 2],
    '入院L3（右）肌力': [3, 4, 2],
    '入院L4（左）': [3, 4, 2],
    '入院L4（右）': [3, 4, 2],
    '入院L5（左）肌力': [3, 4, 2],
    '入院L5（右）肌力': [3, 4, 2],
    '入院S1（左）肌力': [3, 4, 2],
    '入院S1（右）肌力': [3, 4, 2],
    '入院S1针刺觉总分': [2, 4, 1]
})

predictions, probabilities = predict_walking_ability(new_patients)

for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    walking_status = "Can walk" if pred == 1 else "Cannot walk"
    print(f"Patient {i + 1}: {walking_status} (Probability: {prob:.3f})")