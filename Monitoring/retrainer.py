import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

def train_challenger(training_data, features, target='churned', random_state=42):
    # train a new model given whichever training data
    # agnostic from orchestrator and alert system
    # just performs training congruent to that of baseline model
    # possible improvement: make reproducible training objects for baseline model + retraining
    # so that it doesnt have to be model-architecture-specific and more modular
    # plus this is full retraining -- most in-prod models will just be fine-tuning / lora type

    X = training_data[features]
    y = training_data[target]

    # no need for a train-test split since no holdout training happens 
    # and evaluation is done through shadow testing retrained model / challenger against prod

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    class_weights = compute_class_weight('balanced', classes=np.array([0,1]), y=y)
    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        class_weight={0: class_weights[0], 1: class_weights[1]}
    )

    model.fit(X_scaled, y)

    return model, scaler