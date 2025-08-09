import pickle

with open('model_features.pkl', 'rb') as f:
    features = pickle.load(f)

print(features)
