# Demonstrate how to load a model and predict with it:

# load a model
from fastai.text import load_learner

dataset_size="large"
checkpoint_key="AWD_LSTM-clean-5_15"
print(f"Processing {dataset_size} dataset.",flush=True)
path=f"datasets/{dataset_size}"

print("Loading the model",flush=True)
learn=load_learner(path=path)
print("Done loading the model",flush=True)
probabilities = learn.predict("Hey, how is it going?")

print(f"P(is_bullying)={probabilities[2][1]}")
