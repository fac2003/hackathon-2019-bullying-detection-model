# Demonstrate how to load a model and predict with it:

# load a model
from fastai.basic_data import load_data
from fastai.text import AWD_LSTM, language_model_learner, load_learner

dataset_size="small-10"
checkpoint_key="AWD_LSTM-clean-5_15"
print(f"Processing {dataset_size} dataset.",flush=True)
path=f"datasets/{dataset_size}"
model_filename="models/bullying_model-AWD_LSTM-clean-5_15_1-exported"
# model_filename="bullying_model-AWD_LSTM-clean-5_15_20"
data_lm = load_data(path, f'bullying_lm_export.pkl')
data_clas = load_data(path, f'bullying_clas_export.pkl')

model_type=AWD_LSTM

model_path=f"{path}/{model_filename}"

print("Loading the model",flush=True)
learn=load_learner(path="datasets/large")
# learn = language_model_learner(data_lm, model_type, drop_mult=0.5)

probabilities = learn.predict("Hey, how is it going?")

print(f"P(is_bullying)={probabilities[2][1]}")
