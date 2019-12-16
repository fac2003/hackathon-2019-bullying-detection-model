from fastai.text import *

from model.metrics import F1

dataset_size="small-100"
checkpoint_key="AWD_LSTM-clean-5_15"
print(f"Processing {dataset_size} dataset.",flush=True)
path=f"datasets/{dataset_size}"
batch_size_lm=64
batch_size_cls=32
data_lm = load_data(path, f'bullying_lm_export.pkl',bs=batch_size_lm,val_bs=batch_size_lm)
data_clas = load_data(path, f'bullying_clas_export.pkl',bs=batch_size_cls,val_bs=batch_size_cls)
fine_tune_lm=False
model_type=AWD_LSTM
model_filename=f"ft_enc-{checkpoint_key}.pth"
model_path=f"{path}/models/{model_filename}"
if fine_tune_lm or not os.path.exists(model_path):
    print("Fine-tuning the language model",flush=True)
    learn = language_model_learner(data_lm, model_type, drop_mult=0.5)
    learn.fit_one_cycle(5, 1e-2)
    print("Unfreezing the language model", flush=True)
    learn.unfreeze()
    learn.fit_one_cycle(15, 1e-3)
    learn.save_encoder(model_filename)


print("Training the classifier",flush=True)

metrics=[accuracy, F1]
learn = text_classifier_learner(data_clas, model_type, drop_mult=0.5,metrics=metrics)

learn.load_encoder(model_filename)
num_epochs=20
total_epochs=0


def one_round(learn,current_epoch,total_epochs,num_epochs=5, lr=1E-2):

    learn.freeze_to(-2)
    learn.fit_one_cycle(1, slice(5*lr/2/10 / 2., 5*lr/10/2))
    learn.unfreeze()
    learn.fit_one_cycle(num_epochs, slice(2*lr/10 / 100, 2*lr/10))
    current_epoch += num_epochs
    saved_to = learn.save(f"bullying_model-{checkpoint_key}_{current_epoch}", return_path=True)

    print(f"model saved: {saved_to}")
    return current_epoch

max_epochs=100
lr=1E-3
learn.fit_one_cycle(1, lr)
learn.freeze_to(-2)
for round in range(int(max_epochs/num_epochs)):
    total_epochs = one_round(learn,
                             current_epoch=total_epochs,
                             total_epochs=max_epochs,
                             num_epochs=num_epochs, lr=lr)
