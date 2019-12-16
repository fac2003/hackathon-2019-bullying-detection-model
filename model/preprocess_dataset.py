from fastai.text import *

dataset_size = "small-100"
path = f"datasets/{dataset_size}"
df = pd.read_csv(f"{path}/dataset.tsv",delimiter="\t")
print(df.head())
batch_size=4
# Language model data
data_lm = TextLMDataBunch.from_csv(path, "dataset.tsv",
                                   label_cols=0,
                                   text_cols=1,
                                   delimiter="\t")
# Classifier model data
data_clas = TextClasDataBunch.from_csv(path,
                                       'dataset.tsv',
                                       label_cols=0,
                                       text_cols=1,
                                       vocab=data_lm.train_ds.vocab,
                                       bs=batch_size,
                                       delimiter="\t")

data_lm.save('bullying_lm_export.pkl')
data_clas.save('bullying_clas_export.pkl')
