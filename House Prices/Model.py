import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#%matplotlib inline

print("\nTensorFlow v" + tf.__version__)
print("TensorFlow Decision Forests v" + tfdf.__version__)

dataset_df = pd.read_csv("train.csv")
print("\n Full train dataset shape is {}".format(dataset_df.shape))

print("\n", dataset_df.head(3))

dataset_df = dataset_df.drop('Id', axis=1)
print("\n", dataset_df.head(3))

dataset_df.info()

print(dataset_df['SalePrice'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(dataset_df['SalePrice'], color='g', bins=100, hist_kws = {'alpha' : 0.4})
plt.show()

print("\n", list(set(dataset_df.dtypes.tolist())))

df_num = dataset_df.select_dtypes(include=['float64', 'int64'])
print("\n", df_num.head())

df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
plt.show()

import numpy as np

def split_dataset(dataset, test_ratio = 0.30):
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]

train_ds_pd, valid_ds_pd = split_dataset(dataset_df)
print("\n {} examples in training, {} examples in testing.\n".format(len(train_ds_pd), len(valid_ds_pd)))

label = 'SalePrice'
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label = label, task = tfdf.keras.Task.REGRESSION)
valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd, label = label, task = tfdf.keras.Task.REGRESSION)

tfdf.keras.get_all_models()

rf = tfdf.keras.RandomForestModel(hyperparameter_template="benchmark_rank1", task = tfdf.keras.Task.REGRESSION)

rf = tfdf.keras.RandomForestModel(task = tfdf.keras.Task.REGRESSION)
rf.compile(metrics=["mse"])

rf.fit(x=train_ds)

tfdf.model_plotter.plot_model_in_colab(rf, tree_idx=0, max_depth = 3)

logs = rf.make_inspector().training_logs()
plt.plot([log.num_trees for log in logs], [log.evaluation.rmse for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("RMSE (out-of-bag)")
plt.show()

inspector = rf.make_inspector()
inspector.evaluation()

evaluation = rf.evaluate(x=valid_ds, return_dict=True)

print("\n")
for name, value in evaluation.items():
    print(f"{name}: {value:.4f}")

print(f"\n Available variable importances:")
for importance in inspector.variable_importances().keys():
    print("\t", importance)

print(inspector.variable_importances()["SUM_SCORE"])

plt.figure(figsize=(12, 4))

variable_importance_metric = "SUM_SCORE"
variable_importances = inspector.variable_importances()[variable_importance_metric]

feature_names = [vi[0].name for vi in variable_importances]
feature_importances = [vi[1] for vi in variable_importances]

feature_ranks = range(len(feature_names))

bar = plt.barh(feature_ranks, feature_importances, label = [str(x) for x in feature_ranks])
plt.yticks(feature_ranks, feature_names)
plt.gca().invert_yaxis()

for importance, patch in zip(feature_importances, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{importance:.4f}", va = "top")

plt.xlabel(variable_importance_metric)
plt.title("SUM SCORE of the class 1 vs the others")
plt.tight_layout()
plt.show()

test_data = pd.read_csv("test.csv")
ids = test_data.pop('Id')

test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_data, task = tfdf.keras.Task.REGRESSION)

preds = rf.predict(test_ds)
output = pd.DataFrame({'Id' : ids, 'SalePrice' : preds.squeeze()})

print("\n", output.head(), "\n")

sample_submission_df = pd.read_csv('sample_submission.csv')
sample_submission_df['SalePrice'] = rf.predict(test_ds)
print(sample_submission_df.head())