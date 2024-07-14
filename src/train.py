import os
import pandas as pd
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
import joblib
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")


from . import dispatcher

TRAINING_DATA = os.environ.get('TRAINING_DATA')
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get('FOLD'))
MODEL = os.environ.get('MODEL')

FOLD_MAPPING = {
    0 : [1, 2, 3, 4],
    1 : [0, 2, 3, 4],
    2 : [0, 1, 3, 4],
    3 : [0, 1, 2, 4],
    4 : [0, 1, 2, 3]
}


if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPING[FOLD])].reset_index(drop=True)    
    valid_df = df[df.kfold == FOLD].reset_index(drop=True)

    ytrain = train_df.Exited.values
    yvalid = valid_df.Exited.values

    train_df = train_df.drop(["id", "Exited", "kfold"], axis=1)
    valid_df = valid_df.drop(["id", "Exited", "kfold"], axis=1)

    valid_df = valid_df[train_df.columns]
    
    label_encoder = {}

    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        train_df.loc[:, c] = train_df.loc[:, c].astype(str).fillna("NONE")
        valid_df.loc[:, c] = valid_df.loc[:, c].astype(str).fillna("NONE")    
        df_test.loc[:, c] = df_test.loc[:, c].astype(str).fillna("NONE")

        lbl.fit(train_df[c].values.tolist() +
                valid_df[c].values.tolist() +
                df_test[c].values.tolist()
                )

        train_df[c] = lbl.transform(train_df[c].values.tolist())
        valid_df[c] = lbl.transform(valid_df[c].values.tolist())
        
        #df_test transform is only required to collect the encoding features for training
        # In real world you won't have test data available, You try to make code such that if there is a new category it can handle it

        label_encoder[c] = lbl

    # data is ready to train
    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:, 1]
    print(metrics.roc_auc_score(yvalid, preds))
    print(f"The ROC value of data with fold {FOLD} is : ", metrics.roc_auc_score(yvalid, preds))

    joblib.dump(label_encoder, f"models/{MODEL}_{FOLD}_label_encoder.pkl")
    joblib.dump(clf, f"models/{MODEL}_{FOLD}.pkl")
    joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")

