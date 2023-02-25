from fastapi import FastAPI
import pandas as pd
import uvicorn
from xgboost import XGBClassifier
import lightgbm as lgbm
from Config import config
from utils.create_features_em import create_edit_features_em
from utils.helper import name_distance
from itertools import combinations
from sentence_transformers import SentenceTransformer

app = FastAPI()

df = pd.read_csv("/code/app/data/Fuse_exploded_NLD_cleaned.csv")

# loading all models

xgb_model = XGBClassifier()
xgb_model.load_model("/code/app/models/xgboost_final_v5.json")
model = lgbm.Booster(model_file="/code/app/models/lgb_final_v5.txt")
sbert_model = SentenceTransformer('/code/app/models/all-mpnet-base-v2',device="cpu")
sbert_model.max_seq_length = 64

# Loading the pre predicted pair.

def create_feature(df_pairs):
    
    similarity = name_distance(sbert_model,list(df_pairs["sourceNames1"]),list(df_pairs["sourceNames2"]))
    df_pairs = create_edit_features_em(df_pairs)
    df_pairs[config.cat_columns] = df_pairs[config.cat_columns].astype("category")
    df_pairs["similarity"] = similarity
    
    return df_pairs

def create_pairs(df_compare):
    
    matches = list(combinations(df_compare["Id"].tolist(), 2))
    df_pairs = pd.DataFrame(matches)
    df_pairs.columns = ["ltable_id", "rtable_id"]
    df_pairs.drop_duplicates(inplace=True)
    df_pairs = df_pairs[~(df_pairs["ltable_id"] == df_pairs["rtable_id"])]
    df_pairs = pd.merge(df_pairs,
                        df_compare,
                        how='left',
                        left_on=['ltable_id'],
                        right_on=['Id'])
    df_pairs.drop(['Id'], inplace=True, axis=1)
    df_pairs = pd.merge(df_pairs,
                        df_compare,
                        how='left',
                        left_on=['rtable_id'],
                        right_on=['Id'],
                        suffixes=["1", "2"])

    df_pairs.drop(['Id'], inplace=True, axis=1)
    df_pairs = df_pairs[df_pairs["placeId1"]!=df_pairs["placeId2"]]


    if config.is_inference == False:
        df_pairs["duplicate_flag"] = (df_pairs["clusterId1"]
                                    == df_pairs["clusterId2"]) * 1
        
    return df_pairs

@app.get("/")
def root():
    return {"Hello": "This is ML matching api"}

@app.get("/match_score")
async def match_score(poi1: str, poi2: str):
    
    df_compare = df[df["placeId"].isin([poi1,poi2])]
    df_pairs = create_pairs(df_compare)
    df_pairs = create_feature(df_pairs)
    xgb_pred = xgb_model.predict_proba(df_pairs[config.EM_features])[:, 1]
    lgb_pred = model.predict(df_pairs[config.EM_features])
    return {"xgb_pred": str(max(xgb_pred)),"lgb_pred" : str(max(lgb_pred))}
    
    
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)