import pandas as pd
import numpy as np
from utils.create_features import jaro, WRatio, ratio, davies, token_set_ratio, add_lat_lon_distance_features, strike_a_match, leven
from utils.features_utils import  extract_directions, is_direction_match, name_number_match,\
    is_related_cat, category_match, sub_category_match, brand_match, house_match, email_url_match, phone_lcs,phone_category
from Config import config
import ray.util.multiprocessing as ray
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  



def create_edit_features_em(df_pairs):


    df_pairs = add_lat_lon_distance_features(df_pairs)

    df_pairs["direction1"] = df_pairs["sourceNames1"].map(extract_directions)
    df_pairs["direction2"] = df_pairs["sourceNames2"].map(extract_directions)
    df_pairs['Is_direction_match'] = list(map(is_direction_match, df_pairs['direction1'], df_pairs['direction2']))

    df_pairs["name1_number"] = df_pairs["sourceNames1"].str.extract('(\d+)')
    df_pairs["name2_number"] = df_pairs["sourceNames2"].str.extract('(\d+)')
    df_pairs['Is_name_number_match'] = list(map(name_number_match, df_pairs['name1_number'], df_pairs['name2_number']))
    
    df_pairs['Is_related_cat'] = list(map(is_related_cat, df_pairs['category1'], df_pairs['category2']))
    df_pairs['Is_category_match'] = list(map(category_match, df_pairs['category1'], df_pairs['category2']))
    df_pairs['Is_subcategory_match'] =  list(map(sub_category_match, df_pairs['subCategory1'], df_pairs['subCategory2']))

    df_pairs['Is_brand_match'] =list(map(brand_match, df_pairs['brands1'], df_pairs['brands2']))
    df_pairs['Is_house_match'] = list(map(house_match, df_pairs['houseNumber1'], df_pairs['houseNumber2']))
    df_pairs['Is_postal_match'] = list(map(house_match, df_pairs['postalCode1'], df_pairs['postalCode2']))
    
    df_pairs['is_phone_match'] = list(map(phone_category, df_pairs['phoneNumbers1'], df_pairs['phoneNumbers2']))
    df_pairs['Is_email_match'] = list(map(email_url_match, df_pairs['email1'], df_pairs['email2']))
    df_pairs['Is_url_match'] =  list(map(email_url_match, df_pairs['internet1'], df_pairs['internet2']))

    df_pairs['name_davies'] = list(map(davies, df_pairs['sourceNames1'], df_pairs['sourceNames2']))
    df_pairs['name_leven'] = list(map(leven, df_pairs['sourceNames1'], df_pairs['sourceNames2']))
    df_pairs['name_dice'] = list(map(strike_a_match, df_pairs['sourceNames1'], df_pairs['sourceNames2']))
    df_pairs['name_jaro'] = list(map(jaro, df_pairs['sourceNames1'], df_pairs['sourceNames2']))
    df_pairs['name_set_ratio'] = list(map(token_set_ratio, df_pairs['sourceNames1'], df_pairs['sourceNames2']))

    df_pairs['street_davies'] = list(map(davies, df_pairs['streets1'], df_pairs['streets2']))
    df_pairs['street_leven'] = list(map(leven, df_pairs['streets1'], df_pairs['streets2']))
    df_pairs['street_jaro'] = list(map(jaro, df_pairs['streets1'], df_pairs['streets2']))
    
    df_pairs['email_davies'] = list(map(davies, df_pairs['email1'], df_pairs['email2']))
    df_pairs['email_leven'] = list(map(leven, df_pairs['email1'], df_pairs['email2']))
    df_pairs['email_jaro'] = list(map(jaro, df_pairs['email1'], df_pairs['email2']))
    

    df_pairs['url_davies'] = list(map(davies, df_pairs['internet1'], df_pairs['internet2']))
    df_pairs['url_leven'] = list(map(leven, df_pairs['internet1'], df_pairs['internet2']))
    df_pairs['url_jaro'] = list(map(jaro, df_pairs['internet1'], df_pairs['internet2']))

    df_pairs['brands_davies'] = list(map(davies, df_pairs['brands1'], df_pairs['brands2']))
    df_pairs['brand_leven'] = list(map(leven, df_pairs['brands1'], df_pairs['brands2']))
    df_pairs['brand_jaro'] = list(map(jaro, df_pairs['brands1'], df_pairs['brands2']))

    df_pairs['phone_lcs'] = list(map(phone_lcs, df_pairs['phoneNumbers1'], df_pairs['phoneNumbers2']))
    df_pairs['subcat_WRatio'] = list(map(WRatio, df_pairs['subCategory1'], df_pairs['subCategory2']))
    df_pairs['subcat_ratio'] = list(map(ratio, df_pairs['subCategory1'], df_pairs['subCategory2']))
    df_pairs['subcat_token_set_ratio'] = list(map(token_set_ratio, df_pairs['subCategory1'], df_pairs['subCategory2']))
    
    drop_column = ['direction1','direction2', 'name1_number', 'name2_number']
    df_pairs.drop(drop_column, axis=1, inplace = True)

    return df_pairs


def create_edit_features_file_em(file_name):
    df = pd.read_csv(
        config.input_dir + f"Fuse_exploded_{config.country}_cleaned.csv",
        engine='c',
        dtype={
            "postalCode": "str",
            "houseNumber": "str"
        },
    )

    df["phoneNumbers"] = df["phoneNumbers"].apply(eval)

    df_pairs = pd.read_parquet(file_name, engine="pyarrow")

    df_pairs = pd.merge(df_pairs,
                        df,
                        how='left',
                        left_on=['ltable_id'],
                        right_on=['Id'])
    df_pairs.drop(['placeId','Id'], inplace=True, axis=1)
    df_pairs = pd.merge(df_pairs,
                        df,
                        how='left',
                        left_on=['rtable_id'],
                        right_on=['Id'],
                        suffixes=["1", "2"])

    df_pairs.drop(['placeId','Id'], inplace=True, axis=1)
    
    if config.is_inference == False:
        df_pairs["duplicate_flag"] = (df_pairs["clusterId1"]
                                      == df_pairs["clusterId2"]) * 1

    df_pairs = create_edit_features_em(df_pairs)

    return df_pairs


def parallelize_create_edit_features_em(file_list):

    no_of_process = min(3, len(file_list))
    with ray.Pool(processes=no_of_process) as pool:
        results = list(
            tqdm(pool.imap(create_edit_features_file_em, file_list),
                 total=len(file_list)))
        new_df = pd.concat(results, axis=0, ignore_index=True)
    return new_df
