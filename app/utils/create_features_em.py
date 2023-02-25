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



def create_edit_features_em_apply(df_pairs):


    df_pairs = add_lat_lon_distance_features(df_pairs)

    df_pairs["direction1"] = df_pairs["sourceNames1"].apply(extract_directions)
    df_pairs["direction2"] = df_pairs["sourceNames2"].apply(extract_directions)
    df_pairs['Is_direction_match'] = df_pairs.apply(lambda x: is_direction_match(x.direction1, x.direction2), axis=1)

    df_pairs["name1_number"] = df_pairs["sourceNames1"].str.extract('(\d+)')
    df_pairs["name2_number"] = df_pairs["sourceNames2"].str.extract('(\d+)')
    df_pairs['Is_name_number_match'] = df_pairs.apply(lambda x: name_number_match(x.name1_number, x.name2_number), axis=1)
    
    df_pairs['Is_related_cat'] = df_pairs.apply(lambda x: is_related_cat(x.category1, x.category2), axis=1)
    df_pairs['Is_category_match'] = df_pairs.apply(lambda x: category_match(x.category1, x.category2), axis=1)
    df_pairs['Is_subcategory_match'] =  df_pairs.apply(lambda x: sub_category_match(x.subCategory1, x.subCategory2), axis=1)


    df_pairs['Is_brand_match'] = df_pairs.apply(lambda x: brand_match(x.brands1, x.brands2), axis=1)
    df_pairs['Is_house_match'] = df_pairs.apply(lambda x: house_match(x.houseNumber1, x.houseNumber2), axis=1)
    df_pairs['Is_postal_match'] = df_pairs.apply(lambda x: house_match(x.postalCode1, x.postalCode2), axis=1)
    df_pairs['is_phone_match'] = df_pairs.apply(lambda x: phone_category(x.phoneNumbers1, x.phoneNumbers2), axis=1)
    
    
    df_pairs['Is_email_match'] = df_pairs.apply(lambda x: email_url_match(x.email1, x.email2), axis=1)
    df_pairs['Is_url_match'] = df_pairs.apply(lambda x: email_url_match(x.internet1, x.internet2), axis=1)

    df_pairs['name_davies'] = df_pairs.apply(lambda x: davies(x.sourceNames1, x.sourceNames2), axis=1)
    df_pairs['name_leven'] = df_pairs.apply(lambda x: leven(x.sourceNames1, x.sourceNames2), axis=1)
    df_pairs['name_dice'] = df_pairs.apply(lambda x: strike_a_match(x.sourceNames1, x.sourceNames2), axis=1)
    df_pairs['name_jaro'] = df_pairs.apply(lambda x: jaro(x.sourceNames1, x.sourceNames2), axis=1)
    df_pairs['name_set_ratio'] = df_pairs.apply(lambda x: token_set_ratio(x.sourceNames1, x.sourceNames2), axis=1)
    
    df_pairs['street_davies'] =  df_pairs.apply(lambda x: davies(x.streets1, x.streets2), axis=1)
    df_pairs['street_leven'] =  df_pairs.apply(lambda x: leven(x.streets1, x.streets2), axis=1)
    df_pairs['street_jaro'] =  df_pairs.apply(lambda x: jaro(x.streets1, x.streets2), axis=1)

    df_pairs['email_davies'] =  df_pairs.apply(lambda x: davies(x.email1, x.email2), axis=1)
    df_pairs['email_leven'] =  df_pairs.apply(lambda x: leven(x.email1, x.email2), axis=1)
    df_pairs['email_jaro'] =  df_pairs.apply(lambda x: jaro(x.email1, x.email2), axis=1)

    df_pairs['url_davies'] =  df_pairs.apply(lambda x: davies(x.internet1, x.internet2), axis=1)
    df_pairs['url_leven'] = df_pairs.apply(lambda x: leven(x.internet1, x.internet2), axis=1)
    df_pairs['url_jaro'] =  df_pairs.apply(lambda x: jaro(x.internet1, x.internet2), axis=1)
    
    
    df_pairs['brands_davies'] =  df_pairs.apply(lambda x: davies(x.brands1, x.brands2), axis=1)
    df_pairs['brand_leven'] = df_pairs.apply(lambda x: leven(x.brands1, x.brands2), axis=1)
    df_pairs['brand_jaro'] =  df_pairs.apply(lambda x: jaro(x.brands1, x.brands2), axis=1)


    df_pairs['phone_lcs'] = df_pairs.apply(lambda x: phone_lcs(x.phoneNumbers1, x.phoneNumbers2), axis=1)
    
        
    df_pairs['subcat_WRatio'] = df_pairs.apply(lambda x: WRatio(x.subCategory1, x.subCategory2), axis=1)
    df_pairs['subcat_ratio'] = df_pairs.apply(lambda x: ratio(x.subCategory1, x.subCategory2), axis=1)
    df_pairs['subcat_token_set_ratio'] = df_pairs.apply(lambda x: token_set_ratio(x.subCategory1, x.subCategory2), axis=1)
    
    
    drop_column = ['direction1','direction2', 'name1_number', 'name2_number']
    df_pairs.drop(drop_column, axis=1, inplace = True)

    return df_pairs


def create_edit_features_em(df_pairs):


    df_pairs = add_lat_lon_distance_features(df_pairs)

    df_pairs["direction1"] = df_pairs["sourceNames1"].apply(extract_directions)
    df_pairs["direction2"] = df_pairs["sourceNames2"].apply(extract_directions)
    df_pairs['Is_direction_match'] = df_pairs.apply(lambda x: is_direction_match(x.direction1, x.direction2), axis=1)

    df_pairs["name1_number"] = df_pairs["sourceNames1"].str.extract('(\d+)')
    df_pairs["name2_number"] = df_pairs["sourceNames2"].str.extract('(\d+)')
    df_pairs['Is_name_number_match'] = df_pairs.apply(lambda x: name_number_match(x.name1_number, x.name2_number), axis=1)

    df_pairs['Is_related_cat'] = np.vectorize(is_related_cat)(list(df_pairs["category1"]),list(df_pairs["category2"]))
    df_pairs['Is_category_match'] = np.vectorize(category_match)(list(df_pairs["category1"]),list(df_pairs["category2"]))
    df_pairs['Is_subcategory_match'] =  np.vectorize(sub_category_match)(list(df_pairs["subCategory1"]),list(df_pairs["subCategory2"]))

    df_pairs['Is_brand_match'] = np.vectorize(brand_match)(list(df_pairs["brands1"]),list(df_pairs["brands2"]))
    df_pairs['Is_house_match'] = np.vectorize(house_match)(list(df_pairs["houseNumber1"]),list(df_pairs["houseNumber2"]))
    df_pairs['Is_postal_match'] = np.vectorize(house_match)(list(df_pairs["postalCode1"]),list(df_pairs["postalCode2"]))
    df_pairs['is_phone_match'] = df_pairs.apply(lambda x: phone_category(x.phoneNumbers1, x.phoneNumbers2), axis=1)

    df_pairs['Is_email_match'] = np.vectorize(email_url_match)(list(df_pairs["email1"]),list(df_pairs["email2"]))
    df_pairs['Is_url_match'] = np.vectorize(email_url_match)(list(df_pairs["internet1"]),list(df_pairs["internet2"]))

    df_pairs['name_davies'] = np.vectorize(davies)(list(df_pairs["sourceNames1"]),list(df_pairs["sourceNames2"]))
    df_pairs['name_leven'] = np.vectorize(leven)(list(df_pairs["sourceNames1"]),list(df_pairs["sourceNames2"]))
    df_pairs['name_dice'] = np.vectorize(strike_a_match)(list(df_pairs["sourceNames1"]),list(df_pairs["sourceNames2"]))
    df_pairs['name_jaro'] = np.vectorize(jaro)(list(df_pairs["sourceNames1"]),list(df_pairs["sourceNames2"]))
    df_pairs['name_set_ratio'] = np.vectorize(token_set_ratio)(list(df_pairs["sourceNames1"]),list(df_pairs["sourceNames2"]))

    df_pairs['street_davies'] = np.vectorize(davies)(list(df_pairs["streets1"]),list(df_pairs["streets2"]))
    df_pairs['street_leven'] = np.vectorize(leven)(list(df_pairs["streets1"]),list(df_pairs["streets2"]))
    df_pairs['street_jaro'] = np.vectorize(jaro)(list(df_pairs["streets1"]),list(df_pairs["streets2"]))

    df_pairs['email_davies'] = np.vectorize(davies)(list(df_pairs["email1"]),list(df_pairs["email2"]))
    df_pairs['email_leven'] = np.vectorize(leven)(list(df_pairs["email1"]),list(df_pairs["email2"]))
    df_pairs['email_jaro'] = np.vectorize(jaro)(list(df_pairs["email1"]),list(df_pairs["email2"]))

    df_pairs['url_davies'] = np.vectorize(davies)(list(df_pairs["internet1"]),list(df_pairs["internet2"]))
    df_pairs['url_leven'] = np.vectorize(leven)(list(df_pairs["internet1"]),list(df_pairs["internet2"]))
    df_pairs['url_jaro'] = np.vectorize(jaro)(list(df_pairs["internet1"]),list(df_pairs["internet2"]))

    df_pairs['brands_davies'] = np.vectorize(davies)(list(df_pairs["brands1"]),list(df_pairs["brands2"]))
    df_pairs['brand_leven'] = np.vectorize(leven)(list(df_pairs["brands1"]),list(df_pairs["brands2"]))
    df_pairs['brand_jaro'] = np.vectorize(jaro)(list(df_pairs["brands1"]),list(df_pairs["brands2"]))

    df_pairs['phone_lcs'] = df_pairs.apply(lambda x: phone_lcs(x.phoneNumbers1, x.phoneNumbers2), axis=1)
    df_pairs['subcat_WRatio'] = np.vectorize(WRatio)(list(df_pairs["subCategory1"]),list(df_pairs["subCategory2"]))
    df_pairs['subcat_ratio'] = np.vectorize(ratio)(list(df_pairs["subCategory1"]),list(df_pairs["subCategory2"]))
    df_pairs['subcat_token_set_ratio'] = np.vectorize(token_set_ratio)(list(df_pairs["subCategory1"]),list(df_pairs["subCategory2"]))
    
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
