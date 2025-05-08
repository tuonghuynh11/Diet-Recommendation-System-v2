import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from ImageFinder import get_images_links as find_image

import ast
import pandas as pd

import redis
import pickle
import hashlib
import json
import time
import threading
import redis.exceptions

CACHE_EXPIRE_SECONDS = 3600 # 1 hour
REFRESH_THRESHOLD_SECONDS = 300 

pool = redis.ConnectionPool(
    host='redis-13116.c1.ap-southeast-1-1.ec2.redns.redis-cloud.com',
    port=13116,
    username='default',
    password='mx3EPshCQtweT1lvrlSO4NeesRtAyzso',
    decode_responses=False,  # vì dùng pickle
    max_connections=20       # giới hạn số connection mở ra cùng lúc
)

r = redis.Redis(connection_pool=pool)

def make_cache_key(_input, ingredients, params):
    key_source = {
        "input": _input,
        "ingredients": ingredients,
        "params": params
    }
    key_str = json.dumps(key_source, sort_keys=True)
    key_hash = hashlib.md5(key_str.encode()).hexdigest()
    return f"recommendation:{key_hash}"

def invalidate_cache(_input, ingredients, params):
    cache_key = make_cache_key(_input, ingredients, params)
    r.delete(cache_key)
    print(f"[CACHE] Deleted: {cache_key}")

def parse_ingredient_parts(s):
    if pd.isna(s):
        return []
    s = s.strip()
    if s.startswith('c(') and s.endswith(')'):
        s = s[2:-1]  # bỏ 'c(' và ')'
    try:
        parts = ast.literal_eval(f'[{s}]')  # chuyển thành list
        return parts
    except Exception:
        return []

def scaling(dataframe):
    scaler=StandardScaler()
    prep_data=scaler.fit_transform(dataframe.iloc[:,8:17].to_numpy())
    return prep_data,scaler

def nn_predictor(prep_data):
    neigh = NearestNeighbors(metric='cosine',algorithm='brute')
    neigh.fit(prep_data)
    return neigh

def build_pipeline(neigh,scaler,params):
    transformer = FunctionTransformer(neigh.kneighbors,kw_args=params)
    pipeline=Pipeline([('std_scaler',scaler),('NN',transformer)])
    return pipeline

def apply_pipeline(pipeline,_input,extracted_data):
    _input=np.array(_input).reshape(1,-1)
    return extracted_data.iloc[pipeline.transform(_input)[0]]

# def extract_data(dataframe,ingredients):
#     extracted_data=dataframe.copy()
#     extracted_data=extract_ingredient_filtered_data(extracted_data,ingredients)
#     return extracted_data


# def extract_ingredient_filtered_data(dataframe,ingredients):
#     extracted_data=dataframe.copy()
#     regex_string=''.join(map(lambda x:f'(?=.*{x})',ingredients))
#     extracted_data=extracted_data[extracted_data['RecipeIngredientParts'].str.contains(regex_string,regex=True,flags=re.IGNORECASE)]
#     return extracted_data

# def recommend(dataframe,_input,ingredients=[],params={'n_neighbors':5,'return_distance':False}):
#         extracted_data=extract_data(dataframe,ingredients)
#         if extracted_data.shape[0]>=params['n_neighbors']:
#             prep_data,scaler=scaling(extracted_data)
#             neigh=nn_predictor(prep_data)
#             pipeline=build_pipeline(neigh,scaler,params)
#             return apply_pipeline(pipeline,_input,extracted_data)
#         else:
#             return None


######################################## New #####################################

def extract_data(dataframe, ingredients, target_calories=None, calories_tolerance=0.2):
    extracted_data = dataframe.copy()

    # Lọc nguyên liệu trước
    extracted_data = extract_ingredient_filtered_data(extracted_data, ingredients)

    # Lọc calories nếu có yêu cầu
    if target_calories is not None:
        min_calories = target_calories * (1 - calories_tolerance)
        max_calories = target_calories * (1 + calories_tolerance)
        extracted_data = extracted_data[
            (extracted_data['Calories'] >= min_calories) &
            (extracted_data['Calories'] <= max_calories)
        ]

    return extracted_data

def extract_ingredient_filtered_data(dataframe, ingredients):
    extracted_data = dataframe.copy()

    # Bước 1: Parse RecipeIngredientParts thành list
    extracted_data['ParsedIngredientParts'] = extracted_data['RecipeIngredientParts'].apply(parse_ingredient_parts)

    # Bước 2: Check dòng nào chứa tất cả các nguyên liệu yêu cầu
    # def contains_all(parts):
    #     return all(
    #         any(ingredient.lower() in p.lower() for p in parts)
    #         for ingredient in ingredients
    #     )
    def contains_all(parts):
        # Chuyển tất cả các phần thành lowercase và tách thành các từ riêng biệt
        parts_lower = [p.lower() for p in parts]
        
        # Kiểm tra từng nguyên liệu phải là một từ riêng biệt trong danh sách
        return all(
            any(ingredient.lower() == p for p in parts_lower)  # Dùng == thay vì in
            for ingredient in ingredients
    )

    extracted_data = extracted_data[extracted_data['ParsedIngredientParts'].apply(contains_all)]
    return extracted_data

def recommend(dataframe, _input, ingredients=[], params={'n_neighbors':5, 'return_distance':False}):
    cache_key = make_cache_key(_input, ingredients, params)

    # --- Try get cache ---
    try:
        cached = r.get(cache_key)
        if cached:
            print(f"[CACHE] Found: {cache_key}")
            return pickle.loads(cached)
    except Exception as e:
        print(f"[CACHE] Redis get failed: {str(e)}")

    target_calories = _input[0]
    extracted_data = extract_data(dataframe, ingredients, target_calories=target_calories)

    if extracted_data.shape[0] >= params['n_neighbors']:
        prep_data, scaler = scaling(extracted_data)
        neigh = nn_predictor(prep_data)
        pipeline = build_pipeline(neigh, scaler, params)
        result = apply_pipeline(pipeline, _input, extracted_data)

        # --- Try set cache ---
        try:
            r.set(cache_key, pickle.dumps(result), ex=3600)
            print(f"[CACHE] Set: {cache_key}")
        except Exception as e:
            print(f"[CACHE] Redis set failed: {str(e)}")

        return result
    else:
        return None
######################################## New #####################################
def extract_quoted_strings(s):
    # Find all the strings inside double quotes
    strings = re.findall(r'"([^"]*)"', s)
    # Join the strings with 'and'
    return strings

def output_recommended_recipes(dataframe):
    if dataframe is not None:
        output=dataframe.copy()
        output=output.to_dict("records")
        for recipe in output:
            # Chuyển các trường thời gian thành string
            recipe['CookTime'] = str(recipe['CookTime']) if recipe.get('CookTime') is not None else ""
            recipe['PrepTime'] = str(recipe['PrepTime']) if recipe.get('PrepTime') is not None else ""
            recipe['TotalTime'] = str(recipe['TotalTime']) if recipe.get('TotalTime') is not None else ""
            recipe['Image']=find_image(recipe['Name'])

            recipe['RecipeIngredientQuantities']=extract_quoted_strings(recipe['RecipeIngredientQuantities'])
            recipe['RecipeIngredientParts']=extract_quoted_strings(recipe['RecipeIngredientParts'])
            recipe['RecipeInstructions']=extract_quoted_strings(recipe['RecipeInstructions'])
    else:
        output=None
    return output
