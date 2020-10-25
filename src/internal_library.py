from tabulate import tabulate
from sklearn.manifold import MDS
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sklearn
import math
import os
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from itertools import combinations
from statistics import mean, stdev, median
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD

RANDOM_SEED = 42

def import_recipes(num_ingredients):
    #### Import Cleaned Recipes 
    # Args: 
    #     - num_ingredients, number of ingredients to keep

    # Returns:
    #     - dataframe: 'url', 'ingredient', 'source'


    ### Import Raw Recipes  -- pizzas deleted
    sources = ['allrecipes_0', 'allrecipes_1', 'allrecipes_2', 'allrecipes_3', 'allrecipes_4', '1M', 'bbc', 'epicurious', 'cookstr']
    # sources = ['1M']
    DATASET_DIR = './dataset'

    df = pd.DataFrame()
    pizza_filter = pd.read_csv(os.path.join(DATASET_DIR, 'pizzas/pizzas.csv'), names=['url'])
    # print(pizza_filter.keys)

    for source in sources:
        csv_path = os.path.join(DATASET_DIR, f'{source}.csv')
        try:
            one_source = pd.read_csv(csv_path, sep='|', engine='python',index_col=0)
        except Exception as e:
            print(f'{e} encountered at {source}')
            continue
            
        if source.startswith('allrecipes'):
            source = 'allrecipes'
            one_source = one_source[~one_source.url.isin(pizza_filter['url'])]
        
        one_source['source'] = source
            
        try:
            df = df.append(one_source)
        except Exception as e:
            print(f'{e} encountered at {source}')
            continue


    ## Clean characters
    spec_chars = ["!",'"',"#","%","&","'","(",")",
            "*","+",",","-",".","/",":",";","<",
            "=",">","?","@","[","\\","]","^","_",
            "`","{","|","}","~","–", "grams ", "oz ", 
            "ounces ", "ounce ", "tbsp ", "tsp ", "pkg ", 
            "package ", "packages ", "fl ", "gms ", "jar ", "can ", "ml ", "[0-9]"]

    df['parsed_ingredient'] = df['parsed_ingredient'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    df['parsed_ingredient'] = df['parsed_ingredient'].str.lower()

    for char in spec_chars:
        df['parsed_ingredient'] = df['parsed_ingredient'].str.replace(char, ' ')
        
    df['parsed_ingredient'] = df['parsed_ingredient'].str.split().str.join(" ")

    csv_path = os.path.join(DATASET_DIR, 'final_ingredients/mapped_names.csv')

    try:
        ing_map = pd.read_csv(csv_path, sep=',', engine='python')
    except Exception as e:
        print(f'{e} encountered at {source}')
    ing_map = ing_map.loc[ing_map['mapped_ingredient'] != 'NO INGREDIENT']

    ing_map = dict(zip(ing_map['parsed_ingredient'], ing_map['mapped_ingredient']))
    df['ingredient'] = df['parsed_ingredient'].map(ing_map)
    ingredients = df.dropna()
    # ingredients = ingredients.drop(['parsed_ingredient'], axis=1)
    ingredients.drop_duplicates(inplace=True)
    counts = ingredients['ingredient'].value_counts()
    ing_to_keep = counts[:num_ingredients].index
    ingredients = ingredients[ingredients['ingredient'].isin(ing_to_keep)]

    return ingredients




def import_rated_recipes(num_ingredients):
    #### Import Rated Recipes 
    # Args: 
    #     - num_ingredients, number of ingredients to keep
    #     - ingredients df, output of import_recipes()

    # Returns:
    #     - recipes, df: 'url', 'ingredient_vector'
    #     - ingredients, df: 'url', 'ingredient', 'source'
    #     - ratings, df: 'url', 'rating', 'review_count'


    ingredients = import_recipes(num_ingredients)

    sources = ['allrecipes', 'epicurious']
    DATASET_DIR = './dataset'

    ratings = pd.DataFrame()
    pizza_filter = pd.read_csv(os.path.join(DATASET_DIR, 'pizzas/pizzas.csv'), names=['url'])

    for source in sources:
        one_source = pd.read_csv(os.path.join(DATASET_DIR, f'ratings/{source}_ratings.csv'))
        if source == 'allrecipes':
            one_source = one_source[~one_source.url.isin(pizza_filter['url'])]
            
        try:
            ratings = ratings.append(one_source)
        except Exception as e:
            print(f'{e} encountered at {source}')
            continue
    ratings = ratings.replace([0.0], np.nan).dropna()   # Remove recipes that weren't rated
    # ratings = ratings[ratings['review_count']>10]   # Recipe was reviewed by at least 10 people
    ratings = ratings[ratings.url.isin(ingredients.url)] # Remove rated recipes that do not appear in the final dataset
    ratings.set_index('url', drop=True, inplace=True)

    rated_recipes = ingredients[ingredients.url.isin(ratings.index)].drop('source', axis=1)
    rated_recipes = rated_recipes.groupby('url')['ingredient'].value_counts().unstack().fillna(0)

    ingredient_list = np.sort(ingredients['ingredient'].unique())
    rated_ingredients = rated_recipes.columns
    missing_ingredients = set(ingredient_list) - set(rated_ingredients)

    csv_path = os.path.join(DATASET_DIR, 'final_ingredients/mapped_names.csv')
    try:
        ing_map = pd.read_csv(csv_path, sep=',', engine='python')
    except Exception as e:
        print(f'{e} encountered at {source}')


    ing_map = ing_map[~ing_map['mapped_ingredient'].isin(missing_ingredients)]
    ing_map = ing_map.loc[ing_map['mapped_ingredient'] != 'NO INGREDIENT']
    ing_map = dict(zip(ing_map['parsed_ingredient'], ing_map['mapped_ingredient']))

    ingredients['ingredient'] = ingredients['parsed_ingredient'].map(ing_map)
    ingredients = ingredients.dropna()
    ingredients = ingredients.drop(['parsed_ingredient'], axis=1)
    ingredients.drop_duplicates(inplace=True)

    counts = ingredients['ingredient'].value_counts()
    ing_to_keep = counts[:num_ingredients].index
    ingredients = ingredients[ingredients['ingredient'].isin(ing_to_keep)]

    rated_recipes = ingredients[ingredients.url.isin(ratings.index)].drop('source', axis=1)
    rated_recipes = rated_recipes.groupby('url')['ingredient'].value_counts().unstack().fillna(0)
    recipes = ingredients.groupby('url')['ingredient'].value_counts().unstack().fillna(0)

    return recipes, ingredients, ratings


def build_edges(ingredients):
    #### Build Edge List and Edge Matrix
    # Args: 
    #     - ingredients df, output of import_rated_recipes()/import_recipes()

    # Returns:
    #     - edge_list, df: 'ingredient_x', 'ingredient_y', 'pair_freq'
    #     - edge_matrix, df: same info but in matrix rather than series

    tmp = ingredients.drop('source', axis=1)
    tmp = pd.merge(tmp, tmp, on='url')
    tmp = tmp[tmp['ingredient_x'] != tmp['ingredient_y']]
    tmp.drop('url', axis=1, inplace=True)
    edge_matrix = tmp.groupby(['ingredient_x', 'ingredient_y']).size().unstack().fillna(0)   # Dissimilarity matrix

    tmp = tmp.groupby(['ingredient_x', 'ingredient_y']).size().to_frame('pair_freq').reset_index()
    filter_dups = pd.DataFrame(np.sort(tmp[['ingredient_x','ingredient_y']], axis=1))
    edge_list = tmp[~filter_dups.duplicated()].sort_values(by='pair_freq', ascending=False)
    edge_list = edge_list.reset_index(drop=True)     # Edge List

    return edge_list, edge_matrix

def get_feature_matrix(edge_matrix, feature, recipes_per_ingredient=None):
    #### Build Edge List and Edge Matrix
    # Args: 
    #     - edge_matrix, df
    #     - log distance, PMI or IOU

    # Returns:
    #     - Feature matrix, df: 2D -- set up as dissimilarity matrix

    
    feature_matrix = pd.DataFrame()
    total_num_recipes = 194709

    if feature == 'PMI':
        for x in edge_matrix.index:
            for y in edge_matrix.index:
                scale = 700
                # scale = 1
                numerator = edge_matrix.loc[x,y] / total_num_recipes
                denominator = (recipes_per_ingredient[x] * recipes_per_ingredient[y]) / (total_num_recipes**2)
                if denominator == 0:
                    denominator = 1
                pmi = numerator / denominator
                # feature_matrix.loc[x,y] = pmi
                feature_matrix.loc[x,y] = np.log(scale * pmi)     # Scale to keep all distances positive

        feature_matrix = feature_matrix.replace( -np.inf, np.nan)

        min_val = feature_matrix.min().min()             
        max_val = feature_matrix.max().max() 

        feature_matrix = feature_matrix.replace( np.nan, min_val)    # Replace inf with nan
        feature_matrix = feature_matrix.apply(lambda x: max_val - x)  # Flip similarity to dissimilarity
                


    elif feature == 'IOU':
        for x in edge_matrix.index:
            for y in edge_matrix.index:
                numerator = edge_matrix.loc[x,y] 
                denominator = recipes_per_ingredient[x] + recipes_per_ingredient[y] - edge_matrix.loc[x,y]
                if denominator == 0:
                    denominator = 1
                iou = numerator / denominator
                feature_matrix.loc[x,y] = iou   

        feature_matrix = feature_matrix.replace( -np.inf, np.nan)
        max_val = feature_matrix.max().max()             # Flip similarity to dissimilarity

        feature_matrix = feature_matrix.replace( np.nan, max_val)    # Replace inf with nan
        feature_matrix = feature_matrix.apply(lambda x: 1 - x)



    elif feature == 'log_distance' :
        scale = 26600
        feature_matrix = edge_matrix.apply(lambda x: np.log(scale/x) ) # Scale to keep all distances positive

        feature_matrix = feature_matrix.replace( np.inf, np.nan)
        max_val = feature_matrix.max().max() 
        # feature_matrix = feature_matrix.replace( np.nan, max_val * 10)    # Replace inf with nan
        feature_matrix = feature_matrix.replace( np.nan, max_val * 1.2)    # Replace inf with nan

    for i in feature_matrix.index:
        feature_matrix.loc[i,i] = 0

    return feature_matrix

def cluster(X, num_clusters, cluster_type, plot=False):
    #### Build Edge List and Edge Matrix
    # Args: 
    #     - MDS ouput, df: 'ingredient', 'X', 'Y'
    #     - number of clusters
    #     - AgglomerativeClustering 'agg' or KMeans 'km'
    #     - bool to plot clusters or not

    # Returns:
    #     - MDS ouput, df: 'ingredient', 'X', 'Y', 'cluster'
    if cluster_type == 'agg':
        model = AgglomerativeClustering(num_clusters)
    elif cluster_type == 'km':
        model = KMeans(num_clusters)
    clusters = model.fit_predict(X)
    X['cluster'] = clusters

    if plot:
        plt.figure(figsize=(9,9))
        plt.scatter(X.loc[:, 0], X.loc[:, 1], c=clusters, s=50, cmap='viridis')

        # centers = model.cluster_centers_
        # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        plt.show()
    
    return X

def normalize_columns(X):

    idx = X.index
    x = X.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    X = pd.DataFrame(x_scaled)
    X.index = idx

    return X



def calc_distance_row(recipe, ing_map, distance_fn):
    if recipe.sum() < 3:
        return pd.Series({'mean_dist':0, 
                      'stdev': 0, 
                      'median_dist': 0})
    ing_present = recipe[recipe == 1].index.tolist()
    pairs = combinations(ing_present, 2)

    distances = []

    for idx, pair in enumerate(pairs):
        mds_ing_1 = ing_map.loc[pair[0]].values.reshape(1,-1)
        mds_ing_2 = ing_map.loc[pair[1]].values.reshape(1,-1)
        distances.append(distance_fn(mds_ing_1, mds_ing_2)[0][0])
        
    return pd.Series({'mean_dist':mean(distances), 
                      'stdev': stdev(distances), 
                      'median_dist': median(distances)})

def calc_distances(num_recipes, recipes, ing_map, distance_fn):
    X = pd.DataFrame()

    step = num_recipes // 20
    for i in tqdm(range(0, num_recipes, step)):
        tmp = recipes.iloc[i : i+step]

        tmp2 = tmp.apply(calc_distance_row, ing_map=ing_map, distance_fn=distance_fn, axis=1)

        X = pd.concat([X, tmp2], sort=True)
    return X[X['mean_dist']>0]


def get_neighbors(X, ingredient_list, n_neighbors=10, plot=False):
    dist_columns = []
    neigh_columns = []
    for i in range(1, n_neighbors+1):
        dist_columns.append(f'distance_{i}')
        neigh_columns.append(f'neighbor_{i}')
        
    model = NearestNeighbors(n_neighbors+1)  
    model.fit(X)
    distances, neighbors = model.kneighbors(X)
    
    distances = pd.DataFrame(distances)
    distances.drop(0, axis=1, inplace=True)
    distances.columns = dist_columns
    neighbors = pd.DataFrame(neighbors)
    neighbors.drop(0, axis=1, inplace=True)
    neighbors.columns = neigh_columns
    neighbors = neighbors.apply(lambda x: ingredient_list[x])
    
    df = pd.DataFrame(pd.concat([neighbors, distances], axis=1))
    df.index = X.index

    if plot:
        plt.figure()
        plt.hist(distances.values.flatten(), bins=100)
        plt.ylabel('Frequency')
        plt.xlabel('Distance')
        plt.show()
    
    return df

def get_nearest_points(X, radius):
    dist_columns = []
    neigh_columns = []
    for i in range(1, n_neighbors+1):
        dist_columns.append(f'distance_{i}')
        neigh_columns.append(f'neighbor_{i}')
        
    model = NearestNeighbors()  
    model.fit(X)
    distances, neighbors = model.radius_neighbors(X, radius)
    
    distances = pd.DataFrame(distances)
    distances.drop(0, axis=1, inplace=True)
    distances.columns = dist_columns
    neighbors = pd.DataFrame(neighbors)
    neighbors.drop(0, axis=1, inplace=True)
    neighbors.columns = neigh_columns
    neighbors = neighbors.apply(lambda x: ingredient_list[x])
    
    df = pd.DataFrame(pd.concat([neighbors, distances], axis=1))
    df.index = X.index
    
    return df

def find_pairs(X, radius, edge_matrix, ingredient_list, drop_existing=True, threshold=0):
        
    model = NearestNeighbors()  
    model.fit(X)
    distances, neighbors = model.radius_neighbors(radius=radius)
    
    out = pd.DataFrame()
    idx = 0
    
    for x, neighbor_list in enumerate(neighbors):
        for y, neighbor in enumerate(neighbor_list):
            series = pd.Series({'ingredient_x': ingredient_list[x], 
                                'ingredient_y': ingredient_list[neighbor], 
                                'distance': distances[x][y]
                               })
            out[idx] = series
            idx += 1
    out = out.T
    
    filter_dups = pd.DataFrame(np.sort(out[['ingredient_x','ingredient_y']], axis=1))
    out = out[~filter_dups.duplicated()].sort_values('distance').reset_index(drop=True)
    
    if drop_existing:
        print('Filtering...')
        out['pair_freq'] = out.apply(remove_existing_pairs, edge_matrix=edge_matrix, axis=1)
        out = out[out['pair_freq'] <= threshold]
        
    return out.reset_index(drop=True)

def remove_existing_pairs(X, edge_matrix):
    ing_x = X['ingredient_x']
    ing_y = X['ingredient_y']
    
    return edge_matrix.loc[ing_x, ing_y]

def one_out_list(ing_x_list, ing_y_list, feature_matrix, n_components, ingredient_list, mode='MDS'):
    assert len(ing_x_list)==len(ing_y_list)

    out = pd.DataFrame()

    for idx, ing_x in tqdm(enumerate(ing_x_list)):
        ing_y = ing_y_list[idx]
        if mode=='MDS':
            neighbor, distance = one_out(ing_x, ing_y, feature_matrix, n_components, ingredient_list)
        else:
            neighbor, distance = one_out(ing_x, ing_y, feature_matrix, n_components, ingredient_list, mode='LSA')
        out[idx] = pd.Series({'ingredient_x': ing_x, 
                            'ingredient_y': ing_y, 
                            'distance': distance,
                            'neighbor': neighbor, 
                            'og_neighbor' : int(idx/len(ingredient_list)) + 1
                            })

    return out.T


def one_out(ing_x, ing_y, feature_matrix, n_components, ingredient_list, mode='MDS'):
    df = feature_matrix

    if mode=='MDS':
        df.at[ing_x, ing_y] =  feature_matrix.max().max()
        df.at[ing_y, ing_x] =  feature_matrix.max().max()
        mds_out = mds(df, n_components)

        
    else:
        tmp = df.loc[[ing_x, ing_y],:]
        tmp = tmp.loc[ing_x].mask(tmp.sum()==2, 0)
        tmp = tmp.loc[ing_y].mask(tmp.sum()==2, 0)

        df.loc[ing_x, :] = tmp.loc[ing_x]
        df.loc[ing_y, :] = tmp.loc[ing_y]

        mds_out = lsa(df, n_components, ingredient_list)

    df = get_neighbors(mds_out, ingredient_list)
    df = df.loc[[ing_x, ing_y], :]

    return is_in_neighbors(df, ing_x, ing_y)

def is_in_neighbors(df, ing_x, ing_y):
    x = np.array(df.loc[ing_x, :].to_list()[:10])
    try:
        y = np.argwhere(x==ing_y)[0][0]+1
        n_neighbor = df.iloc[:, y+10].name
        if n_neighbor.endswith('10'):
            return n_neighbor[-2], df.iloc[0, y+10]
        else :
            return n_neighbor[-1], df.iloc[0, y+10]
    except Exception as e:
        return np.nan, np.nan


def mds(feature_matrix, n_components):
    model = MDS(n_components=n_components, 
               metric=True, 
               n_init=10, 
               max_iter=1000,
               random_state=RANDOM_SEED, 
               dissimilarity='precomputed')

    out = model.fit_transform(feature_matrix)
    out = pd.DataFrame(out)
    out.index = feature_matrix.index
    
    return out

def lsa(feature_matrix, n_components, ingredient_list):
    model = TruncatedSVD(n_components, algorithm = 'arpack')

    lsa_out = model.fit_transform(feature_matrix)
    lsa_out_df = pd.DataFrame(lsa_out)

    lsa_out_df.index = ingredient_list
    
    return out