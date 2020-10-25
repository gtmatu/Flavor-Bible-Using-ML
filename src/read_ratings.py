import pandas as pd 
import numpy as np
import os

def main():
    # for source in ["allrecipes", "epicurious", "cookstr"]:
    for source in ["allrecipes", "epicurious"]:
        try:
            df = pd.read_json(os.path.join("dataset/raw_recipes/{0}.json".format(source)), lines=True)
        except Exception as e:
            print(f'{e} @ {source}')

        if source == "allrecipes":
            df = df[['url', 'rating_stars', 'review_count']]
            df.columns = 'url', 'rating', 'review_count'

        if source == "epicurious":
            df = df[['url', 'aggregateRating', 'reviewsCount']]
            df.columns = 'url', 'rating', 'review_count'
        df.to_csv(f'dataset/ratings/{source}_ratings.csv', index=False)
        # if source == "cookstr":
        #     df = df[['url', 'rating_value']]
        




if __name__ == '__main__':
    main()