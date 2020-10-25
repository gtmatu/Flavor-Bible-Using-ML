### Allrecipes.com contains 100'000+ copies of the same damn pizza recipe, 
### this script creates a list of all the recipe URLs for that pizza recipe

import pandas as pd
import numpy as np
import os

def main():

    df = pd.read_json(os.path.join("./dataset/raw_recipes/allrecipes.json"), lines=True)
    pizza_index = df[ df['author'] == 'The Kitchen at Johnsonville Sausage']['url']
    pizza_index.to_csv("./dataset/pizzas/pizzas.csv", index=False)









if __name__ == '__main__':
    main()