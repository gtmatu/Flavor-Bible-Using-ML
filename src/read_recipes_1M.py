import pandas as pd
import os


def main():
    df = pd.read_json(os.path.join("/Users/mateosarja/Documents/Uni/Year 4/FYP/Dataset Building/raw_recipes/1M.json"))



    df = df[['url', 'ingredients']]

    # print(df['ingredients'][0])

    ingredients = open(os.path.join("ingredients/1M_ingredients.txt"), 'w')
    keys = open(os.path.join("keys/1M_keys.csv"), 'w')

    for _, recipe in df.iterrows():
        url = recipe['url']
        for ingredient in recipe['ingredients'] :
            keys.write(url + '|' + ingredient['text'] +'\n')
            ingredients.write(ingredient['text'] + '\n')

if __name__ == '__main__':
    main()