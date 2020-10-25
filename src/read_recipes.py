import pandas as pd
import os

def main():
    for source in ['epicurious', 'allrecipes', 'cookstr', 'bbc']
        df = pd.read_json(os.path.join("/Users/mateosarja/Documents/Uni/Year 4/FYP/Dataset Building/raw_recipes/{0}.json".format(source)), lines=True)

        df = df[['url', 'ingredients']]
        
        count = 0

        for index, recipe in df.iterrows():
            url = recipe['url']
            if not isinstance(recipe['ingredients'], list): continue
            count += 1

            if index % 52000 == 0: 
                file_num = int(index / 52000)
                ingredients = open(os.path.join("raw_ingredients/{0}_{1}_ingredients.txt".format(source, file_num)), 'w')
                keys = open(os.path.join("keys/{0}_{1}_keys.csv".format(source, file_num)), 'w')

            for ingredient in recipe['ingredients'] :
                keys.write(url + '|' + ingredient +'\n')
                ingredients.write(ingredient + '\n')
        print("Count: " + str(count))


if __name__ == '__main__':
    main()