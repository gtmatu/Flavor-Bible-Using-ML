import pandas as pd

def main():
    for source in ['bbc', 'cookstr', 'epicurious', 'allrecipes_0', 'allrecipes_1', 'allrecipes_2', 'allrecipes_3', 'allrecipes_4',  '1M']:
        try:
            parsed_df = pd.read_json("dataset/parsed_ingredients/{0}_parsed_ingredients.json".format(source))
            parsed_df.drop(['comment', 'display', 'qty', 'unit', 'other', 'range_end'], axis=1, inplace=True)
            parsed_df.rename({"input":"ingredient", "name":"parsed_ingredient"}, axis=1, inplace=True)
            parsed_df.drop_duplicates(inplace=True)
            print(f'Parsed : {source}')
        except Exception as e:
            print(f'{e} happening at {source}')
            continue
        
        try:
            keys_df = pd.read_csv("dataset/keys/{0}_keys.csv".format(source), delimiter="|", engine='python', quotechar='"')
            print(f'Keys : {source}')
        except Exception as e:
            print(f'{e} happening at {source}')
            continue

        joined_df = pd.merge(left=keys_df, right=parsed_df, how='left', left_on='ingredient', right_on='ingredient')
        joined_df.to_csv("dataset/{}.csv".format(source), sep='|')


if __name__ == '__main__':
    main()