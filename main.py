from sent2vec import Sentence2Vec
import pandas as pd

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model = Sentence2Vec('./ready.model')
    #test for similarity
    print(model.similarity('Nike SB Dunk Low Pro Bucks | Size 10',
                           'adidas Yeezy Boost 350 V2 Carbon | Size 10.5'))

    #find similarity models in dataset
    df = pd.read_csv('./dataset.csv', encoding='ISO-8859-1', index_col=0)
    df = df.drop_duplicates(subset='Model')
    titles = df['Model'].values.tolist()
    line = 'Nike SB Dunk Low Pro Bucks | Size 10'
    count = 0
    for i in titles:
        res = model.similarity(line, i)
        if res > 0.75:
            print(i)
            count += 1
            if count == 100:
                break

