import math

import numpy as np
import pandas as pd


def entropy(column):
    vals, cont_vals = np.unique(column, return_counts=True)
    probabilities = cont_vals / len(column)
    ent = -np.sum(probabilities * np.log2(probabilities))
    return ent


def attribute_entropy(df, attribute):
    uniq_vals = df[attribute].unique()
    ent = 0
    for val in uniq_vals:
        subdf = df[df[attribute] == val]['EsVenenosa']
        pond = len(subdf) / len(df)
        ent += pond * entropy(subdf)
    return ent


def best_attribute(df):
    best_gain = 0
    best_attr = ''
    for key in df.iloc[:, :-1].keys():
        key_gain = gain(df, key)
        if key_gain > best_gain:
            best_gain = key_gain
            best_attr = key
    return best_attr, best_gain


def gain(df, attribute):
    return entropy(df['EsVenenosa']) - attribute_entropy(df, attribute)


# def ID3(df, original_df, features):
#     unique_vals = np.unique(df['EsVenenosa'])
#     if len(unique_vals) == 1:
#         return unique_vals[0]
#
#     elif len(features) == 0:
#         return np.argmax(np.unique(original_df['EsVenenosa'], return_counts=True)[1])
#
#     else:
#         best_feature, gain = best_attribute(df)
#         features = [i for i in features if i != best_feature]
#         tree = {best_feature: {}}
#
#         for value in np.unique(df[best_feature]):
#             sub_data = df.where(df[best_feature] == value).dropna()
#             subtree = ID3(sub_data, df, features)
#             tree[best_feature][value] = subtree
#
#         return tree

def ID3(df):
    unique_vals = np.unique(df['EsVenenosa'])
    if len(unique_vals) == 1:
        return unique_vals[0]
    elif len(df.keys()) <= 1:
        return np.argmax(np.unique(df['EsVenenosa'], return_counts=True)[1])
    if len(df.keys()) > 1:
        best_attr = best_attribute(df)[0]
        tree = {best_attr: {}}
        for val in np.unique(df[best_attr]):
            sub_data = df.where(df[best_attr] == val).dropna().drop([best_attr], axis=1)
            subtree = ID3(sub_data)
            tree[best_attr][val] = subtree

        return tree

if __name__ == '__main__':
    df_train = pd.DataFrame({'PesaMucho': [0, 0, 1, 1, 0, 0, 0, 1],
                             'EsMaloliente': [0, 0, 1, 0, 1, 0, 0, 1],
                             'EsConManchas': [0, 1, 0, 0, 1, 1, 0, 0],
                             'EsSuave': [0, 0, 1, 1, 0, 1, 1, 0],
                             'EsVenenosa': [0, 0, 0, 1, 1, 1, 1, 1]},
                            index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
    df_test = pd.DataFrame({'PesaMucho': [1, 0, 1],
                            'EsMaloliente': [1, 1, 1],
                            'EsConManchas': [1, 0, 0],
                            'EsSuave': [1, 1, 0]},
                           index=['U', 'V', 'W'])
    EsVenenosa_entropy = entropy(df_train['EsVenenosa'])
    print(f"1.a)\nLa entropia de EsVenenosa es: {EsVenenosa_entropy}")
    print(attribute_entropy(df_train, 'EsSuave'))
    print(best_attribute(df_train))

    features = df_train.columns[:-1].tolist()
    tree = ID3(df_train)
    print(tree)
