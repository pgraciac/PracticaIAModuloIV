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
        subdf = df[df[attribute] == val][df.columns[-1]]
        pond = len(subdf) / len(df)
        ent += pond * entropy(subdf)
    return ent


def best_attribute(df, global_entropy):
    best_gain = 0
    best_attr = ''
    for key in df.iloc[:, :-1].keys():
        key_gain = gain(df, key, global_entropy)
        if key_gain > best_gain:
            best_gain = key_gain
            best_attr = key
    return best_attr, best_gain


def gain(df, attribute, global_entropy):
    res = global_entropy - attribute_entropy(df, attribute)
    #print(global_entropy, "-", attribute_entropy(df, attribute), "=", res)
    return res

def ID3(df, global_entropy):
    unique_vals = np.unique(df[df.columns[-1]])
    if len(unique_vals) == 1:
        #print("Todos iguales", unique_vals)
        #print(df)
        return unique_vals[0]
    elif len(df.keys()) <= 1:
        # print("Último atributo")
        # print(df)
        return np.argmax(np.unique(df[df.columns[-1]], return_counts=True)[1])
    if len(df.keys()) > 1:
        # print(df)
        best_attr, best_gain = best_attribute(df, global_entropy)
        # print("Mejor atributo y ganancia es", best_attr, best_gain)
        tree = {best_attr: {}}
        for val in np.unique(df[best_attr]):
            sub_data = df.where(df[best_attr] == val).dropna().drop([best_attr], axis=1)
            subtree = ID3(sub_data, global_entropy)
            tree[best_attr][val] = subtree

        return tree


def predict(observation, tree):
    for nodes in tree.keys():
        value = observation[nodes]
        tree = tree[nodes][value]
        prediction = 0

        if type(tree) is dict:
            prediction = predict(observation, tree)
        else:
            prediction = tree
            break

    return prediction


if __name__ == '__main__':
    #Ejercicio 1
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
    print(f"1.c)\nLa ganancia de EsSuave es: {gain(df_train, 'EsSuave', EsVenenosa_entropy)}")

    tree = ID3(df_train, EsVenenosa_entropy)
    print(f"1.d)\nEl arbol generado por el algoritmo es: {tree}")
    predictions = df_test.apply(predict, axis=1, args=(tree,))
    print(f"1.e)\nLas predicciones para las setas U, V y W son:\n{predictions}")

    # EJERCICIO 2
    df_train = pd.DataFrame({'x1': [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                             'x2': [0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1],
                             'x3': [1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1],
                             'x4': [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1],
                             'x5': [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0],
                             'x6': [0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
                             'T': [1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0]},
                            index=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13',
                                   'A14'])
    T_entropy = entropy(df_train['T'])
    print(f"1.b)\nEl arbol generado por el algoritmo es: {ID3(df_train, T_entropy)}")
