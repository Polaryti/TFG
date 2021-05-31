import os
import pandas as pd
import csv
from cleaner import clean_all, clean_no_stopwords

if __name__ == "__main__":
    # Generació del dataset sense stopwords
    sample_count = 0
    df_res = pd.DataFrame()
    cont = 0
    void_labels = 0
    for path, subdirs, files in os.walk('data/TextClassification/corpus'):
        for name in files:
            cont += 1
            df = pd.read_excel(os.path.join(path, name))
            sample_count += len(df)
            df = df[['Description', 'Classificació']]

            antes = len(df)
            df.dropna(subset=['Classificació'], inplace=True)
            void_labels = antes - len(df)

            df['Description'] = df['Description'].apply(clean_all)
            df.dropna(subset=['Description'], inplace=True)
            df.drop_duplicates(['Description'], inplace=True)

            df_res = pd.concat([df_res, df], copy=False)
            print(f'Processat fitxer \"{name}\"')

    aux = set()
    aux = df_res['Classificació'].unique()

    print("Mostres sense etiquetar: {}".format(void_labels))
    print("Mostres abans del preprocessament: {}".format(sample_count))
    print("Mostres després del preprocessament (sense eliminar les de més d'una classe): {}".format(len(df_res)))
    print("Nombre total de clases úniques (sense eliminar les de més d'una classe): {}".format(len(aux)))

    with open('res/clases_corpus.csv', 'w', encoding="utf-8") as write_file:
        for clase in aux:
            write_file.write("{}\n".format(clase))

    multiple_label = 0
    with open('res/corpus_noStopwords.csv', 'w', encoding="utf-8", newline='') as w_file:
        writer = csv.writer(w_file)
        writer.writerow(['Description', 'Classificació'])
        for index, row in df_res.iterrows():
            aux = row['Classificació'].split('|')
            if len(aux) == 1:
                writer.writerow([row['Description'], row['Classificació'].strip()])
            else:
                multiple_label += 1

    print("Mostres amb més d'una classe: {}".format(multiple_label))

    # Generació del dataset amb stopwords
    sample_count = 0
    cont = 0
    df_res = pd.DataFrame()
    void_labels = 0
    for path, subdirs, files in os.walk('data/TextClassification/corpus'):
        for name in files:
            cont += 1
            df = pd.read_excel(os.path.join(path, name))
            sample_count += len(df)
            df = df[['Description', 'Classificació']]

            antes = len(df)
            df.dropna(subset=['Classificació'], inplace=True)
            void_labels = antes - len(df)

            df['Description'] = df['Description'].apply(clean_no_stopwords)
            df.dropna(subset=['Description'], inplace=True)
            df.drop_duplicates(['Description'], inplace=True)

            df_res = pd.concat([df_res, df], copy=False)
            print(f'Processat fitxer \"{name}\"')

    aux = set()
    aux = df_res['Classificació'].unique()

    print("Mostres sense etiquetar: {}".format(void_labels))
    print("Mostres abans del preprocessament: {}".format(sample_count))
    print("Mostres després del preprocessament (sense eliminar les de més d'una classe): {}".format(len(df_res)))
    print("Nombre total de clases úniques (sense eliminar les de més d'una classe): {}".format(len(aux)))

    multiple_label = 0
    with open('res/corpus_ambStopwords.csv', 'w', encoding="utf-8", newline='') as w_file:
        writer = csv.writer(w_file)
        writer.writerow(['Description', 'Classificació'])
        for index, row in df_res.iterrows():
            aux = row['Classificació'].split('|')
            if len(aux) == 1:
                writer.writerow([row['Description'], row['Classificació'].strip()])
            else:
                multiple_label += 1

    print("Mostres amb més d'una classe: {}".format(multiple_label))
