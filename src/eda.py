import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv(r'res/corpus_ambStopwords.csv', encoding="utf-8")

    # CLASS COUNT
    simple_class_count = {}
    for index, row in df.iterrows():
        raw = row['Classificació'].strip()
        if raw not in simple_class_count:
            simple_class_count[raw] = 0
        simple_class_count[raw] += 1

    with open(r'res/classCount_ambStopwords.csv', 'w', encoding="utf-8", newline='') as w_file:
        simple_class_count = {k: v for k, v in sorted(
            simple_class_count.items(), key=lambda item: item[1], reverse=True)}
        for key, value in simple_class_count.items():
            w_file.write(f'{key}% {value}\n')
