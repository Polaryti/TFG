- [x] Pre-neteja més exhaustiva:
    - [x] ", "
    - [x] ??
    - [x] 271 - Vacia
    - [x] 294 - nan
    - [x] 499, 577 - falsos directes
    - [x] Eliminats signes de puntuació i demés simbols
    - [x] Amb el corpus complet apareixen més problemes de neteja

- [ ] Substitució de apostrofs per paraula completa
- [x] Stopwords (http://latel.upf.edu/morgana/altres/pub/ca_stop.htm)
- [x] Stemmer (http://snowball.tartarus.org/algorithms/catalan/stemmer.html)
    - No disponible a Python
- [ ] Lemmatization (https://github.com/michmech/lemmatization-lists)
    - *En proces de construcció d'un propi*

- [x] Models i la seua entrada Hugging Face:
    - Hi ha un total de 32 models dels quals 26 son per a traducció, 2 de *automatic speech recognition* i la resta de *fill-mask*. Cap de *text classification*.

- [ ] https://github.com/ccoreilly/spacy-catala
    - Es una versió de fa tan sols un any pero pareix que ja no es compatible amb la versió actual de Spacy.
    - *(fent més proves)*

- [ ] *Exploration data analysis* de les clases simples:
    - [x] Representació visual de algunes caracteristiques
    - [x] Class count

- [x] Métrica: *recall* *(dóna errors al utilitzar-la a Sklearn amb SVM i NB)*.

- [x] Sklearn:
    - [x] Naive Bayes
    - [x] SVM
    - Sense stopwords millora un 1%.
    - Agafant a soles una clase la millora es d'un 3-4%.

- [ ] Confusion matrix
- [ ] Most frequent words en cada clase
