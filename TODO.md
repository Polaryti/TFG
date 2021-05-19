- [x] Pre-neteja més exhaustiva:
    - [x] ", "
    - [x] ??
    - [x] 271 - Vacia
    - [x] 294 - nan
    - [x] 499, 577 - falsos directes
    - [x] Eliminats signes de puntuació i demés simbols
    - [x] Amb el corpus complet apareixen més problemes de neteja

- [X] Substitució de apostrofs per paraula completa
    - Es lleva com si fora una stopword
- [x] Stopwords (http://latel.upf.edu/morgana/altres/pub/ca_stop.htm)
- [x] Stemmer (http://snowball.tartarus.org/algorithms/catalan/stemmer.html)
    - No disponible a Python
- [X] Lemmatization (https://github.com/michmech/lemmatization-lists)
    - *En proces de construcció d'un propi*
    - Es decarta, per ara

- [x] Models i la seua entrada Hugging Face:
    - Hi ha un total de 32 models dels quals 26 son per a traducció, 2 de *automatic speech recognition* i la resta de *fill-mask*. Cap de *text classification*.

- [X] https://github.com/ccoreilly/spacy-catala
    - Es una versió de fa tan sols un any pero pareix que ja no es compatible amb la versió actual de Spacy.
    - Es descarta, per ara

- [x] *Exploration data analysis* de les clases simples:
    - [x] Representació visual de algunes caracteristiques
    - [x] Class count

- [x] Métrica: *recall* *(dóna errors al utilitzar-la a Sklearn amb SVM i NB)*.

- [x] Sklearn:
    - [x] Naive Bayes
    - [x] SVM
    - Sense stopwords millora un 1%.
    - Agafant a soles una clase la millora es d'un 3-4%.

- [x] Confusion matrix
- [X] Estadistiques de cada classe

- [X] Hacer experimiento con las 4 clases principales
    - [X] Repetir amb FastText
- [X] Llevar les mostres amb més d'una classe
- [X] Representación vectorial de l'entrada, embeddings (FastText)
- [X] Revisar la SVM

**Reunió 20/05/21**
- [x] Entrenar amb bi-grames *("hola que tal" -> " hola", "hola que", "que tal", "tal ")*
- [X] Llevar els trigrames repetits
- [x] Probar i modificar la SVM
    - El problema era que agafava la representació CountVector en compte de la Tf-IDF
- [X] Comprobar si la classe correcta está en les 2-5 primeres classes
- [X] FastText, veure amb model descarregat si es pot actualitzar i generar els embeddings amb aquest i el corpus d'APunt
- [x] Repetir els experiments amb els mateixos corpus de test/train
    - Resultats en: https://docs.google.com/spreadsheets/d/19pMylt7uQ3ZM6gJ7z0eLV5R55Ust_B0nmb2yHvm_PLQ/edit?usp=sharing
- [X] Utilitzar els embeddings de FastText per al NB i la SVM

**Reunió 03/06/21**
- [ ] Usar get_sentence_vector() (FastText) 
- [ ] decision_function() (SVM)
- [ ] Analisis de recall i precisió per classe
- [ ] Omplir taula de 4-6 classes per a FastText
- [ ] Comprobar si la classe correcta está en les 3-5 primeres classes (precissió)
- [ ] Analisis del corpus de train entre solapament de bi-grames, tri-grames... entre classes
- [ ] Augmentar els n-grames de n-diferents
- [ ] Anar redactant

**Per al futur...**
- [ ] Demo de aplicació per a la defensa i periodistes
- [ ] Gran experiment de Lluis
- [ ] Possible experiment amb mayusculas y minusculas