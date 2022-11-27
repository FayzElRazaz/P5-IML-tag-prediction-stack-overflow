# StackOverflow_tags
Nous avons ici travaillé sur un sujet NLP.
L'idée a été de récupéré des questions posé sur le célèbre site StackOverFlow (en récupérant la donnée via une API que StackOverflow met à disposition).

Nous avons ensuite étudié les tags associé à la question entrée par l'utilisateur, en utilisant les techniques d'apprentissage non supervisé (topic modelling, LDA), semi supervisé (LDA en utilisant les tags), et supervisé (random forest, SVM, boosting) et des techniques de NLP plus avancé pour effectuer ces mêmes taches (t-SNE sur Word2Vec, classification mutlilabel en utilisant BERT).
Nous avons crée une API permettant de suggérer les tags à associés à une question.  
# Python packages
pip3 install -r requirements.txt
