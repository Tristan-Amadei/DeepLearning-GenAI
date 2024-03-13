# Projet de Modèles Génératifs et Classification

Ce projet explore plusieurs aspects des modèles génératifs et de la classification, en utilisant des techniques telles que les Restricted Boltzmann Machines (RBM), les Deep Belief Networks (DBN), les Deep Neural Networks (DNN), ainsi que des modèles génératifs tels que les Variational Autoencoders (VAE) et les Generative Adversarial Networks (GAN). Le but est d'analyser la capacité de ces modèles à générer des images et à classifier des données, ainsi que de comparer différentes techniques d'optimisation.

## Contenu du Projet

### 1. Entraînement de RBM et DBN sur Alpha Digits

Dans cette première partie, nous entraînons des Restricted Boltzmann Machines (RBM) et des Deep Belief Networks (DBN) sur le jeu de données Alpha Digits pour évaluer leur capacité à générer des images.

### 2. Pré-entraînement d'un DNN avec DBN pour la Classification sur MNIST

Dans cette étape, un Deep Belief Network (DBN) est utilisé pour pré-entraîner un Deep Neural Network (DNN) pour la classification sur le jeu de données MNIST. Nous comparons les performances de différentes techniques d'optimisation (descente de gradient standard vs. Adam) avec et sans pré-entraînement.

### 3. Entraînement de Modèles Génératifs sur MNIST

Dans cette dernière partie, différents modèles génératifs (RBM, DBN, β-VAE, GAN) sont entraînés sur le jeu de données MNIST binarisé. Nous utilisons le score de Fréchet Inception Distance (FID) pour évaluer la qualité de génération des images produites par chaque modèle.

## Organisation du Code

- **`rbm_dbn_alpha_digits.ipynb`**: Notebook Jupyter contenant le code pour l'entraînement des RBM et DBN sur Alpha Digits.
- **`dbn_pretraining_classification.ipynb`**: Notebook Jupyter contenant le code pour pré-entraîner un DNN avec un DBN pour la classification sur MNIST.
- **`generative_models_mnist.ipynb`**: Notebook Jupyter contenant le code pour l'entraînement des modèles génératifs sur MNIST et l'évaluation avec le score FID.

## Dépendances

Ce projet utilise les bibliothèques suivantes :
- NumPy
- Scipy
- Pytorch
- Torchsummary

Assurez-vous que toutes les dépendances sont installées en utilisant `pip install -r requirements.txt`.

## Instructions d'Exécution

Les différents modèles et optimiseurs sont codés sous forme de classes. Leurs codes sont à retrouver dans **`src/*.py`**


<br> Le dossier **`src/notebooks`** contient les notebooks des études menées dans la 1ère et 3ème partie du projet, à savoir les parties génératives:
 - **`rbm.ipynb`** contient des expérimentations avec le RBM sur AlphaDigits.
 - **`rbm_study.ipynb`** contient le code utilisé pour créer les images présentes dans le rapport, dans la première partie sur RBM.
 - **`dbn.ipynb`** contient des expérimentations avec le DBN sur AlphaDigits.
 - **`rbm_study.ipynb`** contient le code utilisé pour créer les images présentes dans le rapport, dans la première partie sur DBN.
 - **`dnn.ipynb`** contient des expérimentations avec le DNN sur AlphaDigits et MNIST.


Les études sur la classification supervisée de MNIST sont à retrouver dans **`src/runs adam`**, où l'optimiseur Adam a été utilisé, et **`src/runs gd`** où une descente de gradient standard a été utilisée. Dans chacun de ces dossiers, pour chaque run de modèle, nous avons sauvegardé les poids des DNN, ainsi que les loss/accuracy au cours de l'entrainement. L'élément important est le notebook dans ces dossiers, **`dnn_adam.ipynb`** ou **`dnn_gd.ipynb`**, où se trouve le code utilisé pour entrainer les modèles et créer les différentes figures qui sont présentes dans la partie 2 du rapport.


**`src/notebooks/VAE`** et **`src/notebooks/GAN`** contiennent le code utilisé pour créer les images présentes dans la dernière partie du rapport, à savoir l'entrainement de modèles génératifs sur MNIST binarisée, la génération de nouvelles images à partir de ces modèles entrainés et la quantification de ces images par le score FID.

## Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE.md](LICENSE.md) pour plus de détails.
