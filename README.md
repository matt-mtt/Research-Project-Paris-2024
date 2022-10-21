# Lisez moi

Ce fichier contient une description brève des fichiers utilisés pour la prédiction de points clés de nageurs dans le dossier **Swimmers skeleton reconstitution**. Pour plus d'informations sur les méthodes utilisées, paramètres et processus, merci de lire mon rapport de stage. 
Jordan 

# Dossiers
## 12_labels

**Fichiers** : les labels de chaque vidéo utilisée pour l'entraînement des modèles. Les vidéos sont renommées arbitrairement en "video_1" etc..
### extracted_swimmers 
Contient les nageurs extraits provenant **uniquement** des images **labellisées** extraites des vidéos associées
### full_images 
Contient toutes les images de toutes les vidéos utilisées
### full_images_extracted_swimmers 
Contient les nageurs extraits des vidéos à leur framerate natif, à l'aide de l'interpolation de points clés.
### labeled_images 
Contient **uniquement** les images **labellisées** extraites des vidéos associées
### video_bboxes_swimmers 
Boîtes englobantes extraites à l'aide de la méthode de Nicolas Jacquelin. Plus utilisé.
### videos 
Explicite
### VoTT Exports 
Les traces issues des exportations du logiciel VoTT, c'est un beau mélange d'ailleurs, seuls le fichier csv des labels extraits m'était utile.

## CNN_LSTM
### dataset_*modèle*.py 
Regroupe toutes les méthodes de construction du jeu de données pour le modèle Pytorch, commenté.
### main_*modèle*.py 
Les fichiers d'exécution des modèles. La structure est toujours la même, les lignes commentées au début correspondre au type de données que vous voulez utiliser.
### mode<span>l.p</span>y et model_is_visible.py 
La déclaration des modèles à l'aide de la librairie Pytorch.
### predict_*modèle*.py 
Pipelines pour la prédiction sur des vidéos de test (et non de validation comme l'indique le nom du fichier, erreur de ma part). Il faut renseigner les chemins du modèle, de la vidéo et les hyperparamètres à utiliser. 				
NB:  Une taille de batch trop grande ne fonctionnera pas
### utils<span>.</span>py et utils_is_visible.py 
Pipelines d'affichages des fonctions de pertes ainsi que des images dans le jeu de validation avec leur prédictions.

### Saved models 
Explicite

## data
### 3labels_vott-json-export et Crawl_Nice_2021
Obsolète
### valid
Contient les vidéos utilisées pour le test des modèles (extraites sur YouTube pour la plupart)

## Keypoint interpolation
Codes pour interpoler les images labellisées à 15 FPS sur le framerate de chaque vidéo et ainsi augmenter les données.
### formatting<span>.</span>py
Pipeline de fonctions pour mettre en forme les données de manière similaire aux modèles
### frame_extraction<span>.</span>py
Extraire toutes les frames des vidéos présentes dans le dossier videos
### full_video_extract_swimmers_maxbox.py et full_video_extract_swimmers.py
Extraction des nageurs à l'aide des points clés interpolés
### interpolation<span>.</span>py
Pipeline pour l'interpolation des points clés selon la méthode décrite dans le rapport
### keypoint_visual_study.ipynb
Tests
### vizualisation<span>.</span>py
Visualisation des points clés sur une image donnée

***NB :*** Voici la démarche à suivre pour interpoler des points d'un image labellisée :

- Placer la vidéo dans le dossier videos au format mp4
- Extraire les images labellisées avec VoTT
- Convertir le csv des labels de VoTT à l'aide de **vott_to_centroid_labels.py**
- Extraire toutes les images des vidéos à l'aide de **frame_extraction.py**
- Interpoler les points à l'aide de **interpolation<span></span>.py**
- Extraire les nageurs à l'aide de **full_video_extract_swimmers.py**
- Exécuter les modèles en utilisant les images extraites

## Pre-work
Contient tous les codes du travail préliminaire détaillé dans le rapport, les fichiers sont explicites.

## Res
Divers résultats obtenus sur des sujets de test sous forme de vidéo

## results
Divers résultats obtenus (images et loss) sur des données de validation

## swimmers_detection-main
Copie du repo de Nicolas Jacquelin, utilisé pour l'extraction de blobs, à l'aide de **video_display.py**.

## Racine du dossier
### extract_swimmers_maxbox.py et extract_swimmers.py
Extraction des nageurs à l'aide des labels, **uniquement** sur les images labellisées.
### OLD_extract_swimmers.py
Ancien pipeline d'extraction de nageurs, à l'aide des blobs du code de Nicolas.
### vott_to_centroid_labels.py
Explicité.
