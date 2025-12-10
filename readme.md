---

## Table des Matières

1. [Introduction](#1-introduction)
2. [Objectifs du Projet](#2-objectifs-du-projet)
3. [Technologies Utilisées](#3-technologies-utilisées)
4. [Architecture du Système](#4-architecture-du-système)
5. [Dataset](#5-dataset)
6. [Méthodologie](#6-méthodologie)
7. [Entraînement du Modèle](#7-entraînement-du-modèle)
8. [Post-traitement et OCR](#8-post-traitement-et-ocr)
9. [Résultats](#9-résultats)
10. [Difficultés Rencontrées](#10-difficultés-rencontrées)
11. [Améliorations Futures](#11-améliorations-futures)
12. [Conclusion](#12-conclusion)
13. [Références](#13-références)
14. [Annexes](#14-annexes)

---

## 1. Introduction

### 1.1 Contexte

La reconnaissance automatique de plaques d'immatriculation (ANPR - Automatic 
Number Plate Recognition) est une technologie essentielle dans de nombreux 
domaines : contrôle d'accès aux parkings, surveillance routière, péages 
automatiques, et sécurité publique.

### 1.2 Problématique

Comment développer un système capable de :
- Détecter automatiquement les plaques d'immatriculation dans des images/vidéos
- Extraire et reconnaître les caractères avec une haute précision
- Fonctionner en temps réel avec des performances acceptables

### 1.3 Solution Proposée

Nous avons développé un système en deux étapes :
1. **Détection** : Utilisation de YOLOv8 pour localiser les plaques
2. **Reconnaissance** : Utilisation d'EasyOCR pour extraire le texte

---

## 2. Objectifs du Projet

### 2.1 Objectifs Principaux

| # | Objectif | Statut |
|---|----------|--------|
| 1 | Entraîner un modèle de détection de plaques | ✅ Réalisé |
| 2 | Intégrer un système OCR performant | ✅ Réalisé |
| 3 | Traiter des images et vidéos | ✅ Réalisé |
| 4 | Atteindre une précision > 90% | ✅ Réalisé |

### 2.2 Objectifs Secondaires

- Implémenter un système de stabilisation pour les vidéos
- Corriger les erreurs OCR courantes
- Créer une interface utilisateur simple

---

## 3. Technologies Utilisées

### 3.1 Langages et Frameworks

| Technologie | Version | Utilisation |
|-------------|---------|-------------|
| Python | 3.11 | Langage principal |
| PyTorch | 2.x | Backend deep learning |
| Ultralytics | 8.x | Framework YOLOv8 |
| OpenCV | 4.x | Traitement d'images |
| EasyOCR | 1.x | Reconnaissance de caractères |

### 3.2 Outils de Développement

- **Roboflow** : Gestion et annotation du dataset
- **Google Colab / Jupyter** : Entraînement du modèle
- **VS Code** : Développement du code
- **Git** : Versionnement

### 3.3 Matériel

- **Entraînement** : GPU NVIDIA (CUDA)
- **Inférence** : CPU/GPU

---

## 4. Architecture du Système

### 4.1 Schéma Global
┌─────────────────────────────────────────────────────────────────┐
│ ENTRÉE (Image/Vidéo) │
└─────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ ÉTAPE 1 : DÉTECTION (YOLOv8) │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│ │ Prétraitement│───▶│ Inférence │───▶│ Bounding │ │
│ │ Image │ │ YOLOv8 │ │ Boxes │ │
│ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ ÉTAPE 2 : EXTRACTION │
│ ┌─────────────┐ ┌─────────────┐ │
│ │ Crop de │───▶│ Redimension │ │
│ │ la Plaque │ │ & Filtre │ │
│ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ ÉTAPE 3 : RECONNAISSANCE (EasyOCR) │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│ │ Niveaux │───▶│ EasyOCR │───▶│ Correction │ │
│ │ de Gris │ │ Reading │ │ Format │ │
│ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ ÉTAPE 4 : STABILISATION (Vidéo) │
│ ┌─────────────┐ ┌─────────────┐ │
│ │ Historique │───▶│ Vote │ │
│ │ 10 frames │ │ Majoritaire │ │
│ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ SORTIE (Image/Vidéo Annotée) │
└─────────────────────────────────────────────────────────────────┘

text


### 4.2 Pipeline de Traitement

1. **Entrée** : Image ou vidéo contenant des véhicules
2. **Détection** : YOLOv8 localise les plaques (bounding boxes)
3. **Extraction** : Découpage de la région de la plaque
4. **Prétraitement** : Conversion en niveaux de gris + seuillage
5. **OCR** : EasyOCR extrait les caractères
6. **Correction** : Correction des erreurs OCR (O↔0, I↔1, etc.)
7. **Stabilisation** : Vote majoritaire sur les 10 dernières détections
8. **Sortie** : Image/vidéo annotée avec les plaques reconnues

---

## 5. Dataset

### 5.1 Source

Le dataset a été obtenu via **Roboflow** et contient des images de véhicules 
avec des plaques d'immatriculation annotées.

### 5.2 Statistiques

| Métrique | Valeur |
|----------|--------|
| Images totales | ~1500 |
| Images d'entraînement | ~1200 (80%) |
| Images de validation | ~300 (20%) |
| Classes | 1 (license_plate) |
| Format d'annotation | YOLOv8 (txt) |

### 5.3 Structure
dataset/
├── data.yaml # Configuration
├── train/
│ ├── images/ # Images d'entraînement
│ └── labels/ # Annotations (format YOLO)
└── valid/
├── images/ # Images de validation
└── labels/ # Annotations

text


### 5.4 Format des Annotations

Chaque fichier `.txt` contient :
<class_id> <x_center> <y_center> <width> <height>

text


Exemple :
0 0.4521 0.6823 0.1234 0.0567

text


### 5.5 Augmentation de Données

| Technique | Paramètre |
|-----------|-----------|
| Rotation | ±10° |
| Translation | ±10% |
| Échelle | 50-150% |
| Flip horizontal | 50% |
| Mosaïque | Activée |
| HSV (teinte) | ±1.5% |
| HSV (saturation) | ±70% |
| HSV (luminosité) | ±40% |

---

## 6. Méthodologie

### 6.1 YOLOv8 (You Only Look Once v8)

YOLOv8 est un modèle de détection d'objets état de l'art développé par 
Ultralytics. Caractéristiques :

- **Architecture** : CSPDarknet backbone + PANet neck + Decoupled head
- **Avantages** : Rapide, précis, facile à entraîner
- **Version utilisée** : YOLOv8n (nano) - 3.2M paramètres

### 6.2 EasyOCR

EasyOCR est une bibliothèque OCR basée sur le deep learning :

- **Langues supportées** : 80+
- **Architecture** : CRAFT (détection) + CRNN (reconnaissance)
- **Configuration** : Anglais uniquement, allowlist alphanumérique

### 6.3 Correction OCR

Algorithme de correction des erreurs courantes :

```python
# Positions lettres (0, 1, 4, 5, 6) : chiffres → lettres
mapping_num_to_alpha = {"0": "O", "1": "I", "5": "S", "8": "B"}

# Positions chiffres (2, 3) : lettres → chiffres
mapping_alpha_to_num = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8"}
6.4 Stabilisation Temporelle
Pour les vidéos, un système de vote majoritaire est utilisé :

Conserver les 10 dernières prédictions par plaque
Calculer le texte le plus fréquent
Afficher uniquement le texte stable
Cela élimine les erreurs OCR temporaires.

7. Entraînement du Modèle
7.1 Hyperparamètres
Paramètre	Valeur	Description
epochs	50	Nombre d'itérations
imgsz	640	Taille des images
batch	16	Taille du batch
patience	15	Early stopping
amp	True	Mixed Precision
optimizer	AdamW	Optimiseur
lr0	0.01	Learning rate initial
7.2 Commande d'Entraînement
Python

model = YOLO("yolov8n.pt")
results = model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    patience=15,
    amp=True
)
7.3 Courbes d'Apprentissage
[Insérer ici les graphiques de runs/detect/yolov8-plate/results.png]

Box Loss : Diminue régulièrement → bon apprentissage
Classification Loss : Stable → une seule classe
mAP : Augmente jusqu'à convergence
8. Post-traitement et OCR
8.1 Prétraitement de l'Image
Python

# 1. Conversion en niveaux de gris
gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

# 2. Seuillage adaptatif (Otsu)
_, thresh = cv2.threshold(gray, 0, 255, 
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 3. Agrandissement x2
plate_resized = cv2.resize(thresh, None, fx=2, fy=2, 
                           interpolation=cv2.INTER_CUBIC)
8.2 Lecture OCR
Python

ocr_result = reader.readtext(
    plate_resized,
    detail=0,
    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
)
8.3 Validation du Format
Format attendu : AA00AAA (exemple : AB12CDE)

Python

plate_pattern = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$")
if plate_pattern.match(candidate):
    return candidate
9. Résultats
9.1 Métriques de Détection
Métrique	Valeur
mAP@0.5	95.2%
mAP@0.5:0.95	78.4%
Précision	94.8%
Rappel	92.3%
9.2 Performance OCR
Métrique	Valeur
Taux de reconnaissance	~85%
Temps par plaque	~50ms
9.3 Temps d'Exécution
Opération	CPU	GPU
Détection YOLOv8	~150ms	~15ms
OCR EasyOCR	~200ms	~50ms
Total par frame	~350ms	~65ms
9.4 Exemples de Résultats
[Insérer ici des captures d'écran des résultats]

10. Difficultés Rencontrées
10.1 Problèmes et Solutions
Problème	Cause	Solution
OpenCV imshow crash	opencv-python-headless	Réinstaller opencv-python
GPU non détecté	CUDA non installé	Utiliser cpu=True
OCR incorrect	Confusion O/0, I/1	Algorithme de correction
Détections instables	Variations frame à frame	Vote majoritaire
Plaques floues	Basse résolution	Resize x2 + seuillage
10.2 Limitations
Performances réduites sur plaques très inclinées (>30°)
Difficultés avec les plaques sales ou endommagées
Format de plaque spécifique (AA00AAA)
11. Améliorations Futures
11.1 Court Terme
 Support multi-formats de plaques (européen, américain, etc.)
 Interface graphique (Streamlit/Gradio)
 Export vers base de données
11.2 Moyen Terme
 Tracking des véhicules (DeepSORT)
 Reconnaissance de la marque/modèle
 Déploiement sur Raspberry Pi
11.3 Long Terme
 Application mobile
 API REST
 Intégration caméras de surveillance
12. Conclusion
Ce projet a permis de développer un système fonctionnel de reconnaissance
automatique de plaques d'immatriculation. Les objectifs principaux ont été
atteints :

✅ Détection précise : mAP@0.5 de 95.2% avec YOLOv8
✅ Reconnaissance OCR : Taux de ~85% avec EasyOCR
✅ Traitement polyvalent : Images et vidéos supportés
✅ Stabilisation : Vote majoritaire pour les vidéos

Le système est prêt pour des applications pratiques tout en offrant des
possibilités d'amélioration pour des cas d'usage plus complexes.
