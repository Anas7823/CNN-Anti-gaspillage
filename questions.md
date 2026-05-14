# 🧠 Analyses et Réflexions - Projet Anti-Gaspillage (CNN & Transfer Learning)

Ce document regroupe les réponses aux questions théoriques et les diagnostics comparatifs réalisés tout au long du projet de classification multi-classes (28 catégories de fruits).

---

## 🏗️ Phase 1.3 : Architecture du CNN "From Scratch"

### 1. Calcul de la Shape en sortie du Bloc 2
**Question :** Calculez la dimension (shape) en sortie du bloc 2 (après MaxPooling) pour l'image d'entrée $128 \times 128$.

**Détails du calcul :**
- **Entrée réseau :** $128 \times 128 \times 3$ (Image RGB)
- **Après Bloc 1 (Conv2D 32 + MaxPooling 2x2) :** La division spatiale par 2 donne $64 \times 64 \times 32$.
- **Après Bloc 2 (Conv2D 64 + MaxPooling 2x2) :** La division spatiale par 2 donne $32 \times 32 \times 64$.

> [!NOTE]
> **Shape finale en sortie du Bloc 2 :** `(None, 32, 32, 64)`  
> *Le `None` représente la dimension variable du batch.*

---

### 2. Calcul des paramètres de la couche Dense(128)
**Question :** Calculez le nombre exact de paramètres entraînables de l'avant-dernière couche `Dense(128)` après l'aplatissement (`Flatten`).

**Détails du calcul :**
1. **Sortie du Bloc 3 (Conv2D 128 + MaxPooling 2x2) :** La division spatiale par 2 donne $16 \times 16 \times 128$.
2. **Flattening :** L'aplatissement de ce tenseur 3D en un vecteur 1D donne : $16 \times 16 \times 128 = 32\,768$ caractéristiques (features).
3. **Calcul de la couche Dense (128 neurones) :**
   - **Poids :** $32\,768 \text{ (entrées)} \times 128 \text{ (neurones)} = 4\,194\,304$
   - **Biais :** $128$ (un par neurone)
   - **Total :** $4\,194\,304 + 128 = 4\,194\,432$ paramètres.

> [!WARNING]
> **Observation sur l'Overfitting :**
> Sur les ~4,2 millions de paramètres du modèle, presque **99%** se trouvent dans cette unique connexion. Cette complexité massive crée un goulot d'étranglement mémoriel, causant le surapprentissage (**overfitting**) massif observé en Phase 1.

---

## 📊 Phase 2.3 : Diagnostic Comparatif (Scratch vs Augmenté)

**Question :** Qu'est-ce qui a changé entre les courbes du TP1 et du TP2 ? Le gap train/val s'est-il réduit ? La convergence est-elle plus lente ou plus rapide ? Pourquoi l'augmentation seule ne suffit pas ?

**Interprétation :**
- **Gap Train/Val :** Entre le TP1 (Baseline) et le TP2 (Régularisation), l'écart entre la précision d'entraînement et de validation s'est nettement réduit.
- **Convergence :** La convergence est mécaniquement **plus lente**. L'injection de transformations aléatoires (*Data Augmentation* : zoom, rotation, flip) et la désactivation de neurones (*Dropout* à 40%) obligent le réseau à extraire des motifs visuels robustes plutôt qu'à mémoriser les pixels du dataset d'entraînement.
- **Limites :** Ces techniques restent insuffisantes pour atteindre un niveau de production (>95%). L'augmentation de données ne pallie pas la limite fondamentale de notre architecture : avec seulement 3 blocs convolutifs, notre réseau manque de profondeur et de filtres complexes pour comprendre les nuances subtiles entre 28 classes de fruits.

---

## 🚀 Phase 3.4 : Bilan Cross-Modèles (Transfer Learning)

**Question :** Qu'est-ce que le bilan révèle sur le rapport performance/coût de chaque approche ? Pourquoi MobileNetV2 gagne sur tous les fronts malgré moins de paramètres actifs ?

**Interprétation :**
Le tableau de comparaison final met en évidence l'écrasante supériorité du **Transfer Learning**. Le modèle MobileNetV2 fine-tuné atteint une précision quasi-parfaite (**98%**), tout en étant :
1. **Plus léger :** ~9 Mo en FP32 (~3 Mo en INT8) contre plus de 50 Mo pour le scratch.
2. **Plus rapide :** Temps de calcul par epoch réduit grâce à une architecture optimisée.

**Pourquoi MobileNetV2 gagne ?**
- **L'efficience :** Il utilise des *Depthwise Separable Convolutions* qui réduisent drastiquement la quantité de paramètres nécessaires par rapport à des couches Conv2D standards.
- **La réutilisation des connaissances :** La base du modèle (2,2 millions de paramètres) est "gelée". Elle ne subit pas la rétropropagation des gradients. Le modèle n'apprend pas "à voir" depuis zéro, il utilise les concepts visuels déjà appris sur les 14 millions d'images d'ImageNet pour les appliquer instantanément à notre problématique anti-gaspillage.