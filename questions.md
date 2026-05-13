# 🧠 Questions de Réflexion - Phase 1.3

Ce document regroupe les réponses aux questions théoriques posées durant la phase de conception du modèle CNN pour l'anti-gaspillage.

---

### 📏 1. Shape en sortie du bloc 2
**Question :** Calculez la shape en sortie du bloc 2 (après MaxPooling) pour votre `IMG_SIZE` choisie.

**Détails du calcul :**
- **Entrée :** `128x128x3`
- **Après Bloc 1 (MaxPooling 2x2) :** `64x64x32`
- **Après Bloc 2 (MaxPooling 2x2) :** `32x32x64`

> [!NOTE]
> **Shape finale en sortie du Bloc 2 :** `(None, 32, 32, 64)`

---

### 🔢 2. Paramètres de la couche Dense(128)
**Question :** Calculez le nombre total de paramètres de la couche `Dense(128)` à partir de la shape `Flatten`.

**Détails du calcul :**
1. **Sortie du Bloc 3 (après MaxPooling) :** `16x16x128`
2. **Flattening :** $16 \times 16 \times 128 = 32\,768$ éléments.
3. **Calcul des poids :**
   - $Poids = 32\,768 \text{ (entrées)} \times 128 \text{ (neurones)} = 4\,194\,304$
   - $Biais = 128$
   - **Total :** $4\,194\,304 + 128 = 4\,194\,432$ paramètres.

> [!WARNING]
> **Observation sur l'Overfitting :**
> Ce nombre de paramètres est extrêmement élevé pour une seule couche. C'est précisément cette complexité qui risque de causer un **overfitting** massif dans ce TP, car le modèle aura tendance à mémoriser les données d'entraînement plutôt qu'à généraliser.