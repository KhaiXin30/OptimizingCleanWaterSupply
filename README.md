# Water Quality Forecasting 

Predicts three chemical water quality properties at locations never seen during training, using satellite imagery and climate data as inputs.

---

## What It Predicts

| Target | Unit |
|---|---|
| Total Alkalinity | mg/L |
| Electrical Conductance | µS/cm |
| Dissolved Reactive Phosphorus | µg/L |

---

## The Core Challenge

The 24 validation sites have **zero geographic overlap** with the 162 training sites. The model cannot rely on recognising a location — it must generalise purely from the physical signals at that place.

---

## Data Sources

| Source | Features |
|---|---|
| Landsat (satellite) | Spectral bands: NIR, Red, Green, Blue, SWIR |
| TerraClimate | Temperature, rainfall, evapotranspiration, soil moisture |
| ESA | Land cover: % cropland, % urban, % natural |
| JRC | Water occurrence, seasonality, recurrence |

All sources are merged onto the base dataset using `(latitude, longitude, date)` as the join key.

---

## Feature Engineering

**Spectral indices** computed from Landsat bands:
- NDVI — vegetation cover
- NDWI — open water presence
- SABI, turbidity, chlorophyll ratio — water quality proxies

**Temporal features:**
- Month, year, day-of-year
- Cyclical sine/cosine encoding of seasonality

**Climate interactions:**
- Water stress, runoff ratio, human impact index, crop × discharge

**Total: 39 features** going into the model.

---

## Handling Unseen Locations — Spatial Features

For every site in the dataset (training or validation), the model borrows target statistics from the **5 nearest known training sites**, weighted by how close they are.

```
closer site  →  higher weight
farther site →  lower weight
```

This produces 3 spatial proxy features per row:
- `sp_total_alkalinity`
- `sp_electrical_conductance`
- `sp_dissolved_reactive_phosphorus`

For validation sites, the full training set is used as the reference pool. For CV folds, only that fold's training rows are used — preventing data leakage.

---

## Model

A separate **HistGradientBoostingRegressor** is trained for each target.

| Target | Key Hyperparameters |
|---|---|
| Total Alkalinity | 400 trees, lr=0.05, depth=6 |
| Electrical Conductance | 400 trees, lr=0.05, depth=6 |
| Dissolved Reactive Phosphorus | 500 trees, lr=0.04, depth=5, higher regularisation |

---

## Training & Validation Strategy

5-fold cross-validation is used to evaluate performance before training a final model on all 9,319 rows.

**Leakage prevention:** spatial features are recomputed fresh inside each fold using only that fold's training data as the neighbor reference. The held-out rows never see information they wouldn't have in the real world.

---

## Results (Out-of-Fold R²)

| Target | R² |
|---|---|
| Total Alkalinity | **0.851** |
| Electrical Conductance | **0.869** |
| Dissolved Reactive Phosphorus | **0.694** |



