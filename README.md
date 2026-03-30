# 🏧 FA-2: ATM Intelligence Demand Forecasting with Data Mining
### FinTrust Bank Ltd. — Actionable Insights & Interactive Python Script

---
APP LINK: https://7fpahnc8fiur4fgvshwvys.streamlit.app/
## 📌 About This Project

This project is **Formative Assessment 2 (FA-2)** for the Data Mining module. It builds on the cleaned ATM dataset from FA-1 and applies core data mining techniques to generate actionable insights for FinTrust Bank Ltd.

The goal is to help bank managers make smarter, data-driven decisions about ATM cash management — by understanding patterns, grouping ATMs by behavior, detecting unusual activity, and exploring the data interactively.

**Dataset used:** `atm_cash_management_dataset.csv`  
**Script:** `FA2_ATM_Analysis.py`  
**Language:** Python 3  

---

## 📁 Files in This Project

| File | Description |
|------|-------------|
| `FA2_ATM_Analysis.py` | Main Python script — runs all 4 stages |
| `atm_cash_management_dataset.csv` | ATM transaction dataset (5,658 records, 13 columns) |
| `README.md` | This file |

---

## ▶️ How to Run

1. Place both `.py` and `.csv` files in the **same folder**
2. Install required libraries (one time only):
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn scipy
   ```
3. Run the script:
   ```
   python FA2_ATM_Analysis.py
   ```
4. Charts will appear one by one. For **Stage 6**, type your filters in the terminal when prompted.

> Also works in **Google Colab** or **Jupyter Notebook** using:
> ```python
> exec(open("FA2_ATM_Analysis.py").read())
> ```

---

## 🔬 What the Script Does — Stage by Stage

---

### Stage 3 — Exploratory Data Analysis (EDA)
> *Lines ~35 to ~130 in the script*

This stage explores the dataset visually to find patterns and trends.

**Charts generated:**
- **Histogram** of Total Withdrawals & Total Deposits
- **Box plots** to spot outliers
- **Line chart** of withdrawals over time
- **Bar chart** by Day of Week and Time of Day
- **Bar chart** showing impact of Holiday & Special Event flags
- **Box plots** comparing Weather Condition and Nearby Competitor ATMs
- **Scatter plot**: Previous Day Cash Level vs Cash Demand Next Day
- **Correlation heatmap** of all numeric features

Each chart is followed by a printed observation explaining the key finding.

---

### Stage 4 — Clustering Analysis ⭐
> *Lines ~135 to ~195 in the script*

This is where **ATM clustering** happens.

**What the code does, step by step:**

1. **Encodes** the `Location_Type` column (text → number) using `LabelEncoder` so it can be used in math-based clustering
2. **Selects features** for clustering:
   - `Total_Withdrawals`
   - `Total_Deposits`
   - `Location_Type_Enc` (encoded)
   - `Nearby_Competitor_ATMs`
3. **Standardizes** the features using `StandardScaler` — so no single feature dominates due to scale differences
4. **Runs the Elbow Method** — tests K=2 through K=10 and plots inertia to find where adding more clusters stops helping much
5. **Runs Silhouette Scoring** — picks the K value with the best cluster separation score
6. **Applies K-Means** with the best K value found
7. **Labels clusters** from "Very Low Demand ATMs" → "Extreme Demand ATMs" based on average withdrawals
8. **Plots clusters** as a scatter chart (Withdrawals vs Deposits, color-coded by cluster)
9. **Bar chart** showing how many records fall in each cluster

**Key library used:** `sklearn.cluster.KMeans`, `sklearn.metrics.silhouette_score`

---

### Stage 5 — Anomaly Detection ⭐
> *Lines ~200 to ~255 in the script*

This is where **anomaly detection** happens.

**What the code does, step by step:**

1. **Holiday vs Normal comparison** — histogram overlaying withdrawals on holiday days vs normal days to visually confirm elevated demand
2. **Z-Score method** — calculates how many standard deviations each withdrawal is from the mean. Records with Z-Score > 2.5 are flagged as anomalies
   ```python
   df["Z_Score"] = np.abs(stats.zscore(df["Total_Withdrawals"]))
   df["Anomaly_ZScore"] = df["Z_Score"] > 2.5
   ```
3. **IQR method** — flags records that fall below `Q1 - 1.5×IQR` or above `Q3 + 1.5×IQR` as outliers
   ```python
   df["Anomaly_IQR"] = (df["Total_Withdrawals"] < lower_bound) | (df["Total_Withdrawals"] > upper_bound)
   ```
4. **Isolation Forest** — a machine learning method that isolates anomalies by randomly splitting data. Rare, unusual points are isolated faster. Contamination is set to 5% (meaning ~5% of records expected to be anomalies)
   ```python
   iso = IsolationForest(contamination=0.05, random_state=42)
   df["Anomaly_IF"] = iso.fit_predict(df[["Total_Withdrawals"]]) == -1
   ```
5. **Time-series scatter plot** — shows all records over time, with anomalies highlighted in orange
6. **Bar chart** — compares how many anomalies occurred on holiday days vs normal days

**Key libraries used:** `scipy.stats.zscore`, `sklearn.ensemble.IsolationForest`

---

### Stage 6 — Interactive Planner
> *Lines ~260 to ~330 in the script*

An interactive terminal tool where users can filter ATM data by:
- **Day of Week** (e.g. Monday, Friday)
- **Time of Day** (e.g. Morning, Evening)
- **Location Type** (e.g. Mall, Supermarket, Standalone)

After filtering, it shows:
- Summary statistics (mean, min, max withdrawals)
- Cluster breakdown for filtered ATMs
- Anomaly count and percentage
- A 3-panel dashboard chart (distribution, clusters, anomaly scatter)

Type `exit` at any prompt to quit. Type `yes` to run another filter.

---

## 📊 Results Summary

| Stage | Output |
|-------|--------|
| EDA | 10 visualizations with observations |
| Clustering | ATMs grouped into demand-based clusters using K-Means |
| Anomaly Detection | Anomalies flagged using Z-Score, IQR, and Isolation Forest |
| Interactive Planner | Filter-based dashboard for managers |

---

## 📚 Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Chart plotting |
| `seaborn` | Statistical visualizations |
| `scipy.stats` | Z-Score calculation |
| `sklearn.preprocessing` | StandardScaler, LabelEncoder |
| `sklearn.cluster` | KMeans clustering |
| `sklearn.metrics` | Silhouette score |
| `sklearn.ensemble` | Isolation Forest anomaly detection |

---

*FA-2 | Data Mining | Year 1 | FinTrust Bank ATM Intelligence Project*
