# Round 2 Survey Analysis Artifact

## 1. Overview

This artifact provides the complete analysis code and anonymized survey data used in **Round 2** of our human-subjects evaluation. The goal of this study is to quantify participants’ ability to recognize adversarially manipulated text across five content domains. The provided Jupyter notebooks reproduce all quantitative analyses and figures reported in the paper, including demographic distributions, recognition rates, selection rates, and cross-topic comparisons.

All code has been anonymized and uses **relative paths only**. No personally identifiable information (PII) is included in the dataset.

---

## 2. Artifact Contents

The artifact contains one directory (`SURVEYDATAROUND2/`) with the following structure:

```
SURVEYDATAROUND2/
│
├── Hotel/
│   ├── hotel_data_finished.csv
│   ├── hotel_input_similarity_manually_check_finished.csv
│   └── round2_hotel_analysis.ipynb
│
├── Movie/
│   ├── movie_data_finished.csv
│   ├── movie_input_similarity_manually_check_finished.csv
│   └── round2_movie_analysis.ipynb
│
├── Music/
│   ├── music_data_finished.csv
│   ├── music_input_similarity_manually_check_finished.csv
│   └── round2_music_analysis.ipynb
│
├── Product/
│   ├── product_data_finished.csv
│   ├── product_input_similarity_manually_check_finished.csv
│   └── round2_product_analysis.ipynb
│
├── Restaurant/
│   ├── restaurant_data_finished.csv
│   ├── restaurant_input_similarity_manually_check_finished.csv
│   └── round2_restaurant_analysis.ipynb
│
└── round2_surveys_analysis_combine.ipynb   ← Main combined-analysis notebook

```

Each topic directory contains two preprocessed CSV files and one notebook that performs topic-specific analysis. The top-level combined notebook integrates all five topics and reproduces all final results.

---

## 3. Software Requirements

- **Python 3.12.7**
- **Jupyter Notebook**
- The following Python packages (or the included `requirements.txt`):

```
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn

```

No GPU or specialized hardware is required. All analyses run on a standard laptop CPU.

---

## 4. Data Description

All CSV files included in this artifact are **fully anonymized** and contain:

- cleaned participant responses
- manually validated similarity-based recognition fields
- derived recognition/selection indicators
- aggregated fields used in the analysis

All PII (IP addresses, geolocation, timestamps, response identifiers) was removed prior to packaging the artifact.

---

## 5. Reproducibility Instructions

### **Step 1: Install dependencies**

Create an environment (optional):

```bash
python3.12 -m venv env
source env/bin/activate
pip install -r requirements.txt

```

Or manually install the required libraries.

---

### **Step 2: Launch Jupyter**

```bash
jupyter notebook

```

---

### **Step 3: Run the combined analysis notebook**

Open:

```
round2_surveys_analysis_combine.ipynb

```

Then run:

**Kernel → Restart & Run All**

This notebook reproduces all major study results, including:

- demographic distributions
- recognition-rate and selection-rate aggregation
- per-rule and per-topic performance
- cross-topic comparisons
- final figures used in the paper

---

### **Optional Step 4: Run per-topic notebooks**

Each topic folder contains a notebook such as:

```
round2_hotel_analysis.ipynb
round2_movie_analysis.ipynb
round2_music_analysis.ipynb
round2_product_analysis.ipynb
round2_restaurant_analysis.ipynb
```

These reproduce topic-specific results reported or referenced in the paper. They are not required for the main findings but provide transparency and additional granularity.

---

## 6. Expected Outputs

Running the combined notebook will generate:

- recognition-rate bar charts with confidence intervals
- selection-rate summaries
- demographic tables and visualizations
- cross-domain aggregated results
- rule-level performance plots
- intermediate DataFrames used in the study

All visual outputs match those included in the submission.

---

## 7. Limitations

- The artifact includes only the processed survey data; raw Qualtrics exports containing PII cannot be released.
- Visualization styling may vary slightly depending on local Matplotlib defaults.
- Analysis notebooks assume the directory structure remains unchanged.

---

## 8. Ethical and Privacy Considerations

This artifact contains only anonymized survey responses.

All PII has been removed and no re-identifiable information remains.

The analysis procedures comply with the IRB protocol referenced in the paper, including secure storage, anonymization, and aggregated reporting.

---

## 9. Contact (Anonymized)

Per IEEE S&P double-blind policy, author contact information is omitted.
