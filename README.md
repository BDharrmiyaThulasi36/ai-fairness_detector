⚖️ AI Fairness & Bias Auditor

AI systems are only as fair as the data and models behind them.
This project is a Responsible AI tool that helps detect and measure bias in machine learning models.

With this Streamlit app, you can:

📂 Upload any dataset (CSV)
🎯 Select a target variable (true labels)
🧑‍🤝‍🧑 Select a sensitive attribute (e.g., gender, race, age)
🚀 Run fairness audits across groups
📊 View performance + fairness metrics with charts
💾 Export audit results as CSV reports

Features:

* Supports binary, multi-class, and regression datasets
* Automatically handles categorical labels (Yes/No, Pass/Fail, etc.)
* Provides group-level fairness analysis
* Calculates advanced fairness metrics:

  * Demographic Parity Difference
  * Equal Opportunity Difference
  * Statistical Parity Ratio
* Generates clear visualizations for decision-makers
* One-click CSV report export

Why it matters:

Bias in AI models can harm trust, violate compliance (EU AI Act, US AI Bill of Rights), and expose organizations to risk.
This tool enables teams to audit models for fairness, build responsible AI practices, and ensure ethical decision-making.

---

Tech Stack:

* Python (pandas, numpy, scikit-learn)
* Streamlit (interactive web app)
* Matplotlib (charts)

---

How to Run:

# 1. Clone repo
git clone https://github.com/your-username/fairness-auditor.git
cd fairness-auditor

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run app
streamlit run app.py
