# ⚡ Electricity Consumption Forecasting & Grid Stability Prediction

📌 Project Overview
This project predicts household electricity consumption to ensure smart grid stability. It utilizes the **XGBoost** algorithm to provide a 24-hour predictive horizon, helping grid operators identify potential power surges before they occur.

The system is designed to enable "Peak Shaving," helping utility providers reduce financial risk and prevent transformer overloads.

⚙️ Development Workflow
* The Machine Learning model was developed and researched using **Jupyter Notebooks**.
* The application and user interface were built and deployed using **VS Code**.
* The trained model was serialized and integrated into the application for real-time forecasting.

🚀 Features
* 🔍 **Predicts Global Active Power (kW)** for the next 24 hours.
* 🎯 **Feature Engineering** including Lag 1h and Lag 24h momentum tracking.
* 🚦 **Real-time Alert System** (Stable / Warning / Critical) based on safety thresholds.
* 📊 **Visual Demand Curve** using interactive time-series plots.
* 🧠 **System Integrity Module** to verify model accuracy using RMSE.
* 💻 **User-friendly interface** built with Streamlit.

📂 Dataset
This project uses the Individual Household Electric Power Consumption dataset to predict energy demand.
* 📥 **Dataset source:** UCI Machine Learning Repository.
* 📂 **Instructions:** Download the dataset, extract the files, and place them inside the `data/` folder.

📊 Features Used:
* Global Active Power (Target Variable)
* Voltage & Global Intensity
* Sub-metering 1, 2, and 3
* Temporal Features (Hour, Day, Month)
* Lag Features (Historical Momentum)

🛠️ Technologies Used
* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Streamlit
* Plotly (for interactive visualizations)

📊 How It Works
1.  **Data Collection:** Loading raw household energy sensor data.
2.  **Data Preprocessing:** Handling missing values and resampling data.
3.  **Feature Engineering:** Extracting time cycles and historical lags.
4.  **Model Training:** Training the XGBoost regressor on consumption signatures.
5.  **Prediction:** Generating a 24-hour look-ahead forecast.
6.  **Visualization:** Displaying metrics and alerts on the dashboard.

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
📷 Output
The system provides:

Electricity Demand Prediction (kW): Real-time forecasting of power usage.

24-Hour Predictive Horizon Graph: Visual demand curve for the next day.

Real-time Status Alerts: Color-coded system notifications for grid health.

Performance Error Metrics: Detailed accuracy reporting using RMSE.

🎯 Demand Interpretation
🟢 Below Threshold → Stable Grid

🟡 Approaching Limit → Warning / Load Monitoring

🔴 Above Threshold → Critical / Peak Shaving Required

👩‍💻 Author
VIGNESH KV Register No: 2328K0039
