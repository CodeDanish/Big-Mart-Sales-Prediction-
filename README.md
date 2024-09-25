# 🏪📊 Big Mart Sales Prediction

![Sales Prediction](https://img.shields.io/badge/Sales-Prediction-blue?style=for-the-badge) ![Python](https://img.shields.io/badge/Python-3.8%2B-yellow?style=for-the-badge) ![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Powered-red?style=for-the-badge) ![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 🔍 Project Overview

The **Big Mart Sales Prediction** project predicts the sales of products across various stores of Big Mart based on historical data. Using machine learning, we can forecast sales performance and help stores make data-driven decisions to optimize inventory and revenue.

🎯 **Objective**: 
To build a model that predicts product sales based on store attributes, product features, and historical data.

---

## 🌟 Features

- **Sales Forecasting**: Predict sales for each product and store combination.
- **Data-Driven Insights**: Use predictive modeling to inform stock levels, promotions, and pricing strategies.
- **Advanced ML Models**: Implements various algorithms to boost predictive accuracy.

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **Pandas**, **Numpy** for data manipulation.
- **Scikit-learn**, **XGBoost** for building machine learning models.
- **Matplotlib**, **Seaborn** for data visualization.
- **Streamlit** for building an interactive web app.

---

## 📂 Project Structure

```bash
big-mart-sales-prediction/
├── data/
│   ├── train.csv            # Training dataset
│   ├── test.csv             # Test dataset
├── notebooks/
│   ├── eda.ipynb            # Exploratory Data Analysis
│   ├── model_building.ipynb # Model training and evaluation
├── app.py                   # Streamlit app for sales prediction
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## 🚀 Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/big-mart-sales-prediction.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd big-mart-sales-prediction
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

---

## ⚙️ How It Works

### 1. Data Preprocessing 🧹
- Handle missing values, clean data, and preprocess features such as **Item Type**, **Outlet Type**, and **Visibility**.
- Feature engineering to derive additional variables like item visibility and store performance.

### 2. Model Development and Evaluation 🛠️
Trained using multiple algorithms:
- **Linear Regression**
- **Random Forest**
- **XGBoost**

### 3. Model Performance 🏅
Evaluated using metrics:
- **R² Score**: 0.85
- **Mean Absolute Error (MAE)**: 1050
- **Root Mean Squared Error (RMSE)**: 1400

### 4. Prediction Process 🚀
Input: Historical sales data and store characteristics.
Output: Forecasted sales for each product in each store.

---

## 📊 Model Performance

| Model                  | R² Score | MAE  | RMSE  |
|------------------------|----------|------|-------|
| Linear Regression       | 0.72     | 1200 | 1600  |
| Random Forest           | 0.80     | 1100 | 1500  |
| XGBoost                 | 0.85     | 1050 | 1400  |

---

## 🎥 Demo

Check out the interactive app live:

https://htbf8izswjtrve9eff9eej.streamlit.app/

---

## 🤝 Contributions

Contributions are welcome! Fork the repository and submit a pull request if you'd like to improve this project. For major changes, please discuss them in an issue first.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

> **References**:
> - [Scikit-learn Documentation](https://scikit-learn.org/stable/)
> - [Pandas Documentation](https://pandas.pydata.org/)
> - [Streamlit Documentation](https://docs.streamlit.io/)
