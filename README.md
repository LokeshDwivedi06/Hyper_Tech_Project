
📈 Stock Price Prediction Web App
This project is a Stock Market Prediction Web Application built using LSTM (Deep Learning). It allows users to input a stock symbol and visualize historical trends along with predicted stock prices.
🚀 Features
📊 Fetch real-time stock data using Yahoo Finance
📉 Visualize stock trends with Moving Averages (MA50, MA100, MA200)
🤖 Predict stock prices using a trained LSTM model
🌐 Interactive web interface built with Streamlit
🛠️ Tech Stack
Python 3.10
Streamlit
TensorFlow / Keras
Pandas, NumPy
Matplotlib
yFinance
Scikit-learn
⚙️ Installation & Setup
1️⃣ Clone the Repository
Bash
git clone https://github.com/your-username/stock-price-predictor.git
cd stock-price-predictor
2️⃣ Create Virtual Environment (Recommended)
Bash
python -m venv venv
Activate environment:
Windows:
Bash
venv\Scripts\activate
Mac/Linux:
Bash
source venv/bin/activate
3️⃣ Install Dependencies
Bash
pip install numpy pandas yfinance keras tensorflow streamlit matplotlib scikit-learn
4️⃣ Add Model File
Make sure your trained model file is placed correctly:

Stock Predictions Model.keras
And update path in code if needed:
Python
model = load_model('Stock Predictions Model.keras')
5️⃣ Run the Application
Bash
streamlit run app.py
📊 How It Works
User enters a stock symbol (e.g., GOOG, AAPL)
App fetches historical data using yfinance
Data is split into training (80%) and testing (20%)
Data is scaled using MinMaxScaler
LSTM model predicts future prices
Graphs are plotted:
Price vs MA50
Price vs MA100
Price vs MA200
Original vs Predicted Prices
📂 Project Structure

├── app.py
├── Stock Predictions Model.keras
├── README.md
⚠️ Important Notes
Use Python 3.10 for best compatibility
TensorFlow may not work properly on Python 3.11+
Internet connection required for fetching stock data
💡 Example Stock Symbols
Google → GOOG
Apple → AAPL
Tesla → TSLA
Microsoft → MSFT
📸 Output Preview
Stock data table
Moving averages graphs
Prediction vs actual comparison
👨‍💻 Author
Your Name
B.Tech / BCA / MCA Student
Project: Stock Price Prediction using LSTM
📌 Future Improvements
Add multiple stock comparison
Improve prediction accuracy
Deploy on cloud (AWS / Streamlit Cloud)
Add real-time prediction
⭐ Acknowledgements
Yahoo Finance API
TensorFlow & Keras
Streamlit
