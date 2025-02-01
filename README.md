# Book Recommendation System

## 📌 Project Overview
This project is a **Book Recommendation System** that leverages **deep learning** to suggest books based on user preferences. The model is trained on book ratings and metadata to provide personalized recommendations. The system is deployed using **Streamlit** for an interactive user experience.

## 📂 Dataset
The dataset includes:
- **Book Details:** Title, author, genre, publication year, etc.
- **User Ratings:** Ratings provided by users for different books.
- **Interaction Data:** User-book interactions for training the recommendation model.

## 🛠 Technologies Used
- **Programming Language:** Python
- **Deep Learning Framework:** TensorFlow, Keras
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Recommendation Algorithms:** Neural Collaborative Filtering (NCF), Convolutional Neural Networks (CNN), Graph Neural Networks (GNN)
- **Deployment:** Streamlit

## 🔍 Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Handling missing values and duplicates
- Feature engineering and embeddings
- Data visualization (distribution of ratings, popular books, user activity)

## 📊 Model Training & Evaluation
- **Feature Engineering:** Convert books and users into embedding vectors
- **Model Architecture:** Build and train deep learning models (e.g., NCF, CNN, GNN)
- **Evaluation Metrics:** Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Precision@K, Recall@K
- **Hyperparameter Tuning:** Optimize model performance

## 🚀 How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/book-recommendation-system.git
   ```
2. Navigate to the project directory:
   ```bash
   cd book-recommendation-system
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Train the model (if not already trained):
   ```bash
   python train_model.py
   ```
5. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
6. Open the provided **localhost URL** in your browser to use the recommendation system.


## 🎭 Features
- **Personalized Book Recommendations** based on user preferences.
- **Search Functionality** to find books by title, author, or genre.
- **Interactive UI with Streamlit** for an engaging experience.
- **Real-time Predictions** powered by deep learning.

## 🔮 Future Enhancements
- Implement Reinforcement Learning for recommendations.
- Use NLP-based embeddings for improved book representations.
- Deploy the model as a web service using FastAPI or Flask.

## 🤝 Contributing
Contributions are welcome! Feel free to fork this repository and submit a pull request.

## 📜 License
This project is licensed under the MIT License.

## 📬 Contact
For any questions, reach out via [muskanraj2702@gmail.com] or connect on [LinkedIn](https://www.linkedin.com/in/muskan-raj-2a3613221/).
