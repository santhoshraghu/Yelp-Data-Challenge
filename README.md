# Yelp-Data-Challenge
## README: Yelp Data Challenge - Business Attribute Prediction

The goal was to predict business attributes utilizing textual information extracted from reviews and tips. Leveraging natural language processing (NLP) techniques, the project aimed to uncover patterns and sentiments correlating with various business characteristics. By employing advanced machine learning algorithms, the study aimed to offer actionable recommendations for businesses based on insights derived from Yelp's extensive dataset.

### Methodology Summary:
1. **Data Loading & Text Pre-processing**:
   - Loaded dataset from JSON files using pandas.
   - Conducted basic pre-processing such as handling missing values, filtering data, and splitting into train and test sets.
   - Performed text pre-processing steps including tokenization, stopword removal, lemmatization, and stemming using NLTK.

2. **Text Pre-processing & Feature Engineering**:
   - Converted text data into numerical features using TF-IDF vectorization and Bag of Words representation.
   - Explored Word2Vec embeddings for text data.
   - Tokenized text data using various methods including RoBERTa tokenizer.

3. **Model Configuration**:
   - Trained various machine learning models over different feature sets:
     - Naive Bayes
     - Bernoulli Naive Bayes
     - Random Forest
     - Decision Tree
     - XGBoost
     - RoBERTa (Transformer model)

4. **Training and Evaluation**:
   - Trained models separately using training data and evaluated performance on validation sets.
   - Evaluated models based on accuracy and F1 score metrics.

5. **Evaluation Metrics and Results**:
   - Reported accuracy and F1 score for each model and feature representation method.
   - Summarized evaluation results in a tabular format for comparison.

### Analysis and Conclusion:
- **Naive Bayes Models**:
  - Demonstrated good performance, with TF-IDF outperforming Bag of Words.
  - Achieved accuracy ranging from 0.8531 to 0.8644 and F1 scores ranging from 0.85 to 0.86.

- **Gaussian Naive Bayes**:
  - Achieved relatively lower accuracy and F1 score with Word2Vec embeddings.

- **Bernoulli Naive Bayes**:
  - Showed moderate performance across different feature sets, comparable to Gaussian Naive Bayes with Word2Vec embeddings.

- **Random Forest**:
  - Exhibited strong performance across all feature sets, slightly better with TF-IDF and BoW.

- **Decision Tree**:
  - Achieved moderate performance, with Word2Vec embeddings performing relatively better.

- **XGBoost**:
  - Demonstrated strong performance, similar to Random Forest, with TF-IDF and BoW performing slightly better.

- **RoBERTa**:
  - Achieved the highest accuracy of 0.92, indicating superior performance in predicting star ratings.

In conclusion, RoBERTa outperformed traditional models and word embedding approaches in terms of accuracy. However, factors such as computational complexity and scalability need consideration for deployment. Further analysis and tuning could provide deeper insights into model performance and generalizability.
