# ResumeClassifier

**ResumeClassifier** is a machine learning project designed to automate resume categorization. By utilizing Natural Language Processing (NLP) and machine learning algorithms, it can analyze the text content of resumes and classify them into relevant job categories like Data Science, Mechanical Engineering, HR, and more. This tool is ideal for HR teams, recruiters, and job boards to enhance resume screening efficiency.

## Features

- **Data Preprocessing**: Cleans resume text, removing URLs, special characters, and symbols for accurate analysis.
- **TF-IDF Vectorization**: Transforms resume text into numerical features based on word relevance.
- **Multiclass Classification**: Uses a k-Nearest Neighbors (k-NN) model with One-vs-Rest classification to assign resumes to job roles.
- **Category Mapping**: Maps category IDs to job names for accurate labeling.
- **Visualizations**: Displays category distribution in the dataset with Seaborn.

## Technologies Used

- **Programming Language**: Python
- **Libraries**: 
  - Data Processing: `NumPy`, `Pandas`
  - Visualization: `Matplotlib`, `Seaborn`
  - Machine Learning & NLP: `sklearn` (LabelEncoder, TfidfVectorizer, KNeighborsClassifier)

## Getting Started

### Prerequisites

- Python 3.6 or higher
- Install the necessary packages:
  
  ```bash
  pip install numpy pandas matplotlib seaborn scikit-learn
  ```

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/ResumeClassifier.git
   cd ResumeClassifier
   ```
2. **Download the Dataset**: Place `UpdatedResumeDataSet.csv` in the project directory.

### Usage

1. **Run the Jupyter Notebook** or script to preprocess and train the model on the dataset:
   - The model uses `TfidfVectorizer` to vectorize the resume text.
   - A `k-NN` classifier (with One-vs-Rest strategy) is trained to classify resumes.
  
2. **Classify New Resumes**:
   - Use `cleanResume()` to preprocess new resumes.
   - Use the trained model to predict categories for new resumes.

3. **Visualize Data**:
   - The notebook contains code to visualize category distribution in the dataset for exploration.

### Example Code

```python
# Clean a new resume and predict its category
cleaned_resume = cleanResume(myresume)
input_features = tfidf.transform([cleaned_resume])
prediction_id = clf.predict(input_features)[0]
category_name = category_mapping.get(prediction_id, "Unknown")
print("Predicted Category:", category_name)
```

## Project Structure

- **`UpdatedResumeDataSet.csv`**: The dataset containing resumes and their categories.
- **`ResumeClassifier.ipynb`**: Jupyter Notebook with the full code for preprocessing, training, and testing the model.
- **`category_mapping`**: Maps the predicted labels to job categories.

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

This README file includes setup instructions, usage examples, and a project overview that should help users and contributors understand and navigate the project. Let me know if you'd like any adjustments!
