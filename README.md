# Employee Turnover Prediction
Employee Turnover Prediction
This project aims to predict employee turnover using machine learning models. The application is built with Python and Streamlit, providing an interactive frontend for users to input features and receive predictions on whether an employee will leave the company.<br>

Employee Turnover is the measurement of the total number of employees who leave an organization in a particular year. Employee Turnover Prediction means to predict whether an employee is going to leave the organization in the coming period.<br>

A Company uses this predictive analysis to measure how many employees they will need if the potential employees will leave their organization. A company also uses this predictive analysis to make the workplace better for employees by understanding the core reasons for the high turnover ratio.<br>

Here 'left' is the target (dependent) variable <br>

# Steps

1. Import Library and Dataset
2. Data Preprocessing
    - Gethering dataset info
    - Data Statistics
    - Missing value handling 
    - columns operation
3. Categorical to Numerical 
4. Data Visualization
    - Get Insights
    - Outlier detection
    - Correlation 
5. Feature Selection
6. Train and Test Split
7. Model Building
    - Logistic Regression
    - Decision Tree
    - Random Forest
8. Model Evaluation

# Model Details
Random Forest<br>
A powerful ensemble learning method that combines multiple decision trees.
Provides feature importance scores indicating the impact of each feature on the prediction.
Logistic Regression<br>
A simple yet effective linear model for binary classification.
Suitable for interpreting the relationship between features and the target variable.
Evaluation Metrics<br>
Accuracy: Overall correctness of the model.
Confusion Matrix: Detailed breakdown of true positives, true negatives, false positives, and false negatives.
ROC Curve: Graphical representation of the model's performance at different threshold levels.

## Installation

To get started with the project, follow these steps:

1. Clone the repository:
  
  
    ```bash
    git clone https://github.com/yourusername/Employee_Turnover_Prediction.git
    ```
    
2. Navigate to the Project directory:
   ```bash
    cd Employee_Turnover_Prediction
    ```
3. Create and activate a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    ```
    
     On Windows:
             
       venv\Scripts\activate
      
     On macOS/Linux:
   
       source venv/bin/activate

4. Install the required dependencies:
   
    ```bash
    pip install -r requirements.txt
    ```
## Usage

 Run the Streamlit application:
 
    ```bash
    streamlit run predict.py
    ```
    This will start the Streamlit server and open the application in your web browser. You can interact with the app by adjusting the input features on the sidebar and clicking the "Predict" button to see the prediction results.