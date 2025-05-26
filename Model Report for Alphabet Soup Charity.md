# Neural Network Model Report for Alphabet Soup

## Overview of the Analysis

The purpose of this analysis was to create a binary classifier using deep learning techniques to predict whether applicants funded by Alphabet Soup will be successful in their ventures. Using machine learning and neural networks, I analyzed a dataset of more than 34,000 organizations that have received funding from Alphabet Soup over the years. The goal was to create a predictive model that could help the foundation select applicants with the best chance of success, ultimately optimizing their funding decisions and maximizing the impact of their charitable contributions.

## Results

### Data Preprocessing

• **Target Variable:**
  - `IS_SUCCESSFUL` - This binary variable indicates whether the funding was used effectively (1 for successful, 0 for unsuccessful)

• **Feature Variables:**
  - `APPLICATION_TYPE` - Alphabet Soup application type (after binning rare categories as "Other")
  - `AFFILIATION` - Affiliated sector of industry
  - `CLASSIFICATION` - Government organization classification (after binning rare categories as "Other")
  - `USE_CASE` - Use case for funding
  - `ORGANIZATION` - Organization type
  - `STATUS` - Active status
  - `INCOME_AMT` - Income classification
  - `SPECIAL_CONSIDERATIONS` - Special considerations for application
  - `ASK_AMT` - Funding amount requested

• **Variables Removed:**
  - `EIN` - Employer Identification Number (unique identifier, not predictive)
  - `NAME` - Organization name (unique identifier, not predictive)

### Compiling, Training, and Evaluating the Model

• **Base Model Architecture:**
  - **Input Layer:** 43 features (after preprocessing and one-hot encoding)
  - **First Hidden Layer:** 80 neurons with ReLU activation
  - **Second Hidden Layer:** 30 neurons with ReLU activation
  - **Output Layer:** 1 neuron with sigmoid activation (for binary classification)
  - **Total Parameters:** Approximately 4,000+ trainable parameters

• **Rationale for Architecture:**
  - Used ReLU activation for hidden layers to prevent vanishing gradient problem
  - Sigmoid activation for output layer appropriate for binary classification
  - Started with roughly 2x the input features for first layer (common rule of thumb)
  - Second layer with fewer neurons to create a funnel effect

• **Target Performance Achievement:**
  - **Target:** 75% accuracy
  - **Result:** Did not achieve target performance
  - **Best Performance:** 73.19% accuracy (Attempt 2)

• **Optimization Attempts:**

  **Attempt 1 - Reduced Neurons:**
  - Reduced neurons in hidden layers
  - **Result:** 72.52% accuracy (0.24% decrease from base model)
  - **Analysis:** Reducing model capacity hurt performance

  **Attempt 2 - Increased Capacity and Training:**
  - Doubled neurons in both hidden layers
  - Increased epochs from 100 to 200
  - **Result:** 73.19% accuracy (0.43% improvement from base model)
  - **Analysis:** More capacity and training time improved performance

  **Attempt 3 - Architecture and Hyperparameter Changes:**
  - Doubled neuron amounts
  - Added custom learning rate of 0.001
  - Changed activation function to tanh
  - **Result:** 72.87% accuracy (0.11% improvement from base model)
  - **Analysis:** tanh activation and lower learning rate provided modest improvement

## Summary

The deep learning model achieved a maximum accuracy of 73.19%, falling short of the 75% target by approximately 1.8 percentage points. While the model demonstrates reasonable predictive capability, there is room for improvement in identifying successful funding applicants.

**Key Findings:**
- Increasing model capacity (more neurons) and training time yielded the best results
- The dataset may have inherent limitations that prevent higher accuracy
- Feature engineering and data preprocessing were crucial for model performance

**Recommendation for Alternative Approach:**

I recommend implementing a **Random Forest Classifier** or **Gradient Boosting Machine (XGBoost)** for this classification problem for the following reasons:

1. **Feature Importance:** Tree-based models provide clear feature importance rankings, helping Alphabet Soup understand which factors most influence funding success

2. **Handling Mixed Data Types:** These models naturally handle the mix of categorical and numerical features without extensive preprocessing

3. **Robustness:** Less prone to overfitting and more interpretable than deep neural networks

4. **Performance:** Often achieve comparable or better performance on tabular data like this dataset

5. **Efficiency:** Faster training and prediction times, making them more practical for operational use

**Implementation Strategy:**
- Start with Random Forest to establish baseline performance and identify key features
- Use feature importance insights to guide further feature engineering
- Implement XGBoost with hyperparameter tuning for potentially higher accuracy
- Consider ensemble methods combining multiple algorithms for optimal performance

This alternative approach would likely provide better interpretability for business decisions while potentially achieving the target 75% accuracy through more appropriate algorithm selection for this tabular dataset.