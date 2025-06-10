# Neural Network Model Report for Alphabet Soup

# Neural Network Model Report for Alphabet Soup

## Overview of the Analysis

The purpose of this analysis was to create a resource for nonprofits to use to aid in the expectation of whether applicants funded by Alphabet Soup will be successful in their ventures. I fed the model historical data of more than 34,000 organizations that have received funding from Alphabet Soup in the past to predict future outcomes by adjusting it's architecture and parameters to improve the predictive accuracy of the model. 

## Results:
### Data Preprocessing

• **Target Variable:**
  - `IS_SUCCESSFUL` - This binary variable was targeted because of it's desired outcome of the model (1 for successful, 0 for unsuccessful)

• **Feature Variables:**
  - `APPLICATION_TYPE` 
  - `AFFILIATION` 
  - `CLASSIFICATION` 
  - `USE_CASE` 
  - `ORGANIZATION` 
  - `STATUS` 
  - `INCOME_AMT` 
  - `SPECIAL_CONSIDERATIONS` 
  - `ASK_AMT` 

• **Variables Removed:** Removed to reduce noise in model as they are unique values, not characteristics of predictability
  - `EIN` - Employer Identification Number (not predictive)
  - `NAME` - Organization name (not predictive)

### Compiling, Training, and Evaluating the Model

• **Base Model Architecture:**
  - **Input Layer:** 43 features (after preprocessing and one-hot encoding)
  - **First Hidden Layer:** 80 neurons with ReLU activation (Almost double the amount of the input features)
  - **Second Hidden Layer:** 30 neurons with ReLU activation (Smaller second layer to create a funnel effect)
  - **Output Layer:** 1 neuron with sigmoid activation (for binary classification)
  - **The Why:** A control group with a healthy number of neurons was needed before assessing the model for changes. ReLU is the best activation to use for the first and second hidden layers as we were building the model off an "if" statement that required minimal computational power. Sigmoid is best for the output layer as the model needs to provide a "yes" or "no" answer keeping it simplististic and minimizing the oppurtunity to over saturate our model.  

• **Target Performance Achievement:**
  - **Target:** 75% accuracy
  - **Result:** Did not achieve target performance
  - **Best Performance:** 72.99% accuracy (Attempt 1 and 2)

• **Optimization Attempts:**

  **Attempt 1 - Reduced Neurons:**
  - Reduced neurons in hidden layers
  - **Result:** 72.99% accuracy (0.14% increase from base model)
  - **Analysis:** Reducing model capacity hurt performance

  **Attempt 2 - Increased Capacity and Training:**
  - Doubled neurons in both hidden layers
  - Increased epochs from 100 to 200
  - **Result:** 72.99% accuracy (0.14% increase from base model)
  - **Analysis:** More capacity and training time improved performance

  **Attempt 3 - Architecture and Hyperparameter Changes:**
  - Doubled neuron amounts
  - Added custom learning rate of 0.001
  - Changed activation functions of layer 1 and 2 to tanh
  - **Result:** 72.62% accuracy (0.23% reduction from base model)
  - **Analysis:** tanh activation and lower learning rate provided modest improvement

![image](https://github.com/user-attachments/assets/11589289-8573-4e5a-8143-f54ab88ab2dc)


## Summary

The deep learning model achieved a maximum accuracy of 72.99%, falling short of the 75% target by approximately 2.01 percentage points. While the model demonstrates reasonable predictive capability, there is room for improvement in identifying successful funding applicants.

**Key Findings:**
- Increasing model capacity (more neurons) and training time yielded the best results
- The dataset may have inherent limitations that prevent higher accuracy
- Feature engineering and data preprocessing were crucial for model performance

**Recommendation for Alternative Approach:**

I recommend using **Random Forest** for this classification problem for the following reasons:

1. **Feature Importance:** Tree-based models, like Random Forest, provide clear feature importance rankings. Weight is placed more on the desired feature thus prioritizing modeling that outcome.

2. **Robustness:** Less prone to overfitting and more interpretable than deep neural networks

3. **Performance:** Often achieve comparable or better performance on tabular data like this dataset

4. **Efficiency:** Faster training and prediction times, making them more practical for operational use

The nature of this data set is tabular in style. This alternative approach would likely provide better interpretability for business decisions while potentially achieving the target accuracy of 75% or more through more as it is more appropriate for data sets of this style.
