

## 1. A dataset contains the prices of houses in a city. Find the 25th and 75th percentiles and calculate the interquartile range (IQR). How does the IQR help in understanding the price variability?

**PROGRAM:**

```python
import numpy as np

# Example dataset of house prices
house_prices = [250000, 300000, 320000, 350000, 400000, 420000, 450000, 500000, 550000,
600000]

# Calculate the 25th percentile (Q1)
q1 = np.percentile(house_prices, 25)

# Calculate the 75th percentile (Q3)
q3 = np.percentile(house_prices, 75)

# Calculate the Interquartile Range (IQR)
iqr = q3 - q1

# Print results
print("25th percentile (Q1):", q1)
print("75th percentile (Q3):", q3)
print("Interquartile Range (IQR):", iqr)
```

**OUTPUT:**

```
25th percentile (Q1): 327500.0
75th percentile (Q3): 487500.0
Interquartile Range (IQR): 160000.0
```

## 2. You are given a dataset with categorical variables about customer satisfaction levels (Low, Medium, High) and whether customers made repeat purchases (Yes/No). Create visualizations such as bar plots or stacked bar charts to explore the relationship between satisfaction level and repeat purchases. What can you infer from the data?

**PROGRAM:**

```python
import pandas as pd
import matplotlib.pyplot as plt

# dataset
data = {
    'Satisfaction': ['Low', 'Medium', 'High', 'Medium', 'High', 'Low', 'High', 'Medium', 'Low', 'High'],
    'RepeatPurchase': ['No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes']
}

# Create DataFrame
df = pd.DataFrame(data)

# Create a crosstab to count occurrences
cross_tab = pd.crosstab(df['Satisfaction'], df['RepeatPurchase'])

print(cross_tab)

# --- Visualization 1: Grouped Bar Plot ---
cross_tab.plot(kind='bar', figsize=(8,5))
plt.title('Repeat Purchase by Satisfaction Level')
plt.xlabel('Satisfaction Level')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0)
plt.show()

# --- Visualization 2: Stacked Bar Chart ---
cross_tab.plot(kind='bar', stacked=True, figsize=(8,5), color=['red','green'])
plt.title('Repeat Purchase by Satisfaction Level (Stacked)')
plt.xlabel('Satisfaction Level')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0)
plt.show()
```

**OUTPUT:**

```
RepeatPurchase  No  Yes
Satisfaction
High             0    4
Low              3    0
Medium           1    2
```

## 3. A dataset contains information about car models, including the engine size (in Liters), fuel efficiency (miles per gallon), and car price. Use a pair plot or correlation matrix to explore the relationships between these variables. Which variables seem to have the strongest relationships, and what might be the practical significance of these findings?

**PROGRAM:**

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# dataset
data = {
    'EngineSize_L': [1.6, 2.0, 2.5, 3.0, 3.5, 4.0, 2.2, 1.8, 3.2, 2.8],
    'FuelEfficiency_MPG': [35, 30, 28, 25, 22, 20, 29, 33, 24, 26],
    'Price_USD': [20000, 25000, 28000, 32000, 35000, 40000, 27000, 22000, 33000, 31000]
}

# Create DataFrame
df = pd.DataFrame(data)

# --- Pair Plot ---
sns.pairplot(df)
plt.suptitle("Pair Plot of Car Variables", y=1.02)
plt.show()

# --- Correlation Matrix ---
corr_matrix = df.corr()
print("Correlation Matrix:\n", corr_matrix)

# --- Heatmap for better visualization ---
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Car Variables")
plt.show()
```

**OUTPUT:**

```
Correlation Matrix:
                     EngineSize_L  FuelEfficiency_MPG  Price_USD
EngineSize_L            1.000000           -0.985854   0.992210
FuelEfficiency_MPG     -0.985854            1.000000  -0.993457
Price_USD               0.992210           -0.993457   1.000000
```

## 4. You want to estimate the mean salary of software engineers in a country. You take 10 different random samples, each containing 50 engineers, and calculate the sample mean for each. Plot the distribution of these sample means. How does the Central Limit Theorem explain the shape of this sampling distribution, even if the underlying salary distribution is skewed?

**PROGRAM:**

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate population salaries (skewed distribution)
np.random.seed(42)
population_salaries = np.random.exponential(scale=50000, size=10000)  # skewed data

# Number of samples and sample size
num_samples = 10
sample_size = 50

# Calculate sample means
sample_means = []
for _ in range(num_samples):
    sample = np.random.choice(population_salaries, size=sample_size, replace=False)
    sample_means.append(np.mean(sample))

# Plot the distribution of sample means
plt.hist(sample_means, bins=5, edgecolor='black')
plt.title("Distribution of Sample Means (n=50, 10 samples)")
plt.xlabel("Sample Mean Salary")
plt.ylabel("Frequency")
plt.show()

# Print sample means
print("Sample Means:", sample_means)
```

**OUTPUT:**

```
Sample Means: [50407.49240985955, 47441.48917339016, 46359.98208875922,
51123.35940175571, 54792.45402080486, 42987.13036469506, 53695.91997831957,
57123.356348168345, 51519.533296452275, 37272.40519664386]
```

## 5. A researcher conducts an experiment with a sample of 20 participants to determine if a new drug affects heart rate. The sample has a mean heart rate increase of 8 beats per minute and a standard deviation of 2 beats per minute. Perform a hypothesis test using the t-distribution to determine if the mean heart rate increase is significantly different from zero at the 5% significance level.

**PROGRAM:**

```python
from scipy import stats
import math

# Given data
sample_mean = 8      # mean heart rate increase
sample_std = 2       # standard deviation
n = 20               # sample size
alpha = 0.05         # significance level

# Calculate t-statistic
t_stat = sample_mean / (sample_std / math.sqrt(n))

# Degrees of freedom
df = n - 1

# Two-tailed p-value
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))

# Print results
print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.5f}")

# Conclusion
if p_value < alpha:
    print("Reject the null hypothesis: The mean heart rate increase is significantly different from zero.")
else:
    print("Fail to reject the null hypothesis: No significant difference from zero.")
```

**OUTPUT:**

```
T-statistic: 17.889
P-value: 0.00000
Reject the null hypothesis: The mean heart rate increase is significantly different from zero.
```

## 6. A company is testing two versions of a webpage (A and B) to determine which version leads to more sales. Version A was shown to 1,000 users and resulted in 120 sales. Version B was shown to 1,200 users and resulted in 150 sales. Perform an A/B test to determine if there is a statistically significant difference in the conversion rates between the two versions. Use a 5% significance level.

**PROGRAM:**

```python
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

# Given data
successes = np.array([120, 150])  # number of sales for A and B
n = np.array([1000, 1200])        # number of users for A and B
alpha = 0.05                       # significance level

# Perform two-proportion z-test
z_stat, p_value = proportions_ztest(successes, n)

# Print results
print(f"Z-statistic: {z_stat:.3f}")
print(f"P-value: {p_value:.5f}")

# Conclusion
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference in conversion rates between A and B.")
else:
    print("Fail to reject the null hypothesis: No significant difference in conversion rates between A and B.")
```

**OUTPUT:**

```
Z-statistic: -0.356
P-value: 0.72193
Fail to reject the null hypothesis: No significant difference in conversion rates between A and B.
```

## 7. You are comparing the average daily sales between two stores. Store A has a mean daily sales value of $1,000 with a standard deviation of $100 over 30 days, and Store B has a mean daily sales value of $950 with a standard deviation of $120 over 30 days. Conduct a two-sample t-test to determine if there is a significant difference between the average sales of the two stores at the 5% significance level.

**PROGRAM:**

```python
from scipy import stats
import math

# Given data
mean_A = 1000
std_A = 100
n_A = 30

mean_B = 950
std_B = 120
n_B = 30

alpha = 0.05  # significance level

# Calculate the t-statistic for two independent samples (unequal variances)
t_stat = (mean_A - mean_B) / math.sqrt((std_A**2 / n_A) + (std_B**2 / n_B))

# Degrees of freedom using Welch-Satterthwaite equation
df = ((std_A**2 / n_A + std_B**2 / n_B)**2) / (((std_A**2 / n_A)**2 / (n_A - 1)) + ((std_B**2 / n_B)**2 / (n_B - 1)))

# Two-tailed p-value
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))

# Print results
print(f"T-statistic: {t_stat:.3f}")
print(f"Degrees of freedom: {df:.2f}")
print(f"P-value: {p_value:.5f}")

# Conclusion
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference between the average sales of the two stores.")
else:
    print("Fail to reject the null hypothesis: No significant difference between the average sales of the two stores.")
```

**OUTPUT:**

```
T-statistic: 1.753
Degrees of freedom: 56.17
P-value: 0.08502
Fail to reject the null hypothesis: No significant difference between the average sales of the two stores.
```

## 8. A company collects data on employees’ salaries and records their education level as a categorical variable with three levels: “High School”, “Bachelor’s”, and “Master’s”. Fit a multiple linear regression model to predict salary using education level (as a factor variable) and years of experience. Interpret the coefficients for the education levels in the regression model.

**PROGRAM:**

```python
import pandas as pd
import statsmodels.formula.api as smf

# dataset
data = {
    'Salary': [40000, 50000, 60000, 45000, 55000, 65000, 48000, 58000, 70000, 62000],
    'Education': ['High School', "Bachelor's", "Master's", 'High School', "Bachelor's", "Master's", 'High
School', "Bachelor's", "Master's", "Bachelor's"],
    'Experience': [2, 5, 7, 3, 6, 8, 4, 5, 9, 6]
}

df = pd.DataFrame(data)

# Fit multiple linear regression model
# Explicitly set 'High School' as the baseline category
model = smf.ols('Salary ~ Experience + C(Education, Treatment(reference="High School"))',
data=df).fit()

# Print model summary
print(model.summary())
```

**OUTPUT:**

```
                            OLS Regression Results
==============================================================================
Dep. Variable:                 Salary   R-squared:                       0.928
Model:                            OLS   Adj. R-squared:                  0.892
Method:                 Least Squares   F-statistic:                     25.72
Date:                Wed, 03 Sep 2025   Prob (F-statistic):           0.000799
Time:                        08:50:14   Log-Likelihood:                -92.071
No. Observations:                  10   AIC:                             192.1
Df Residuals:                       6   BIC:                             193.4
Df Model:                           3
Covariance Type:            nonrobust
================================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept                                                       3.083e+04   4547.690      6.780      0.001    1.97e+04     4.2e+04
C(Education, Treatment(reference="High School"))[T.Bachelor's]   666.6667   4215.821      0.158      0.880   -9649.076     1.1e+04
C(Education, Treatment(reference="High School"))[T.Master's]   -1833.3333   7411.827     -0.247      0.813      -2e+04    1.63e+04
Experience                                                      4500.0000   1392.440      3.232      0.018    1092.822    7907.178
==============================================================================
Omnibus:                        0.016   Durbin-Watson:                   1.820
Prob(Omnibus):                  0.992   Jarque-Bera (JB):                0.149
Skew:                          -0.002   Prob(JB):                        0.928
Kurtosis:                       2.403   Cond. No.                         55.7
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

## 9. You have data on housing prices and square footage and notice that the relationship between square footage and price is nonlinear. Fit a spline regression model to allow the relationship between square footage and price to change at 2,000 square feet. Explain how spline regression can capture different behaviours of the relationship before and after 2,000 square feet.

**PROGRAM:**

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# dataset
data = {
    'Price': [200000, 250000, 300000, 320000, 350000, 400000, 420000, 450000, 500000, 550000],
    'SqFt': [1500, 1600, 1800, 1900, 2000, 2100, 2200, 2400, 2600, 2800]
}
df = pd.DataFrame(data)

# Define the spline term for SqFt with knot at 2000
df['sqft_knot'] = np.maximum(0, df['SqFt'] - 2000)

# Define independent variables (including intercept)
X = sm.add_constant(df[['SqFt', 'sqft_knot']])
y = df['Price']

# Fit linear regression with spline
model = sm.OLS(y, X).fit()

# Print summary
print(model.summary())

# Plot the fitted spline regression
plt.scatter(df['SqFt'], df['Price'], color='blue', label='Data')
plt.plot(df['SqFt'], model.predict(X), color='red', label='Spline Fit')
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.title('Spline Regression of Price on Square Footage')
plt.legend()
plt.show()
```

**OUTPUT:**

```
                            OLS Regression Results
==============================================================================
Dep. Variable:                  Price   R-squared:                       0.992
Model:                            OLS   Adj. R-squared:                  0.990
Method:                 Least Squares   F-statistic:                     458.5
Date:                Wed, 03 Sep 2025   Prob (F-statistic):           3.78e-08
Time:                        08:53:50   Log-Likelihood:                -105.38
No. Observations:                  10   AIC:                             216.8
Df Residuals:                       7   BIC:                             217.7
Df Model:                           2
Covariance Type:            nonrobust
================================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
const      -2.557e+05   4.11e+04     -6.217      0.000   -3.53e+05   -1.58e+05
SqFt         308.7019     22.587     13.667      0.000     255.293     362.111
sqft_knot    -73.5822     32.468     -2.266      0.058    -150.356       3.191
================================================================================
Omnibus:                        1.654   Durbin-Watson:                   1.972
Prob(Omnibus):                  0.437   Jarque-Bera (JB):                0.915
Skew:                           0.381   Prob(JB):                        0.633
Kurtosis:                       1.729   Cond. No.                     2.55e+04
================================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 2.55e+04. This might indicate that there are
strong multicollinearity or other numerical problems.
```

## 10. A hospital is using a Poisson regression model (a type of GLM) to predict the number of emergency room visits per week based on patient age and medical history. The model is given by: • Log(λ) =2.5-0.03Age+0.5condition where λ is the expected number of visits per week, Age is the patient’s age, and condition is a binary variable (1 if the patient has a chronic condition, 0 otherwise). Interpret the coefficients of Age and condition. • What is the expected number of visits per week for a 60-year-old patient with a chronic condition? • How would the expected number of visits change if the patient did not have a chronic condition?

**PROGRAM:**

```python
import numpy as np

# Given Poisson regression coefficients
intercept = 2.5
coef_age = -0.03
coef_condition = 0.5

# Patient information
age = 60
condition = 1  # 1 if patient has chronic condition, 0 otherwise

# Calculate log(lambda)
log_lambda = intercept + coef_age * age + coef_condition * condition

# Convert to expected number of visits
expected_visits = np.exp(log_lambda)
print(f"Expected number of visits (with chronic condition): {expected_visits:.2f}")

# If patient does NOT have chronic condition
condition = 0
log_lambda_no_condition = intercept + coef_age * age + coef_condition * condition
expected_visits_no_condition = np.exp(log_lambda_no_condition)
print(f"Expected number of visits (without chronic condition): {expected_visits_no_condition:.2f}")
```

**OUTPUT:**

```
Expected number of visits (with chronic condition): 3.32
Expected number of visits (without chronic condition): 2.01
```
## 11. A researcher is comparing the calorie content of a new recipe to an old one. The old recipe has a mean of 200 calories. A sample of 40 servings of the new recipe has a mean of 190 calories and a standard deviation of 15 calories. Perform a one-sample t-test to determine if the new recipe has significantly fewer calories at the 5% significance level.

**PROGRAM:**

```python
from scipy import stats
import math

# Given data
old_mean = 200        # mean calories of old recipe
sample_mean = 190     # mean calories of new recipe
sample_std = 15       # standard deviation of new recipe
n = 40                # sample size
alpha = 0.05          # significance level

# Calculate t-statistic
t_stat = (sample_mean - old_mean) / (sample_std / math.sqrt(n))

# Degrees of freedom
df = n - 1

# One-tailed p-value (testing if new mean < old mean)
p_value = stats.t.cdf(t_stat, df=df)

# Print results
print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.5f}")

# Conclusion
if p_value < alpha:
    print("Reject the null hypothesis: The new recipe has significantly fewer calories.")
else:
    print("Fail to reject the null hypothesis: No significant difference in calories.")
```

**OUTPUT:**

```
T-statistic: -4.714
P-value: 0.00002
Reject the null hypothesis: The new recipe has significantly fewer calories.
```
