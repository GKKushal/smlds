# VTU Circle Program Summaries

## 1. House Prices Percentiles and IQR

**Program Name:** Calculating 25th and 75th percentiles and IQR for house prices.

**Result:**
```
25th percentile (Q1): 327500.0
75th percentile (Q3): 487500.0
Interquartile Range (IQR): 160000.0
```

**Interpretation:** The IQR of 160,000 indicates the spread of the middle 50% of house prices, showing moderate variability. A larger IQR suggests greater price differences among typical houses, helping identify outliers (prices below Q1 - 1.5*IQR or above Q3 + 1.5*IQR).

## 2. Customer Satisfaction and Repeat Purchases Visualization

**Program Name:** Exploring relationship between satisfaction levels and repeat purchases using bar plots and crosstab.

**Result:**
```
RepeatPurchase  No  Yes
Satisfaction
High             0    4
Low              3    0
Medium           1    2
```

**Interpretation:** Higher satisfaction levels correlate with more repeat purchases. All high-satisfaction customers made repeat purchases, while low-satisfaction ones did not. Medium shows mixed results. This suggests improving satisfaction could boost repeat business.

## 3. Car Variables Relationships

**Program Name:** Analyzing relationships between engine size, fuel efficiency, and price using pair plot and correlation matrix.

**Result:**
```
Correlation Matrix:
                     EngineSize_L  FuelEfficiency_MPG  Price_USD
EngineSize_L            1.000000           -0.985854   0.992210
FuelEfficiency_MPG     -0.985854            1.000000  -0.993457
Price_USD               0.992210           -0.993457   1.000000
```

**Interpretation:** Engine size and price have a strong positive correlation (0.99), while fuel efficiency negatively correlates with both (-0.99 and -0.98). Larger engines cost more but are less efficient. This highlights trade-offs in car design for performance vs. economy.

## 4. Sampling Distribution of Means

**Program Name:** Plotting distribution of sample means from a skewed population to demonstrate Central Limit Theorem.

**Result:**
```
Sample Means: [50407.49240985955, 47441.48917339016, 46359.98208875922,
51123.35940175571, 54792.45402080486, 42987.13036469506, 53695.91997831957,
57123.356348168345, 51519.533296452275, 37272.40519664386]
```

**Interpretation:** The sample means form a distribution approximating normality, as per CLT. Even with a skewed exponential population, the means cluster around the true mean, with variability decreasing as sample size increases, enabling reliable inference.

## 5. Hypothesis Test for Heart Rate Increase

**Program Name:** One-sample t-test to check if mean heart rate increase differs from zero.

**Result:**
```
T-statistic: 17.889
P-value: 0.00000
Reject the null hypothesis: The mean heart rate increase is significantly different from zero.
```

**Interpretation:** With p-value < 0.05, the drug significantly increases heart rate by 8 bpm on average. This suggests the drug has a measurable physiological effect, warranting further study on safety and efficacy.

## 6. A/B Test for Webpage Conversion Rates

**Program Name:** Two-proportion z-test comparing conversion rates of webpage versions A and B.

**Result:**
```
Z-statistic: -0.356
P-value: 0.72193
Fail to reject the null hypothesis: No significant difference in conversion rates between A and B.
```

**Interpretation:** No significant difference in sales conversion between versions (A: 12%, B: 12.5%). The slight edge for B is not statistically meaningful, so either version can be used without expecting different outcomes.

## 7. Two-Sample T-Test for Store Sales

**Program Name:** Comparing average daily sales between two stores using two-sample t-test.

**Result:**
```
T-statistic: 1.753
Degrees of freedom: 56.17
P-value: 0.08502
Fail to reject the null hypothesis: No significant difference between the average sales of the two stores.
```

**Interpretation:** Store A's higher mean sales ($1000 vs $950) is not statistically significant (p=0.085 > 0.05). Differences may be due to chance, suggesting similar performance or need for larger samples.

## 8. Multiple Linear Regression for Salary Prediction

**Program Name:** Fitting regression model predicting salary from education level and experience.

**Result:**
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
```

**Interpretation:** Experience significantly increases salary ($4500/year), while education effects are not significant. Bachelor's adds ~$667 (p=0.88), Master's subtracts ~$1833 (p=0.81) vs high school. Model explains 92% variance, but small sample limits reliability.

## 9. Spline Regression for Housing Prices

**Program Name:** Fitting spline regression for nonlinear relationship between square footage and price.

**Result:**
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
```

**Interpretation:** Spline captures nonlinearity: price increases $309/sq ft up to 2000 sq ft, then $235/sq ft after (309-74). This allows modeling different behaviors, like diminishing returns for larger homes, improving fit over linear models.

## 10. Poisson Regression for Emergency Visits

**Program Name:** Interpreting coefficients and predicting visits using Poisson regression model.

**Result:**
```
Expected number of visits (with chronic condition): 3.32
Expected number of visits (without chronic condition): 2.01
```

**Interpretation:** Age coefficient (-0.03) means visits decrease 3% per year. Condition coefficient (0.5) means 65% more visits with chronic condition. For 60-year-old with condition, expect 3.32 visits/week; without, 2.01 – a 65% increase due to condition.

## 11. One-Sample T-Test for Recipe Calories

**Program Name:** Testing if new recipe has significantly fewer calories than old one.

**Result:**
```
T-statistic: -4.714
P-value: 0.00002
Reject the null hypothesis: The new recipe has significantly fewer calories.
```

**Interpretation:** New recipe's mean 190 cal is significantly lower than old's 200 cal (p<0.05). This indicates a real reduction, potentially improving health appeal, though clinical significance depends on context.
