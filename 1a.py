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