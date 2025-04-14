import pandas as pd
import numpy as np


df = pd.read_csv('UP-5_Impact_Modeling_SynData.csv')

# Set the desired number of rows
n_rows = 5000

# Perform bootstrapping (sampling with replacement)
bootstrapped_df = df.sample(n=n_rows, replace=True, random_state=42)


bootstrapped_df.to_csv('bootstrapped_UP-5.csv', index=False)

print("Bootstrapped dataset saved as 'bootstrapped_UP-5.csv' with",
      len(bootstrapped_df), "rows.")
