import pandas as pd
import numpy as np

# Generate dummy Excel file
t = np.arange(0, 10, 1/250.0)
y = np.sin(2 * np.pi * 1.0 * t) + np.random.normal(0, 0.1, size=len(t))

df = pd.DataFrame({
    "Begin (seconds)": t,
    "Sampling Frequency": [250.0] * len(t),
    "y 1": y
})
df.to_excel("dummy_input.xlsx", sheet_name="PixelCorrelation Segment 1", index=False)
print("Created dummy_input.xlsx")
