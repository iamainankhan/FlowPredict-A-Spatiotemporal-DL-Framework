import pandas as pd

# Read the original H5 file
df = pd.read_hdf("data/METR-LA.h5")

# Convert the timestamp index into a normal column
df = df.reset_index()

# Rename the index column
df = df.rename(columns={"index": "timestamp"})

# Save to CSV
df.to_csv("data/traffic.csv", index=False)

print("CSV created successfully!")
print(df.head())
print(df.shape)