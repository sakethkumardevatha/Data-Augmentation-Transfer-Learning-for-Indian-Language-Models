import pandas as pd

# Telugu
te = pd.read_csv("data/telugu.csv")
te[["en","te"]].dropna().to_csv("data/te_no_aug.csv", index=False)
te[["en","te"]].dropna().to_csv("data/te_aug.csv", index=False)

# Kannada
kn = pd.read_csv("data/kannada.csv")
kn[["en","kn"]].dropna().to_csv("data/kn_no_aug.csv", index=False)
kn[["en","kn"]].dropna().to_csv("data/kn_aug.csv", index=False)

print("Training CSV files created")
