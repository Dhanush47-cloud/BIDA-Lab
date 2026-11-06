import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
filename = "Assignment-1_Data.csv"

# low_memory=False avoids the mixed-dtype warning on the BillNo column
df = pd.read_csv(filename, encoding="unicode_escape", low_memory=False)

# The CSV is saved with a UTF-8 BOM, which pandas keeps attached to the
# first column name (turning it into "ï»¿BillNo"). Strip the BOM/whitespace
# from all column names so we can refer to columns normally (e.g. "BillNo").
df.columns = df.columns.str.replace("ï»¿", "", regex=False).str.strip()

print("Successfully read the CSV file.")
print(df.columns.tolist())
df.dropna(axis=0, subset=["CustomerID"], inplace=True)

df["BillNo"] = df["BillNo"].astype(str)
df = df[~df["BillNo"].str.contains("C")]

df["Itemname"] = df["Itemname"].astype(str).str.strip()

print(df.head())
transactions = df.groupby("BillNo")["Itemname"].apply(list).tolist()

print("Total transactions:", len(transactions))
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)

df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

print(df_encoded.head())
# NOTE: min_support was lowered from 0.01 to 0.002. At 0.01 every
# frequent itemset found was a single item, so no association rules
# (which need itemsets of size >= 2) could ever be generated below.
frequent_itemsets = apriori(
    df_encoded,
    min_support=0.002,
    use_colnames=True
)

print(frequent_itemsets)

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.5
)

print(rules)
