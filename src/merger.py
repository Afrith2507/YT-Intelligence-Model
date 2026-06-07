import pandas as pd

df1 = pd.read_csv(r"C:\Users\Afrith\Desktop\Afrith\UOWD\CSCI370\Project\data\raw\7kdataset.csv")
df2 = pd.read_csv(r"C:\Users\Afrith\Desktop\Afrith\UOWD\CSCI370\Project\data\raw\8kdataset.csv")

# Stack rows from both files
merged_df = pd.concat([df1, df2], ignore_index=True)

# Optional: remove duplicates by comment id (or use 'id' if that's your column)
key_col = "comment_id" if "comment_id" in merged_df.columns else "id"
merged_df = merged_df.drop_duplicates(subset=[key_col])

merged_df.to_csv(
    r"C:\Users\Afrith\Desktop\Afrith\UOWD\CSCI370\Project\data\raw\final_dataset.csv",
    index=False
)

print("Rows after merge:", len(merged_df))