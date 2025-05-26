from classificationEval import Evaluator
from load_data import load_data
from sklearn.model_selection import train_test_split
import pandas as pd
import os

fp_keys = ["ecfp4", "ap", "tt", "rdkit", "maccs"]
class_colname = "Class"

# Get pre-processed data
processed_dta_path = r"C:\Users\huanqing\Desktop\ARIA_Documents\Processed_data"
df_lst, new_filenames = load_data(processed_dta_path)

df_lst = df_lst[len(df_lst) - 2: len(df_lst)] # for testing
new_filenames = new_filenames[len(new_filenames) - 2: len(new_filenames)] # for testing
output_path = r"C:\Users\huanqing\Desktop\ARIA_Documents\eval_results"


for i in range(len(df_lst)):
    df = df_lst[i]
    output_dct = {}
    print(f"\nProcessing {new_filenames[i]} ... ")

    for j in range(len(fp_keys)):
        fp_colname = fp_keys[j]
        print(fp_colname)

        # Stratified Splitting into train, test, val in the ratio of 80:10:10
        train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df[class_colname], random_state=42)

        # Then split temp into 10% val and 10% test
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df[class_colname], random_state=42)
    
        # encapsulate all below into object
        # Now pass them into Evaluator
        evaluator = Evaluator(train_df, test_df, val_df,
                            epochs=10,
                            fp_colname=fp_colname,
                            class_colname=class_colname)
        
        results = evaluator.test()

        # collect results
        f1_val = results["f1_val"]
        auc_val = results["auc_val"]
        f1_test = results["f1_test"]
        auc_test = results["auc_test"]

        print(f"Validation F1-score: {f1_val:.4f}, AUC: {auc_val:.4f}")
        print(f"Test F1-score: {f1_test:.4f}, AUC: {auc_test:.4f}")

        output_dct[fp_colname] = {
            "f1_val" : f1_val,
            "auc_val" : auc_val,
            "f1_test" : f1_test,
            "auc_test" : auc_test
        }
        print("output:")
        print(output_dct)
        print()

    new_path = os.path.join(output_path, f"eval_{new_filenames[i]}")
    output_df = pd.DataFrame.from_dict(output_dct, orient='index')  # Make rows = fp keys
    output_df.to_csv(new_path)
    print(f"Results file saved successfully.\n")

