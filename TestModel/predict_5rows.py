import os
import pandas as pd
from rag_gis.webapp.backend.model_utils import load_bundle, predict_single

# Paths
MODELS_DIR = r"C:\Users\SushantGS\Projects\Group\NasaSpaceApps\backend2\rag_gis\webapp\backend\models"
CSV_PATH = r"C:\Users\SushantGS\Projects\Group\NasaSpaceApps\backend2\TestModel\Exoplanets_data.csv"
OUT_PATH = r"C:\Users\SushantGS\Projects\Group\NasaSpaceApps\backend2\TestModel\Exoplanets_with_preds.csv"


def main():
    bundle = load_bundle(models_dir=MODELS_DIR)
    if bundle.model is None:
        raise SystemExit(f"No model found in {MODELS_DIR}")

    df = pd.read_csv(CSV_PATH)
    df5 = df.head(5).copy()

    # Keep original id and class for reporting
    orig = df5[['id']].copy() if 'id' in df5.columns else pd.DataFrame({'id': range(len(df5))})
    true_labels = df5['class'].tolist() if 'class' in df5.columns else [None] * len(df5)

    # Drop id and class for features
    drop_cols = [c for c in ('id', 'class') if c in df5.columns]
    X_df = df5.drop(columns=drop_cols)

    results = []
    for i, row in X_df.iterrows():
        feats = row.tolist()
        try:
            res = predict_single(bundle, feats)
        except Exception as e:
            res = {'raw': None, 'prediction': None, 'success': False, 'error': str(e)}
        results.append(res)

    # Attach predictions back to df5
    preds = [r.get('prediction') for r in results]
    df5['predicted'] = preds

    # Print results
    for i in range(len(df5)):
        print(f"Row {i} id={df5.iloc[i].get('id', '')} true={df5.iloc[i].get('class', '')}")
        print("  features:", X_df.iloc[i].tolist())
        print("  raw:", results[i].get('raw'))
        print("  predicted:", results[i].get('prediction'))
        print("  success:", results[i].get('success'))
        print('-' * 40)

    # Write CSV with predictions
    df5.to_csv(OUT_PATH, index=False)
    print(f"Wrote results to {OUT_PATH}")


if __name__ == '__main__':
    main()
