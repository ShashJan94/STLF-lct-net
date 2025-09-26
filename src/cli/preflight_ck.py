import argparse, torch
from src.training.preflight import run_preflight
from src.splits.rolling import rolling_quarter_folds
from src.windowing.loaders import make_loaders_for_fold, WindowConfig
from src.model.cnn_lct_att import CNN_LCT_Att
from src.data.merge_weather import load_and_merge

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--T", type=int, default=72)
    args = ap.parse_args()

    df = load_and_merge(args.csv)                  # your merged df
    folds, df_test = rolling_quarter_folds(df)
    tr_df, va_df, spec = folds[0]

    feature_cols = ["load","temperature","humidity",
                    "hour_sin","hour_cos","day_sin","day_cos","month_sin","month_cos"]
    win = WindowConfig(T=args.T, horizon=1, batch_size=64, scale_load_in_X=True)
    dl_tr, dl_va, dl_te, invert_y, used = make_loaders_for_fold(
        tr_df, va_df, df_test, win, feature_cols,
        mask_train=True, use_context_val=True, use_context_test=True
    )

    model = CNN_LCT_Att(in_feats=len(used), max_len=args.T)
    run_preflight(model, dl_tr, dl_va, invert_y_fn=invert_y, T_expected=args.T)

if __name__ == "__main__":
    main()
