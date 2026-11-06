"""Make simple presentation plots from `outputs/validation_predictions.csv`.

Run `python run.py validate` first, then `python analyze.py`.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent
pred = pd.read_csv(ROOT / "outputs" / "validation_predictions.csv")
pred["ape"] = 100 * (pred.predicted_price - pred.sold_price).abs() / pred.sold_price
pred["signed"] = 100 * (pred.predicted_price - pred.sold_price) / pred.sold_price
out = ROOT / "outputs" / "figures"
out.mkdir(parents=True, exist_ok=True)

# Typical error by property type.
by_type = pred.groupby(["track", "house_type_name"]).ape.median().unstack(0).sort_values("off_market")
ax = by_type.plot.bar(figsize=(10, 4), color=["#2878B5", "#E56B3F"])
ax.set_ylabel("Median absolute percentage error")
ax.set_xlabel("")
ax.set_title("Validation error by property type")
plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig(out / "error_by_type.png", dpi=160); plt.close()

# Use predicted, not actual, price bands to avoid a regression-to-the-middle story.
pred["price_band"] = pd.qcut(pred.predicted_price, 10, duplicates="drop")
by_band = pred.groupby(["track", "price_band"], observed=True).ape.median().unstack(0)
ax = by_band.plot(marker="o", figsize=(9, 4), color=["#2878B5", "#E56B3F"])
ax.set_ylabel("Median absolute percentage error")
ax.set_xlabel("Predicted-price decile")
ax.set_title("Validation error by predicted-price band")
plt.xticks(range(len(by_band)), range(1, len(by_band) + 1)); plt.tight_layout(); plt.savefig(out / "error_by_predicted_price.png", dpi=160); plt.close()

# Signed error over the walk-forward months highlights market bias.
by_month = pred.groupby(["track", "month"]).signed.median().unstack(0)
ax = by_month.plot(marker="o", figsize=(9, 4), color=["#2878B5", "#E56B3F"])
ax.axhline(0, color="black", linewidth=0.8); ax.set_ylabel("Median signed error (%)")
ax.set_xlabel(""); ax.set_title("Validation bias by month")
plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig(out / "signed_error_by_month.png", dpi=160); plt.close()

print(f"Wrote plots to {out.relative_to(ROOT)}")
