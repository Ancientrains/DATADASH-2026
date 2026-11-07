# SSU Data Dash presentation script

**Target runtime:** 4:30 total (about 1:30 each). This leaves a small buffer under the five-minute video limit.

**Slide order:** title and problem → two-track setup → time-safe pipeline → validation/results → error-analysis figures → reproducibility and takeaway.

## Justin Li — problem and framing (0:00–1:30)

Hi everyone! We’re Justin Li, Kevin Hu, and Michael Shen, and our project asks a pretty practical question: **what will this home actually sell for?**

The competition has two very different moments. Off-market means the home is not listed yet: no asking price, just the home, its location, and earlier sales. It is the harder, double-weighted track. List-aware means the home is listed, so the question becomes, “Will it sell above, below, or around list?”

We split them on purpose. A buyer needs an independent value; a seller needs help interpreting an asking price. One model would blur those decisions and could accidentally let list-price information improve off-market predictions. That distinction became our north star for every feature, model, and check that came next.

Our other rule was: act like we are standing in that moment in time. If we would not have known something yet, the model does not get to know it either. The data gives us 150,000 historical synthetic listings, then asks us to predict later homes from June through September 2026. That timeline shaped the whole project: time-safe modelling first, score second.

## Kevin Hu — pipeline and leakage safety (1:30–3:00)

So what is inside the model? For off-market, we combine the basics—home type, size, location, and parking—with condition signals from the listing text, comparable earlier sales, and a seasonal market adjustment. We chose LightGBM because house prices have interactions: location can matter differently for a condo than a detached home, and size is not perfectly linear. We blend three deterministic seeds—not a huge ensemble—to smooth one model's quirks while keeping the pipeline simple and reproducible.

For list-aware, we made one small but important change: instead of predicting the whole sale price from scratch, we predict log sold-to-list ratio: how far above or below asking will this home close? The log treats similarly sized premiums and discounts more symmetrically. List price is allowed here—but never in off-market features.

The part we cared about most was leakage. Every comparable is as-of: it must have sold before the prediction cutoff. We also fixed a same-month index issue, so a later sale cannot quietly help predict an earlier one.

That is the difference between a flashy score and a model we would trust.

[Show a simple timeline here: earlier sales → cutoff → homes predicted that month. Avoid code screenshots.]

This gives us a more honest estimate of how the model should behave on the future hidden period.

## Michael Shen — validation, results, and takeaway (3:00–4:30)

To test it honestly, we used monthly walk-forward validation from December 2025 through May 2026. Each month is held out; the model only learns from what happened before it. That gave us 10,069 distinct held-out homes for each track. We chose this over a random split because market conditions move over time. A random split mixes future sales into training and makes deployment look easier.

The result we are most excited about is off-market: **7.19% median absolute percentage error**, versus **20.8%** for the starter nearest-neighbour baseline. On list-aware, we reached **3.82% MdAPE**, compared with **7.3%** for simply predicting the list price. Our mean errors were 9.52% off-market and 4.82% list-aware.

These plots show how error changes by home type, predicted price band, and month—where the model is steady and where people should be more cautious. A low average is not enough if one kind of home is consistently priced badly.

These are validation results, not hidden-test claims. They are reproducible: our repository can rerun validation, generate predictions, and check the submission files.

Strong performance did not come from one magical model. It came from matching the model to the decision, building fair comparables, and being strict about time. Thank you!

[Show `error_by_type.png`, `error_by_predicted_price.png`, and `signed_error_by_month.png` here.]

## Delivery notes

- Do not claim hidden-test accuracy, a predicted leaderboard rank, uncertainty intervals, or coverage; we have not measured those.
- Do not call the data real Toronto records. The competition data is synthetic.
- Keep the numerical results exactly as above; every video number should come from the project validation.
- If time runs short, shorten the baseline comparison, not the validation and leakage explanation.
