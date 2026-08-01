# Cricket T20 Analytics & Win Prediction

Can we predict who wins a T20 match before it's played — and explain why?*

That's the question this project set out to answer, using ball-by-ball
and match-level data from the 2022 T20 World Cup. It walks through the
full journey of a data analytics project: scraping and cleaning real
data, structuring it in a proper database, uncovering patterns through
SQL, and finally building a model that predicts outcomes and explains its
own reasoning.

* Live App:** [Add your Streamlit Community Cloud link here]
* Power BI Dashboard:** [Add link/screenshot here, if included]

---

## Why This Project

Sports organizations, broadcasters, and fantasy platforms all care about
the same underlying question: *what factors actually predict a team's
chances of winning?* Is it recent form? Historical rivalry? Home
advantage? This project treats T20 cricket as a business analytics
problem — the same kind of thinking used for customer churn or sales
forecasting, just applied to a sport I enjoy following. The goal wasn't
just "build a model" — it was to go through the full pipeline a data
analyst or data scientist would in a real job: understand the data,
clean it, structure it for querying, find patterns, and communicate
findings in a way non-technical stakeholders could actually use.

---

## Origin Story (the honest version)

I started this project by following codebasics' *"End to End Cricket
Data Analytics Project Using Web Scraping, Python, Pandas and Power BI"*
tutorial — a great way to learn the basics of scraping and dashboarding.
But a tutorial clone doesn't show what *I* can do, so I used it purely as
a launchpad and rebuilt it into something more substantial:

- Restructured the entire pipeline into clean, modular components
  (scraper, cleaner, database loader, feature engineering, model
  training, and a live app)
- Added a proper **SQL layer** with a normalized database — the original
  tutorial worked directly off pandas and CSVs
- Added a **machine learning layer** to predict match outcomes, complete
  with explainability — this wasn't part of the original tutorial at all
- Added a **live, interactive app** so anyone (including you, reading
  this) can try the prediction tool themselves, instead of just looking
  at static charts

The original tutorial's Power BI dashboard is still included in this
repo as-is — I haven't reworked it with the new data yet, and I'd rather
tell you that directly than let you assume it's more original than it is.

---

## How It's Built

```
Scrape (Python) → Clean (Pandas) → Store (SQLite) → Analyze (SQL) →
Model (XGBoost + SHAP) → Present (Streamlit + Power BI)
```

Think of it as a relay race — each stage hands off clean, structured work
to the next: raw web data gets cleaned, cleaned data gets stored properly,
stored data gets queried for patterns, and those patterns feed a model
that makes a prediction and explains itself in plain language.

---

## What This Project Actually Answers (Business Questions)

Framed the way a stakeholder — a team analyst, a broadcaster, a fantasy
platform — might actually ask it:

- **"Which players consistently perform at specific venues?"** — answered
  via SQL queries joining batting/bowling stats with venue data
- **"Does winning the toss actually matter?"** — explored through team
  performance patterns across matches
- **"How much does recent form predict future results?"** — directly
  tested as a feature in the prediction model, and it turned out to be
  one of the strongest signals
- **"Can we quantify a team's chances before a ball is bowled?"** — the
  core deliverable: a live win-probability tool, with an explanation of
  *why* the model landed on that number for any given matchup

---

## What I Added Beyond the Tutorial

1. **Modular scraper** — split cleanly into batting summary, bowling
   summary, player info, and match results, each with error handling and
   logging (functional in principle — see limitations below)
2. **A real database, not just CSVs** — a normalized SQL schema
   (`teams`, `players`, `matches`, `batting`, `bowling`) with analytical
   queries answering the business questions above
3. **A win-probability model** — engineered features capturing team
   form, head-to-head history, and venue advantage; compared a simple
   model against a more powerful one using proper time-aware validation
4. **Explainability, not just a number** — SHAP values break down
   exactly which factors pushed a prediction up or down for any given
   matchup, so the model isn't a black box
5. **A live app** — a dashboard, an interactive prediction tool, and a
   plain-language summary of insights, all in one place, with a clean,
   icon-free interface and custom styling

---

## What I Found

- The more sophisticated model (XGBoost) beat a simple baseline
  (Logistic Regression) — not just on average accuracy (~60% vs ~40%),
  but it was also far more *consistent* across different slices of the
  data, which matters more than a single good score
- **Recent team form and head-to-head history were the two strongest
  predictors** — more influential than venue in most matchups
- With only one tournament's worth of data (45 matches), these
  predictions should be read as directional signals, not certainties —
  and I'd rather say that clearly than oversell the model

---

## Behind the Scenes: What Actually Went Wrong (and What I Learned)

Real projects don't go in a straight line, and I think the debugging is
more interesting than the polished result — here's what I actually ran
into:

**🔍 The model was "too good" — and that's a red flag, not a win.**
My first version scored ~100% accuracy, which sounds great until you
remember that predicting sports outcomes is genuinely hard. I dug in and
found the bug: my "recent form" feature was accidentally including the
outcome of the very match it was trying to predict — like handing a
student the answer key by accident. Once I fixed the feature to only
look at *past* matches, accuracy dropped to a much more believable ~60%.
That taught me to be suspicious of results that look too perfect, not
just happy about them.

**📊 45 matches isn't a lot of data to test on.** A standard 80/20 split
left only 5 matches to evaluate the model on — one lucky or unlucky
prediction could swing the score by 20%. I switched to cross-validation
that respects match chronology, giving a far more trustworthy accuracy
estimate instead of a single noisy number.

**🚧 The live scraper got blocked.** ESPN Cricinfo's current bot
protection rejects standard scraping requests outright (a 403 error,
even with realistic browser headers). Rather than burn hours fighting
infrastructure-level blocking, I made the call to use the existing
dataset and document the limitation honestly — a real judgment call
about where to spend limited time, which I think is itself a useful
skill to show.

**🔧 My database and my code fell out of sync more than once.** While
iterating, I'd update my database structure but forget my query code
still expected the old structure (or vice versa) — leading to some
confusing "table not found" errors. Fixed by treating the database
schema as the single source of truth and working outward from it.

**🧩 A library incompatibility that no version change could fix.**
SHAP's explainer kept failing to load my XGBoost model, throwing a
parsing error on an internal value called `base_score`. I assumed it was
a version mismatch and tried retraining with an explicit value, then
downgrading XGBoost entirely — neither fixed it, which told me the issue
was deeper than a simple version pin. Rather than keep chasing it, I
switched to XGBoost's own built-in feature-contribution output (the same
underlying math SHAP uses), and built the explanation chart manually.
Sometimes the right fix isn't forcing the original tool to work — it's
recognizing when to route around it.

---

## Try It Yourself

```bash
# 1. Clean the raw data
python clean.py

# 2. Load it into a proper database
python sql/load_data.py

# 3. Train and evaluate the model
python models/train.py
python models/evaluate.py

# 4. Launch the app
python -m streamlit run streamlit_app.py
```

The app has been tested end-to-end in a clean virtual environment against
the pinned versions in `requirements.txt`, and is deployed live on
Streamlit Community Cloud (see link at the top of this README).

---

## Where This Could Go Next

- Add more tournaments/seasons — the biggest current limitation is a
  small, single-tournament dataset
- Add toss data as a feature — not available in the current schema, but
  a genuinely important factor in T20 outcomes
- Get the live scraper working again, likely using a tool built to
  handle modern anti-bot protections, so the dataset can grow over time
- Extend the Power BI dashboard with the new data and model outputs, to
  match the depth of the Streamlit app
- Investigate why two features (head-to-head record, neutral venue flag)
  sometimes show zero contribution in the prediction explanation for
  certain matchups — a known display-layer issue, not a flaw in the
  underlying model, that I've deprioritized in favor of shipping a
  working, deployed app on time

---

## Tech Stack

Python · Pandas · BeautifulSoup · SQLite · scikit-learn · XGBoost · SHAP ·
Streamlit · Power BI

---

*If you're reading this as a recruiter or hiring manager — thanks for
taking the time. I'm happy to walk through any part of this in more
detail, especially the debugging story above, which taught me more than
the final result did.*
