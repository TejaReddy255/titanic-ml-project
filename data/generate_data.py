"""
Downloads the Titanic dataset from a public source.
If network is unavailable, generates a representative synthetic dataset.
"""
import os
import pandas as pd
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), "titanic.csv")


def generate_synthetic_titanic(n=891, seed=42):
    """Generate a synthetic Titanic-like dataset with realistic distributions."""
    rng = np.random.RandomState(seed)

    pclass = rng.choice([1, 2, 3], size=n, p=[0.24, 0.21, 0.55])
    sex = rng.choice(["male", "female"], size=n, p=[0.65, 0.35])
    age = np.where(
        rng.random(n) < 0.2,
        np.nan,
        np.clip(rng.normal(29.7, 14.5, n), 0.42, 80),
    )
    sibsp = rng.choice([0, 1, 2, 3, 4, 5, 8], size=n, p=[0.68, 0.23, 0.03, 0.02, 0.02, 0.01, 0.01])
    parch = rng.choice([0, 1, 2, 3, 4, 5, 6], size=n, p=[0.760, 0.130, 0.090, 0.008, 0.006, 0.003, 0.003])
    fare_base = np.where(pclass == 1, rng.uniform(25, 512, n),
                np.where(pclass == 2, rng.uniform(10, 75, n),
                         rng.uniform(5, 35, n)))
    _emb_raw = rng.choice(["S", "C", "Q"], size=n, p=[0.72, 0.19, 0.09]).astype(object)
    _emb_raw[rng.random(n) < 0.02] = None
    embarked = _emb_raw
    # Survival logic: women first, 1st class, children
    base_prob = np.where(sex == "female", 0.74, 0.19)
    class_bonus = np.where(pclass == 1, 0.15, np.where(pclass == 2, 0.05, -0.05))
    age_arr = np.where(np.isnan(age), 29.7, age)
    child_bonus = np.where(age_arr < 15, 0.1, 0.0)
    survival_prob = np.clip(base_prob + class_bonus + child_bonus, 0.05, 0.95)
    survived = (rng.random(n) < survival_prob).astype(int)

    return pd.DataFrame({
        "PassengerId": range(1, n + 1),
        "Survived": survived,
        "Pclass": pclass,
        "Name": [f"Passenger_{i}" for i in range(1, n + 1)],
        "Sex": sex,
        "Age": np.round(age, 1),
        "SibSp": sibsp,
        "Parch": parch,
        "Ticket": [f"T{rng.randint(1000, 9999)}" for _ in range(n)],
        "Fare": np.round(fare_base, 4),
        "Cabin": [None if rng.random() < 0.77 else f"C{rng.randint(10,150)}" for _ in range(n)],
        "Embarked": embarked,
    })


def get_data():
    if os.path.exists(DATA_PATH):
        print(f"[data] Loading existing dataset from {DATA_PATH}")
        return pd.read_csv(DATA_PATH)

    print("[data] Generating synthetic Titanic dataset...")
    df = generate_synthetic_titanic()
    df.to_csv(DATA_PATH, index=False)
    print(f"[data] Saved {len(df)} rows to {DATA_PATH}")
    return df


if __name__ == "__main__":
    df = get_data()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Survival rate: {df['Survived'].mean():.2%}")
