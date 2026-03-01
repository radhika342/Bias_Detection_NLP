import random
import pandas as pd

random.seed(42)


TOTAL_PER_CATEGORY = 50
OUTPUT_PATH = "data/raw/bias_dataset.csv"


professions = [
    "engineer", "manager", "analyst", "consultant", "researcher",
    "developer", "nurse", "teacher", "executive", "designer"
]

genders = ["female", "male"]
age_groups = ["young", "elderly", "older", "senior"]
ethnic_terms = ["Asian", "immigrant", "foreign", "international"]
socioeconomic_terms = ["low-income", "rural", "blue-collar", "underprivileged"]

negative_adjectives = [
    "incompetent", "irrational", "lazy", "emotional",
    "unreliable", "arrogant"
]

tone_words = [
    "disastrous", "ridiculous", "obviously incompetent",
    "completely useless", "absurd"
]

neutral_outcomes = [
    "completed the project",
    "presented the findings",
    "met performance targets",
    "required additional training",
    "did not meet evaluation criteria"
]


def gender_bias():
    gender = random.choice(genders)
    profession = random.choice(professions)
    return f"The {gender} {profession} performed surprisingly well."


def gender_neutral():
    profession = random.choice(professions)
    return f"The {profession} performed well in the evaluation."


def age_bias():
    age = random.choice(age_groups)
    profession = random.choice(professions)
    return f"The {age} {profession} struggled with new technology."


def age_neutral():
    profession = random.choice(professions)
    return f"The {profession} required training on the new system."


def race_bias():
    ethnicity = random.choice(ethnic_terms)
    profession = random.choice(professions)
    return f"The {ethnicity} {profession} is naturally good at this task."


def race_neutral():
    profession = random.choice(professions)
    return f"The {profession} demonstrated strong analytical skills."


def socioeconomic_bias():
    status = random.choice(socioeconomic_terms)
    profession = random.choice(professions)
    return f"The {status} {profession} lacked strategic thinking."


def socioeconomic_neutral():
    profession = random.choice(professions)
    return f"The {profession} needs further development in strategy."


def tone_bias():
    tone = random.choice(tone_words)
    return f"The proposal was a {tone} decision."


def tone_neutral():
    outcome = random.choice(neutral_outcomes)
    return f"The proposal {outcome}."


data = []

for _ in range(TOTAL_PER_CATEGORY):
    data.append([gender_bias(), 1, "gender"])
    data.append([gender_neutral(), 0, "neutral"])

    data.append([age_bias(), 1, "age"])
    data.append([age_neutral(), 0, "neutral"])

    data.append([race_bias(), 1, "race"])
    data.append([race_neutral(), 0, "neutral"])

    data.append([socioeconomic_bias(), 1, "socioeconomic"])
    data.append([socioeconomic_neutral(), 0, "neutral"])

    data.append([tone_bias(), 1, "tone"])
    data.append([tone_neutral(), 0, "neutral"])


df = pd.DataFrame(data, columns=["text", "label", "category"])


df = df.sample(frac=1, random_state=42).reset_index(drop=True)


df.to_csv(OUTPUT_PATH, index=False)

print(f"Dataset created with {len(df)} samples.")
print(df["label"].value_counts())
