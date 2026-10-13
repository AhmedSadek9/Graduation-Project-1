from datasets import load_dataset
import pandas as pd

# 1. Download official CLINC150 dataset from Hugging Face
print("1. Loading CLINC150 dataset from Hugging Face...")
dataset = load_dataset("clinc/clinc_oos", "small", split="train")
df_clinc = pd.DataFrame(dataset)

# Extract original intent labels
intent_names = dataset.features["intent"].names
df_clinc["intent_name"] = df_clinc["intent"].apply(lambda x: intent_names[x])

# Map CLINC150 categories to project target intents
clinc_mapping = {
    "reminder": "GET_MEMORY",
    "reminder_update": "GET_MEMORY",
    "help": "HELP"
}

df_filtered = df_clinc[df_clinc["intent_name"].isin(clinc_mapping.keys())].copy()
df_filtered["intent"] = df_filtered["intent_name"].map(clinc_mapping)
df_from_clinc = df_filtered[["text", "intent"]]

# 2. Add Alzheimer-specific queries for missing categories
alzheimer_augmented_data = [
    # IDENTIFY_PERSON
    {"text": "who is this standing in front of me", "intent": "IDENTIFY_PERSON"},
    {"text": "who is this person", "intent": "IDENTIFY_PERSON"},
    {"text": "who is that man", "intent": "IDENTIFY_PERSON"},
    {"text": "who is she", "intent": "IDENTIFY_PERSON"},
    {"text": "do i know him", "intent": "IDENTIFY_PERSON"},
    {"text": "do i know her", "intent": "IDENTIFY_PERSON"},
    {"text": "who are you", "intent": "IDENTIFY_PERSON"},
    {"text": "can you tell me who is in front of me", "intent": "IDENTIFY_PERSON"},
    {"text": "who am i looking at right now", "intent": "IDENTIFY_PERSON"},
    {"text": "do you recognize this face", "intent": "IDENTIFY_PERSON"},
    {"text": "who just walked into the room", "intent": "IDENTIFY_PERSON"},
    {"text": "tell me who is sitting next to me", "intent": "IDENTIFY_PERSON"},
    {"text": "who is this visitor", "intent": "IDENTIFY_PERSON"},
    {"text": "identify this person for me", "intent": "IDENTIFY_PERSON"},
    {"text": "is this my son or daughter", "intent": "IDENTIFY_PERSON"},
    {"text": "who is talking to me", "intent": "IDENTIFY_PERSON"},
    {"text": "who is by the door", "intent": "IDENTIFY_PERSON"},
    {"text": "tell me the name of this person", "intent": "IDENTIFY_PERSON"},
    {"text": "what is this person's name", "intent": "IDENTIFY_PERSON"},
    {"text": "do you know who this is", "intent": "IDENTIFY_PERSON"},
    {"text": "who is standing near me", "intent": "IDENTIFY_PERSON"},
    {"text": "can you recognize him", "intent": "IDENTIFY_PERSON"},
    {"text": "who came to see me", "intent": "IDENTIFY_PERSON"},

    # CHECK_MEDICATION
    {"text": "did i take my medicine today", "intent": "CHECK_MEDICATION"},
    {"text": "did i take my pills this morning", "intent": "CHECK_MEDICATION"},
    {"text": "is it time for my medication", "intent": "CHECK_MEDICATION"},
    {"text": "have i taken my pills yet", "intent": "CHECK_MEDICATION"},
    {"text": "remind me to take my medicine", "intent": "CHECK_MEDICATION"},
    {"text": "when should i take my blood pressure pills", "intent": "CHECK_MEDICATION"},
    {"text": "did i forget my pills", "intent": "CHECK_MEDICATION"},
    {"text": "check my medication schedule", "intent": "CHECK_MEDICATION"},
    {"text": "is my pill due right now", "intent": "CHECK_MEDICATION"},
    {"text": "did i take my evening medication", "intent": "CHECK_MEDICATION"},
    {"text": "what pills do i need to take now", "intent": "CHECK_MEDICATION"},
    {"text": "have i had my insulin shot", "intent": "CHECK_MEDICATION"},
    {"text": "medication check please", "intent": "CHECK_MEDICATION"},
    {"text": "time for my daily medicine", "intent": "CHECK_MEDICATION"},
    {"text": "did i swallow my tablets", "intent": "CHECK_MEDICATION"},
    {"text": "should i take my pills", "intent": "CHECK_MEDICATION"},
    {"text": "when is my next dose", "intent": "CHECK_MEDICATION"},
    {"text": "did i miss my medicine", "intent": "CHECK_MEDICATION"},
    {"text": "is it time for pills", "intent": "CHECK_MEDICATION"},
    {"text": "did i drink my syrup", "intent": "CHECK_MEDICATION"},

    # Additional HELP queries
    {"text": "help me please", "intent": "HELP"},
    {"text": "i need urgent help", "intent": "HELP"},
    {"text": "please assist me right now", "intent": "HELP"},
    {"text": "i fell down help me", "intent": "HELP"},
    {"text": "emergency i need someone", "intent": "HELP"},
    {"text": "can you help me", "intent": "HELP"},
    {"text": "i am lost please help", "intent": "HELP"}
]

df_augmented = pd.DataFrame(alzheimer_augmented_data)
df_final = pd.concat([df_from_clinc, df_augmented], ignore_index=True)

# 3. Balance dataset across all 4 intents
min_samples_per_class = min(df_final["intent"].value_counts())
df_balanced = (
    df_final.groupby("intent", group_keys=False)
    .apply(lambda x: x.sample(n=min(len(x), 35), random_state=42))
    .reset_index(drop=True)
)

output_filename = "wanas_augmented_dataset.csv"
df_balanced.to_csv(output_filename, index=False)

print("\nDataset successfully prepared and balanced:")
print(df_balanced["intent"].value_counts())