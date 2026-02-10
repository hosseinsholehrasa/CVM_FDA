
import pandas as pd
import numpy as np
from preprocessing.terminology_standardization import ATCVetConverter, HLTMapper
from preprocessing.data_cleaning import (
    clean_ingredient_name,
    clean_age,
    clean_weight,
    clean_dose,
    normalize_basic_columns,
    process_breed_component,
)


def load_raw_data(base_dir="data"):
    df_all = pd.read_csv(f"{base_dir}/all.csv", low_memory=False)
    df_drugs = pd.read_csv(f"{base_dir}/drugs.csv", low_memory=False)
    df_outcomes = pd.read_csv(f"{base_dir}/outcomes.csv", low_memory=False)
    df_reactions = pd.read_csv(f"{base_dir}/reactions.csv", low_memory=False)

    df_outcomes["medical_status"] = df_outcomes["medical_status"].fillna(
        "Outcome Unknown"
    )
    df_drugs = df_drugs.dropna(subset=["name"])

    return df_all, df_drugs, df_outcomes, df_reactions


def filter_animals(df_all, df_drugs, df_outcomes, df_reactions):

    allowed_species = [
        "Dog","Cat","Cattle","Horse","Pig","Sheep",
        "Chicken","Goat","Turkey","Rabbit"
    ]

    df_all_animals = df_all[df_all["animal.species"].isin(allowed_species)]

    df_outcomes_animals = df_outcomes[
        df_outcomes["unique_aer_id_number"]
        .isin(df_all_animals["unique_aer_id_number"])
    ]
    df_reactions_animals = df_reactions[
        df_reactions["unique_aer_id_number"]
        .isin(df_all_animals["unique_aer_id_number"])
    ]
    df_drugs_animals = df_drugs[
        df_drugs["unique_aer_id_number"]
        .isin(df_all_animals["unique_aer_id_number"])
    ]

    dup_ids = (
        df_outcomes_animals["unique_aer_id_number"]
        .value_counts()
        .loc[lambda s: s > 1]
        .index
    )

    df_outcomes_animals = df_outcomes_animals[
        ~df_outcomes_animals["unique_aer_id_number"].isin(dup_ids)
    ]

    df_reactions_animals = df_reactions_animals[
        df_reactions_animals["unique_aer_id_number"]
        .isin(df_outcomes_animals["unique_aer_id_number"])
    ]
    df_all_animals = df_all_animals[
        df_all_animals["unique_aer_id_number"]
        .isin(df_outcomes_animals["unique_aer_id_number"])
    ]
    df_drugs_animals = df_drugs_animals[
        df_drugs_animals["unique_aer_id_number"]
        .isin(df_outcomes_animals["unique_aer_id_number"])
    ]

    return df_all_animals, df_drugs_animals, df_outcomes_animals, df_reactions_animals


# ============================================================
# MERGE OUTCOMES + REACTIONS + DRUGS + ALL
# ============================================================

def build_analysis_table(df_all_animals, df_drugs_animals,
                         df_outcomes_animals, df_reactions_animals):

    df_outcomes_animals = df_outcomes_animals.rename(
        columns={"number_of_animals_affected":
                 "number_of_animals_affected_outcome"}
    )
    df_reactions_animals = df_reactions_animals.rename(
        columns={"number_of_animals_affected":
                 "number_of_animals_affected_reactions"}
    )

    df_outcomes_animals = df_outcomes_animals.drop(
        columns=["year","date"], errors="ignore"
    )
    df_reactions_animals = df_reactions_animals.drop(
        columns=["year","date"], errors="ignore"
    )
    df_drugs_animals = df_drugs_animals.drop(
        columns=["year","date"], errors="ignore"
    )

    tmp = df_outcomes_animals.merge(
        df_reactions_animals,
        on="unique_aer_id_number",
        how="inner"
    )
    tmp = tmp.merge(
        df_drugs_animals,
        on="unique_aer_id_number",
        how="inner"
    )
    df_analysis = tmp.merge(
        df_all_animals,
        on="unique_aer_id_number",
        how="inner"
    )

    df_analysis["veddra_term_name"] = (
        df_analysis["veddra_term_name"].str.lower()
    )

    df_analysis["number_of_animals_affected_outcome"] = (
        df_analysis["number_of_animals_affected_outcome"].fillna(1)
    )
    df_analysis["number_of_animals_affected_reactions"] = (
        df_analysis["number_of_animals_affected_reactions"].fillna(1)
    )

    # df_analysis["medical_status"] = df_analysis["medical_status"].replace({
    #     "Recovered with Sequela": "Recovered/Normal"
    # })

    # df_analysis = df_analysis[
    #     df_analysis["medical_status"].isin(["Died", "Recovered/Normal"])
    # ]

    return df_analysis


def run_fda_cleaning(raw_dir="data",
                     out_npz="data/np_analysis_cleaned.npz"):

    print("Loading raw data...")

    df_all, df_drugs, df_outcomes, df_reactions = load_raw_data(raw_dir)

    df_all_animals, df_drugs_animals, df_outcomes_animals, df_reactions_animals = \
        filter_animals(df_all, df_drugs, df_outcomes, df_reactions)

    print("Building analysis table...")
    df_analysis = build_analysis_table(
        df_all_animals, df_drugs_animals,
        df_outcomes_animals, df_reactions_animals
    )
    print()

    # ----------------------------
    # APPLY VEDDRA → HLT CLEANING
    # ----------------------------
    print("Building Veddra → HLT map...")
    hlt_mapper = HLTMapper(
        veddra2018_xls=f"{raw_dir}/vm_reaction2018.xls",
        veddra2018_add=f"{raw_dir}/vm_reaction_additional.xls",
        veddra2025_xlsx=f"{raw_dir}/vm_reaction2025.xlsx",
    )
    hlt_mapper.load_reference_files()
    veddra2hlt, veddra_not_found = hlt_mapper.map_veddra_to_hlt(df_analysis)
    print(f"Not found Veddra terms: {len(veddra_not_found)}")

    print("Filtering and mapping Veddra terms...")
    before_veddra_filter = df_analysis["unique_aer_id_number"].nunique()
    df_analysis = df_analysis[
        ~df_analysis["veddra_term_name"].isin(veddra_not_found)
    ]

    df_analysis["veddra_term_name"] = (
        df_analysis["veddra_term_name"]
        .map(veddra2hlt)
        .fillna(df_analysis["veddra_term_name"])
    )

    # keep only frequent terms (>20)
    term_counts = df_analysis["veddra_term_name"].value_counts()
    df_analysis = df_analysis[
        df_analysis["veddra_term_name"]
        .map(term_counts) > 20
    ]
    # Remove reactions which is Death to avoid biasing the analysis
    df_analysis = df_analysis[~df_analysis['veddra_term_name'].str.contains('death')]

    # Drop lack of efficacy to avoid biasing the analysis
    df_analysis = df_analysis[
        df_analysis["veddra_term_name"] != "lack of efficacy"
    ]
    after_veddra_filter = df_analysis["unique_aer_id_number"].nunique()
    print()


    # --------------------------------------------------
    # EXISTING NUMERIC + BASIC CLEANING STEPS
    # --------------------------------------------------
    cols = [
        'medical_status','number_of_animals_affected_outcome',
        'unique_aer_id_number','veddra_term_name','veddra_term_code',
        'route','dosage_form','atc_vet_code','name',
        'dose.numerator','dose.numerator_unit','dose.denominator',
        'dose.denominator_unit','animal.species','animal.gender',
        'animal.age.min','animal.age.unit','animal.weight.min',
        'animal.weight.unit','animal.breed.breed_component',
        'animal.age.max','animal.weight.max','year','drug_id'
    ]

    print("Starting numerical and categorical cleaning...")
    df_clean = df_analysis[cols].copy()

    print("Normalizing ingredient names...")
    df_clean = clean_ingredient_name(df_clean)
    print("Normalizing age...")
    df_clean = clean_age(df_clean)
    print("Normalizing weight...")
    df_clean = clean_weight(df_clean)
    print("Normalizing dose...")
    df_clean = clean_dose(df_clean)
    print()
    df_clean = normalize_basic_columns(df_clean)

    print("Normalizing breeds...")
    df_clean["animal.breed.breed_component"] = (
        df_clean["animal.breed.breed_component"]
        .apply(process_breed_component)
    )
    print()

    df_clean = df_clean.drop(columns=[
        'animal.age.min','animal.age.max','animal.age.unit',
        'animal.weight.min','animal.weight.max','animal.weight.unit',
        'dose.numerator','dose.numerator_unit',
        'dose.denominator','dose.denominator_unit'
    ], errors="ignore")


    final_cols = [
        'medical_status','number_of_animals_affected_outcome',
        'unique_aer_id_number','veddra_term_name','name',
        'route','dosage_form','atc_vet_code','year',
        'animal.species','animal.gender',
        'animal.breed.breed_component',
        'animal_age','animal_weight','dose','drug_id'
    ]

    df_clean = df_clean[final_cols]

    np.savez_compressed(out_npz, data=df_clean)
    print(f"Saved cleaned data to: {out_npz}")

    return df_clean


if __name__ == "__main__":
    run_fda_cleaning()
