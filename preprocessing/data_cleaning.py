import os
import json
import ast
import pandas as pd
import numpy as np
from tqdm import tqdm


def clean_ingredient_name(df):
    # cleaning ingredients name

    df['name'] = df['name'].str.replace(r',\s+|;\s+|\+\s+|/|;', ' / ', regex=True)
    df['name'] = df['name'].str.replace(
        r'Tablet[s]?|Chew(able)?|Oral|Solution|Injection|Injectable|Suspension|Powder|Paste|Granules|Capsule|Liquid|Gel' \
        r'|Cats?|Dogs?|Rabbits?|Cattles?|Horses?|Pet$|For$|And$|\(Unknown\)|Unknown|Unkown|\(?Not-Specified\)?|\(?Unspecified\)?|Spot-On|Spot On|',
        '',
        regex=True
    )
    # Remove numeric values with units (Mg, Mcg, g, Kg, ml, L)
    df['name'] = df['name'].str.replace(
        r'\b\d+(\.\d+)?\s?(Mg|Mcg|g|Kg|ml|Ml|L)(/[A-Za-z]+)?\b',
        '',
        regex=True
    ).str.strip()

    # remove durations (Month, Week, Day, Year, etc.)
    df['name'] = df['name'].str.replace(
        r'\b\d+(\.\d+)?-?(Month|Week|Day|Year|Hour|Minute|Second)s?\b',
        '',
        regex=True
    )

    # remove trailing '- number'
    df['name'] = df['name'].str.replace(r'-\s?[0-9A-Za-z.]+$', '', regex=True)

    # Remove percentages like 25%
    df['name'] = df['name'].str.replace(
        r'\d+(\.\d+)?\s*\(?%\)?',
        '',
        regex=True
    )
    df['name'] = df['name'].str.replace('()', "")
    # Enforce " / " spacing
    df["name"] = df["name"].str.replace(r'\s*/\s*', ' / ', regex=True)
    # Remove trailing or leading slash with optional spaces
    df["name"] = df["name"].str.replace(r'\s*/\s*$', '', regex=True)  # end
    df["name"] = df["name"].str.replace(r'^\s*/\s*', '', regex=True)  # start
    # Collapse multiple spaces
    df["name"] = df["name"].str.replace(r'\s+', ' ', regex=True).str.strip()

    # Remove device: *
    df["name"] = df["name"].str.replace(r'(?i)device:\s*.*$', '', regex=True)

    # Remove single-letter names
    df["name"] = df["name"].str.replace(r'^\s*[A-Za-z]\s*$', '', regex=True)

    # Special Case
    df['name'] = df['name'].str.replace(r'(?i)As Pamoate Salt|\.|Vaccine(s)|\(S\)|', '', regex=True)

    df['name'] = df['name'].str.strip()

    # df['name'] = df['name'][~df['name'].isin(["", " ", None])]
    # df['name'] = df['name'].dropna()

    drugs_name = df['name'].unique().tolist()
    print("Number of unique drug names:", len(drugs_name))

    return df


def clean_age(df):
    df = df.copy()

    df["animal_age"] = df[["animal.age.min","animal.age.max"]].mean(axis=1)

    unit_to_year = {
        "Month": 1/12,
        "Week": 1/52,
        "Day": 1/365,
        "Hour": 1/(24*365),
        "Minute": 1/(60*24*365),
        "Second": 1/(60*60*24*365),
    }

    for unit, factor in unit_to_year.items():
        df.loc[df["animal.age.unit"] == unit, "animal_age"] *= factor

    df.loc[df["animal.age.unit"] == "C17998", "animal_age"] = pd.NA

    # report-level aggregation
    df_report = (
        df.groupby("unique_aer_id_number")
        .agg({
            "animal_age":"mean",
            "animal.species":"first",
            "number_of_animals_affected_outcome":"mean"
        })
        .reset_index()
    )

    # species-weighted average
    species_avg = (
        df_report
        .groupby("animal.species")[["animal_age","number_of_animals_affected_outcome"]]
        .apply(
            lambda x:
            (x["animal_age"] * x["number_of_animals_affected_outcome"]).sum()
            / x["number_of_animals_affected_outcome"].sum()
        )
        .to_dict()
    )

    report_map = df_report.set_index("unique_aer_id_number")["animal_age"].to_dict()
    df["animal_age"] = df["unique_aer_id_number"].map(report_map)

    df["animal_age"] = df.apply(
        lambda r:
        species_avg.get(r["animal.species"], r["animal_age"])
        if pd.isna(r["animal_age"]) else r["animal_age"],
        axis=1
    )

    return df


def clean_weight(df):
    df = df.copy()

    df["animal_weight"] = df[["animal.weight.min","animal.weight.max"]].mean(axis=1)

    df_report = (
        df.groupby("unique_aer_id_number")
        .agg({
            "animal_weight":"mean",
            "animal.species":"first",
            "number_of_animals_affected_outcome":"mean"
        })
        .reset_index()
    )

    species_avg = (
        df_report
        .groupby("animal.species")[["animal_weight","number_of_animals_affected_outcome"]]
        .apply(
            lambda x:
            (x["animal_weight"] * x["number_of_animals_affected_outcome"]).sum()
            / x["number_of_animals_affected_outcome"].sum()
        )
        .to_dict()
    )

    report_map = df_report.set_index("unique_aer_id_number")["animal_weight"].to_dict()
    df["animal_weight"] = df["unique_aer_id_number"].map(report_map)

    df["animal_weight"] = df.apply(
        lambda r:
        species_avg.get(r["animal.species"], r["animal_weight"])
        if pd.isna(r["animal_weight"]) else r["animal_weight"],
        axis=1
    )

    return df


def clean_dose(df):
    df = df.copy()

    def make_fallback(row):
        return f"{row['dose.numerator']} {row['dose.numerator_unit']} / {row['dose.denominator']} {row['dose.denominator_unit']}"

    # Apply conversion
    # df_analysis_cleaned['dose_mg'] = df_analysis_cleaned.apply(convert_to_mg, axis=1)
    df['dose'] = df.apply(make_fallback, axis=1)
    return df


def normalize_basic_columns(df):
    # Basic column cleaning (Gender, Administration Route, Dosage Form, Medical Status)
    df = df.copy()

    print("Normalizing gender...")
    df.loc[df["animal.gender"].isin(["Mixed","Unknown"]), "animal.gender"] = pd.NA
    df["animal.gender"] = df.groupby("animal.species")["animal.gender"].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else pd.NA)
    )
    print("Normalizing route of administration...")
    df.loc[df["route"].isin(["Other","Unknown","Unassigned"]), "route"] = pd.NA
    df["route"] = df.groupby("animal.species")["route"].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else pd.NA)
    )

    print("Normalizing dosage form...")
    df.loc[df["dosage_form"].isin(["Unassigned","Other"]), "dosage_form"] = pd.NA
    df["dosage_form"] = df.groupby("animal.species")["dosage_form"].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else pd.NA)
    )

    print("Normalizing medical status...")
    df = df[df["medical_status"] != "Euthanized"]
    df["atc_vet_code"] = df["atc_vet_code"].str.strip()

    valid_ids = (
        df.groupby("unique_aer_id_number")["medical_status"]
        .nunique()
        .loc[lambda s: s == 1]
        .index
    )

    df = df[df["unique_aer_id_number"].isin(valid_ids)]
    return df


def process_breed_component(value):
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed[0]
        return value
    except:
        return value

