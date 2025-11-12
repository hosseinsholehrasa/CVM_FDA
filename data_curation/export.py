import os
import json
import pandas as pd
import csv

from io import StringIO
from sqlalchemy import create_engine
from typing import Tuple
from pandas import DataFrame
from tqdm import tqdm


def extract_cvm2dfs(json_dir: str = "../data/json_files/") -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    list_all_results = []
    list_all_drugs = []
    list_all_reactions = []
    list_all_outcomes = []

    for folder_name in tqdm(sorted(os.listdir(json_dir)[:])):
        if not os.path.isdir(f'{json_dir}/{folder_name}'):
            continue
        # folder_name = "2005 Q4 (all)"
        with open(f'{json_dir}/{folder_name}/animalandveterinary-event-0001-of-0001.json', 'r') as json_file:
            json_object = json.load(json_file)
            results = json_object['results']

            # normalize data
            df_results = pd.json_normalize(results)

            # add the folder name to the dataframe which is the date
            df_results['date'] = folder_name.replace('(all)', "")

            # Get reactions, outcomes and drugs from the dataframe
            df_reactions = df_results[['reaction', 'date', 'unique_aer_id_number']]
            df_outcomes = df_results[['outcome', 'date', 'unique_aer_id_number']]
            df_drugs = df_results[['drug', 'date', 'unique_aer_id_number']]

            # Drop outcomes and reactions and drugs from the df_results
            df_results = df_results.drop(columns=['drug', 'reaction', 'outcome'])
            # add the dataframes to the all dataframes list
            list_all_results.append(df_results)

            # reactions
            # explode the dataframes which duplicates list values to the rows
            df_reactions = df_reactions.explode('reaction')
            df_reactions = df_reactions.reset_index(drop=True)
            # Flatten the column into multiple columns
            df_reaction_normalized = pd.json_normalize(df_reactions['reaction'])
            # add the date and unique_aer_id_number to the normalized dataframe
            df_reaction_normalized['date'] = df_reactions['date']
            df_reaction_normalized['unique_aer_id_number'] = df_reactions['unique_aer_id_number']
            # add the dataframes to the list
            list_all_reactions.append(df_reaction_normalized)

            # outcomes
            # explode the dataframes which duplicates list values to the rows
            df_outcomes = df_outcomes.explode('outcome')
            df_outcomes = df_outcomes.reset_index(drop=True)
            # Flatten the column into multiple columns
            df_outcome_normalized = pd.json_normalize(df_outcomes['outcome'])
            # add the date and unique_aer_id_number to the normalized dataframe
            df_outcome_normalized['date'] = df_outcomes['date']
            df_outcome_normalized['unique_aer_id_number'] = df_outcomes['unique_aer_id_number']
            # add the dataframes to the list
            list_all_outcomes.append(df_outcome_normalized)

            # drugs
            # explode the dataframes which duplicates list values to the rows
            df_drugs = df_drugs.explode('drug')
            df_drugs = df_drugs.reset_index(drop=True)
            # Flatten the column into multiple columns
            df_drug_normalized = pd.json_normalize(df_drugs['drug'])
            # add the date and unique_aer_id_number to the normalized dataframe
            df_drug_normalized['date'] = df_drugs['date']
            df_drug_normalized['unique_aer_id_number'] = df_drugs['unique_aer_id_number']
            df_drug_normalized['drug_id'] = df_drug_normalized.apply(lambda row: f"{row['unique_aer_id_number']}_{row.name}", axis=1)
            # Active ingredients in drugs
            # explode the active ingredients in drugs
            df_drug_normalized = df_drug_normalized.explode('active_ingredients')
            df_drug_normalized = df_drug_normalized.reset_index(drop=True)

            # Flatten the column into multiple columns
            df_drug_active_ingredients = pd.json_normalize(df_drug_normalized['active_ingredients'])

            # drop the active ingredients column from the normalized drugs dataframe
            df_drug_normalized = df_drug_normalized.drop(columns=['active_ingredients'])

            # add all columns of df_drug_active_ingredients to the df_drug_normalized
            df_drug_normalized['name'] = df_drug_active_ingredients['name']
            df_drug_normalized['dose.numerator'] = df_drug_active_ingredients['dose.numerator']
            df_drug_normalized['dose.numerator_unit'] = df_drug_active_ingredients['dose.numerator_unit']
            df_drug_normalized['dose.denominator'] = df_drug_active_ingredients['dose.denominator']
            df_drug_normalized['dose.denominator_unit'] = df_drug_active_ingredients['dose.denominator_unit']

            # # add the dataframes to the list
            list_all_drugs.append(df_drug_normalized)

    # Concatenate the list of dataframes to a single dataframe
    df_all_results = pd.concat(list_all_results)
    df_all_reactions = pd.concat(list_all_reactions)
    df_all_outcomes = pd.concat(list_all_outcomes)
    df_all_drugs = pd.concat(list_all_drugs)

    # Add year to the dataframes
    df_all_results['year'] = df_all_results['date'].str.replace('Q1', '').str.replace('Q2', '').str.replace('Q3', '').str.replace('Q4', '')
    df_all_reactions['year'] = df_all_reactions['date'].str.replace('Q1', '').str.replace('Q2', '').str.replace('Q3', '').str.replace('Q4', '')
    df_all_outcomes['year'] = df_all_outcomes['date'].str.replace('Q1', '').str.replace('Q2', '').str.replace('Q3', '').str.replace('Q4', '')
    df_all_drugs['year'] = df_all_drugs['date'].str.replace('Q1', '').str.replace('Q2', '').str.replace('Q3', '').str.replace('Q4', '')

    return df_all_results, df_all_drugs, df_all_reactions, df_all_outcomes


def cvm2csv(json_dir: str = "../data/json_files/", save_dir: str = '../data') -> None:
    # Extract the data from the json files
    df_all_results, df_all_drugs, df_all_reactions, df_all_outcomes = extract_cvm2dfs(json_dir)
    print("Data extracted successfully")

    print("Saving data to csv")
    # save the dataframes to csv
    df_all_results.to_csv(f'{save_dir}/all.csv', index=False)
    df_all_drugs.to_csv(f'{save_dir}/drugs.csv', index=False)
    df_all_reactions.to_csv(f'{save_dir}/reactions.csv', index=False)
    df_all_outcomes.to_csv(f'{save_dir}/outcomes.csv', index=False)


def psql_insert_copy(table, conn, keys, data_iter):
    # gets a DBAPI connection that can provide a cursor
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        s_buf = StringIO()
        writer = csv.writer(s_buf)
        writer.writerows(data_iter)
        s_buf.seek(0)

        columns = ', '.join('"{}"'.format(k) for k in keys)
        if table.schema:
            table_name = '{}.{}'.format(table.schema, table.name)
        else:
            table_name = table.name

        sql = 'COPY {} ({}) FROM STDIN WITH CSV'.format(
            table_name, columns)
        cur.copy_expert(sql=sql, file=s_buf)


def cvm2postgres(engine, json_dir: str = "../data/json_files/", data_source="existing_data", save_dir: str = "../data/") -> None:
    if data_source == "existing_data":
        print("Reading data from csv")
        df_all_results = pd.read_csv(f'{save_dir}/all.csv', low_memory=False)
        df_all_drugs = pd.read_csv(f'{save_dir}/drugs.csv', low_memory=False)
        df_all_reactions = pd.read_csv(f'{save_dir}/reactions.csv', low_memory=False)
        df_all_outcomes = pd.read_csv(f'{save_dir}/outcomes.csv', low_memory=False)

    elif data_source == "new_data":
        df_all_results, df_all_drugs, df_all_reactions, df_all_outcomes = extract_cvm2dfs(json_dir)

    else:
        raise ValueError("Data source should be either 'existing_data' or 'new_data'")

    print("Saving data to postgres")
    df_all_results.to_sql('results', engine, method=psql_insert_copy)
    df_all_drugs.to_sql('drugs', engine, method=psql_insert_copy)
    df_all_reactions.to_sql('reactions', engine, method=psql_insert_copy)
    df_all_outcomes.to_sql('outcomes', engine, method=psql_insert_copy)


if __name__ == '__main__':
    cvm2csv()

    #postgres_engine = create_engine('postgresql://postgres:animalpass@127.0.0.1:5432/animalvetdb')
    # check if connection is successful
    # print(postgres_engine)

    #cvm2postgres(postgres_engine, data_source="existing_data")
