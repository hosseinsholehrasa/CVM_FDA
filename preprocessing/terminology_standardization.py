import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup


class HLTMapper:
    def __init__(
        self,
        veddra2018_xls="../data/vm_reaction2018.xls",
        veddra2018_add="../data/vm_reaction_additional.xls",
        veddra2025_xlsx="../data/vm_reaction2025.xlsx",
    ):
        self.veddra2018_xls = veddra2018_xls
        self.veddra2018_add = veddra2018_add
        self.veddra2025_xlsx = veddra2025_xlsx

        # will be populated after loading
        self.df_new = None
        self.df_old = None
        self.df_new_all = None
        self.df_old_all = None
        self.df_new_code = None
        self.df_old_code = None


    def load_reference_files(self):
        rc_LTT = pd.read_excel(self.veddra2018_xls, sheet_name="LTT", skiprows=11)
        rc_PT  = pd.read_excel(self.veddra2018_xls, sheet_name="PT",  skiprows=11)
        rc_HLT = pd.read_excel(self.veddra2018_xls, sheet_name="HLT", skiprows=11)
        rc_SOC = pd.read_excel(self.veddra2018_xls, sheet_name="SOC", skiprows=11)
        df_new = pd.read_excel(self.veddra2025_xlsx)

        rc2_LPHS = pd.read_excel(
            self.veddra2018_add,
            sheet_name="Adverse Event Terms",
            skiprows=1
        )
        rc2_LPHS = rc2_LPHS[
            ['LLT_CODE','LLT_NAME','PT_CODE','PT_NAME',
             'HLT_CODE','HLT_NAME','SOC_CODE','SOC_NAME']
        ]

        # ---- process 2018 terms ----
        rc_LTT['LLTNew'] = np.where(
            rc_LTT['DEPRECATED']=='Y',
            rc_LTT['REMAP_TO_CODE'],
            rc_LTT['LLT Key']
        )

        rc_LTT_key = rc_LTT[['LLT Key', 'PARENT_CODE']]

        rc_LTT_REMAP = pd.merge(
            rc_LTT[['LLTNew','LLT Term']],
            rc_LTT_key[['LLT Key','PARENT_CODE']],
            left_on=['LLTNew'],
            right_on=['LLT Key'],
            how='left'
        )

        rc_LP = pd.merge(
            rc_LTT_REMAP[['LLTNew','LLT Term','PARENT_CODE']],
            rc_PT[['PT Key','PT Term','PARENT_CODE']],
            left_on=['PARENT_CODE'],
            right_on=['PT Key'],
            how='left'
        )

        rc_LPH = pd.merge(
            rc_LP,
            rc_HLT[['HLT Key','HLT Term','PARENT_CODE']],
            left_on=['PARENT_CODE_y'],
            right_on=['HLT Key'],
            how='left'
        )

        rc_LPHS = pd.merge(
            rc_LPH,
            rc_SOC[['SOC Key','SOC Term']],
            left_on=['PARENT_CODE'],
            right_on=['SOC Key'],
            how='left'
        )

        rc1_LPHS = rc_LPHS[
            ['LLTNew','LLT Term','PT Key','PT Term',
             'HLT Key','HLT Term','SOC Key','SOC Term']
        ]

        rc1_LPHS.columns = [
            'LLT_CODE','LLT_NAME','PT_CODE','PT_NAME',
            'HLT_CODE','HLT_NAME','SOC_CODE','SOC_NAME'
        ]

        rc_all = pd.concat([rc1_LPHS, rc2_LPHS])
        df_old = rc_all.drop_duplicates(['LLT_CODE'])
        df_old = df_old.map(lambda s: s.lower() if isinstance(s,str) else s)

        # ---- process 2025 terms ----
        df_new.columns = [
            "index","SOC_NAME","SOC_KEY","HLT_NAME",
            "HLT_CODE","PT_NAME","PT_CODE",
            "LLT_NAME","LLT_CODE","TERM_TYPE"
        ]
        df_new = df_new.map(lambda s: s.lower() if isinstance(s,str) else s)

        # ---- build lookup tables (new) ----
        df_new_pt = df_new[["PT_NAME","HLT_NAME"]].rename(
            columns={"PT_NAME":"veddra_term_name"}
        )
        df_new_llt = df_new[["LLT_NAME","HLT_NAME"]].rename(
            columns={"LLT_NAME":"veddra_term_name"}
        )
        df_new_hlt = df_new[["HLT_NAME"]].rename(
            columns={"HLT_NAME":"veddra_term_name"}
        )
        df_new_hlt["HLT_NAME"] = df_new_hlt["veddra_term_name"]

        df_new_all = pd.concat(
            [df_new_pt, df_new_llt, df_new_hlt],
            ignore_index=True
        )

        # ---- build lookup tables (old) ----
        df_old_pt = df_old[["PT_NAME","HLT_NAME"]].rename(
            columns={"PT_NAME":"veddra_term_name"}
        )
        df_old_llt = df_old[["LLT_NAME","HLT_NAME"]].rename(
            columns={"LLT_NAME":"veddra_term_name"}
        )
        df_old_hlt = df_old[["HLT_NAME"]].rename(
            columns={"HLT_NAME":"veddra_term_name"}
        )
        df_old_hlt["HLT_NAME"] = df_old_hlt["veddra_term_name"]

        df_old_all = pd.concat(
            [df_old_pt, df_old_llt, df_old_hlt],
            ignore_index=True
        )

        df_new_code = df_new[["LLT_CODE","HLT_NAME"]].rename(
            columns={"LLT_CODE":"veddra_term_code"}
        )
        df_old_code = df_old[["LLT_CODE","HLT_NAME"]].rename(
            columns={"LLT_CODE":"veddra_term_code"}
        )

        # store everything as attributes
        self.df_new = df_new
        self.df_old = df_old
        self.df_new_all = df_new_all
        self.df_old_all = df_old_all
        self.df_new_code = df_new_code
        self.df_old_code = df_old_code

        return self


    def map_veddra_to_hlt(self, df_clean):
        if self.df_new_all is None:
            raise ValueError(
                "Call load_reference_files() before mapping."
            )

        unique_terms = (
            df_clean[["veddra_term_name","veddra_term_code"]]
            .drop_duplicates()
        )

        merged = (
            unique_terms
            .merge(self.df_new_all, on="veddra_term_name", how="left")
            .merge(
                self.df_old_all,
                on="veddra_term_name",
                how="left",
                suffixes=("_new","_old")
            )
            .merge(self.df_new_code, on="veddra_term_code", how="left")
            .merge(
                self.df_old_code,
                on="veddra_term_code",
                how="left",
                suffixes=("_new_code","_old_code")
            )
        )

        merged["HLT_NAME"] = (
            merged["HLT_NAME_new"]
            .combine_first(merged["HLT_NAME_old"])
            .combine_first(merged["HLT_NAME_new_code"])
            .combine_first(merged["HLT_NAME_old_code"])
        )

        mask = merged["veddra_term_name"].str.contains(
            "lack of efficacy", case=False, na=False
        )
        merged.loc[mask,"HLT_NAME"] = "lack of efficacy"

        veddra2hlt = {}
        for term, hlt in tqdm(
            zip(merged["veddra_term_name"], merged["HLT_NAME"]),
            total=len(merged),
            desc="🔁 Building veddra → HLT dictionary..."
        ):
            veddra2hlt[term] = hlt

        not_found = merged.loc[
            merged["HLT_NAME"].isna(),
            "veddra_term_name"
        ].tolist()

        return veddra2hlt, not_found


class ATCVetConverter:
    def __init__(self, dict_path="../data/atc_vet_code_dict.json"):
        self.dict_path = dict_path
        self.cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.dict_path):
            with open(self.dict_path, "r") as f:
                return json.load(f)
        return {}

    def save(self):
        os.makedirs(os.path.dirname(self.dict_path), exist_ok=True)
        with open(self.dict_path, "w") as f:
            json.dump(self.cache, f, indent=4)

    def atcvet_html_info(self, atcvet_code: str) -> str:
        url = f"https://www.whocc.no/atcvet/atcvet_index/?code={atcvet_code}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        content_html = soup.find("div", id="content")
        if content_html is None:
            raise ValueError(f"Could not find content div for ATCvet code: {atcvet_code}")
        # Our specific p tag has no id or class name so we have to find it by index
        p_tags = content_html.find_all("p")
        atcvet_tag = p_tags[1]
        return atcvet_tag.text
    
    def atcvet_single_info(self,atcvet_code: str) -> str:
        """
        Return the specific ATCvet code information from the WHOCC website
        Example:
            QP54AB52: moxidectin, combinations
        """
        atcvet_info = self.atcvet_html_info(atcvet_code)
        atcvet_list = atcvet_info.split('\n')
        for atcvet in atcvet_list:
            if atcvet_code in atcvet:
                atcvet_info = atcvet
                break
        atcvet_info = atcvet_info.replace(atcvet_code, '')
        return (f"{atcvet_code}: {atcvet_info}")

    def atcvet_dict(self,atcvet_code: str) -> dict:
        """
        Return the ATCvet information as a dictionary
        Example:
        {'QP': 'ANTIPARASITIC PRODUCTS, INSECTICIDES AND REPELLENTS', 'QP54': 'ENDECTOCIDES',
        'QP54A': 'MACROCYCLIC LACTONES', 'QP54AB': 'Milbemycins', 'QP54AB52': 'moxidectin, combinations'}

        """
        atcvet_dict = {}
        atcvet_info = self.atcvet_html_info(atcvet_code)
        atcvet_list = atcvet_info.split('\n')
        for atcvet in atcvet_list:
            splited_sentence = atcvet.split(" ", 1)
            if splited_sentence == ['']:
                continue
            code, info = splited_sentence[0], splited_sentence[1]
            atcvet_dict[code] = info

        return atcvet_dict
    
    def atcvet_converter(self, atc_vet_code: str) -> str:
        code_len = len(str(atc_vet_code))

        if code_len in [6, 8, 9]:
            atc_dict = self.atcvet_dict(atc_vet_code)
            keys = list(atc_dict.keys())

            try:
                atc_code = keys[3]  # 4th level
            except:
                print("Fallback triggered for:", atc_vet_code)
                print("Keys:", keys)
                atc_code = keys[-1]

            result = self.atcvet_single_info(atc_code)
            result = result.split(":")[1].strip()

            # special cases
            if "QP51AH" in atc_vet_code:
                return "Polyether ionophores"
            elif "QJ51DA" in atc_vet_code:
                return "Intramammary cephalosporins"

            return result

        else:
            return atc_vet_code
        
    @staticmethod
    def clean_atc_codes_series(atc_series: pd.Series) -> pd.Series:
        """
        Clean ATC-Vet codes on a Series of atc vet codes.
        """

        invalid = ["UNKNOWN", "unknown", "na", "NOT APPLIC"]

        # Remove invalid labels
        s = atc_series[~atc_series.isin(invalid)]

        # Remove codes of length 2, 4, 5, or 7 which are invalid atc vet codes.
        s = s[s.apply(lambda x: len(str(x)) not in [2, 4, 5, 7])]

        return s

    def build_dictionary_from_series(
        self,
        atc_series: pd.Series,
        save_every: int = 50
    ):
        """
        Build (or extend) the ATC-Vet dictionary from a Series of codes.

        Args:
            atc_series: pandas Series containing atc_vet_code values
            save_every: save cache every N new conversions
        """

        # keep only unique, non-null values
        unique_codes = atc_series.dropna().unique()

        newly_added = 0

        for code in tqdm(unique_codes, desc="ATCVet conversion"):
            if code not in self.cache:
                try:
                    self.cache[code] = self.atcvet_converter(code)
                    newly_added += 1
                except Exception as e:
                    print(f"Error for {code}: {e}")
                    self.cache[code] = None
                    newly_added += 1  # still count it so we don’t retry endlessly

                # periodic save
                if newly_added % save_every == 0:
                    print(f"💾 Saving progress after {newly_added} new codes...")
                    self.save()

        # final save
        print(f"✅ Final save the atc vet code cache.")
        self.save()

        return self.cache
