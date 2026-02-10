import os
import json
import glob
import time
import requests
import pandas as pd
from tqdm import tqdm
import pubchempy as pcp


class PubChemClient:
    """
    A client class for interacting with PubChem REST APIs and local
    PUG-View JSON records.

    Capabilities:
    - Query compound CIDs by name
    - Retrieve related compounds and synonyms
    - Download PUG-View JSON records
    - Recursively extract chemical/physical properties
    - Parse multiple records into a unified pandas DataFrame
    """
        
    def __init__(
        self,
        base_url: str = "https://pubchem.ncbi.nlm.nih.gov/rest",
        conversion_json: str = "../data/drugs_conversion.json",
        pugview_dir: str = "../data/pugviews_json",
    ):
        self.BASE = base_url
        self.conversion_json = conversion_json
        self.pugview_dir = pugview_dir

        os.makedirs(self.pugview_dir, exist_ok=True)

        # Load or initialize conversion dictionary
        if os.path.exists(self.conversion_json):
            with open(self.conversion_json, "r") as f:
                self.drugs_conversion = json.load(f)
        else:
            self.drugs_conversion = {}


    def build_drug_to_cid_map(self, drugs_name: list[str], save_every: int = 100):
        """
        Build or update a mapping from drug names → PubChem CID and
        persist it incrementally to the conversion JSON file.

        Args:
            drugs_name (list[str]): List of drug names to query.
            save_every (int): Save to disk every N records (default=100).

        Side effects:
            - Updates self.drugs_conversion in memory
            - Periodically writes to self.conversion_json on disk

        Returns:
            dict: Updated drugs_conversion dictionary.
        """
        save_counter = 0

        for drug in tqdm(drugs_name, desc="Building drug to CID map"):
            if drug in self.drugs_conversion:
                continue

            compound_drug = pcp.get_compounds(drug, 'name')
            time.sleep(0.22)  # avoid rate limits

            if compound_drug:
                self.drugs_conversion[drug] = compound_drug[0].cid
            else:
                self.drugs_conversion[drug] = None

            # Periodic checkpoint save
            if save_counter % save_every == 0:
                with open(self.conversion_json, "w") as f:
                    json.dump(self.drugs_conversion, f, indent=4)

            save_counter += 1

        # Final save
        with open(self.conversion_json, "w") as f:
            json.dump(self.drugs_conversion, f, indent=4)

        return self.drugs_conversion


    def get_related_compounds(self, name: str = "ivermectin"):
        """
        Retrieve PubChem CIDs and synonyms related to a compound name.

        This method:
        1. Queries PubChem for all matching compound IDs (CIDs).
        2. Uses pubchempy to fetch compound metadata.
        3. Collects known synonyms for each CID.

        Args:
            name (str): Compound name to search for.

        Returns:
            dict[int, list[str]]:
                Mapping from CID to list of synonyms.
        """
        name = name.replace(" / ", " . ")

        url = f"{self.BASE}/pug/compound/name/{name}/cids/JSON?name_type=word"
        r = requests.get(url)
        r.raise_for_status()

        cids = r.json().get("IdentifierList", {}).get("CID", [])
        # print("CIDs found:", cids)

        related = {}

        for cid in cids:
            c = pcp.Compound.from_cid(cid)
            time.sleep(0.25)  # avoid rate limit
            related[cid] = c.synonyms

        return related


    def fetch_pubchem_pugview(self, cid: int, output_file: str = None):
        """
        Download the PUG-View JSON record for a given compound CID.

        Args:
            cid (int): PubChem Compound ID.
            output_file (str, optional): Path to save the JSON file. If not provided,
                the file will be saved in the default PUG-View directory.

        Returns:
            str or None: Path to the saved JSON file, or None if download failed.
        """
        url = f"{self.BASE}/pug_view/data/compound/{cid}/JSON"

        if output_file is None:
            output_file = os.path.join(self.pugview_dir, f"{cid}.json")

        response = requests.get(url, allow_redirects=True)

        if response.status_code == 200:
            try:
                data = response.json()
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                return output_file
            except ValueError:
                print(f"⚠️ Response was not valid JSON for CID {cid}.")
                return None
        else:
            print(f"❌ Failed to fetch data for CID {cid}. Status code: {response.status_code}")
            return None


    @staticmethod
    def extract_info(section, data_dict):
        """
        Recursively extract structured information from a PubChem
        PUG-View JSON 'Section' object.

        The function walks through nested sections and normalizes
        different value types (String, Number, Boolean, Date, etc.)
        into plain Python strings stored in `data_dict`.

        Args:
            section (dict):
                A nested PubChem JSON section.
            data_dict (dict):
                Dictionary being populated with extracted key-value pairs.
        """
        section_title = section.get("TOCHeading", "Unknown Section")

        if "Information" in section:
            for info in section["Information"]:
                key = info.get("Name") or section_title
                val = info.get("Value", {})
                value_str = None

                if "StringWithMarkup" in val:
                    strings = [s.get("String", "") for s in val["StringWithMarkup"]]
                    value_str = " | ".join(strings)

                elif "Number" in val:
                    num = val["Number"][0]
                    unit = val.get("Unit")
                    value_str = f"{num} {unit}" if unit else str(num)

                elif "Boolean" in val:
                    value_str = str(val["Boolean"][0])

                elif "DateISO8601" in val:
                    value_str = val["DateISO8601"][0]

                elif "Integer" in val:
                    value_str = str(val["Integer"][0])

                elif "Float" in val:
                    value_str = str(val["Float"][0])

                elif "String" in val:
                    value_str = str(val["String"])

                if value_str is not None:
                    data_dict[key] = value_str

        if "Section" in section:
            for subsection in section["Section"]:
                PubChemClient.extract_info(subsection, data_dict)


    def parse_pubchem_json(self, file_path: str) -> pd.DataFrame:
        """
        Parse a single downloaded PUG-View JSON file into a DataFrame.

        Focuses specifically on the
        "Chemical and Physical Properties" section.

        Args:
            file_path (str): Path to a local PubChem JSON file.

        Returns:
            pandas.DataFrame:
                One-row DataFrame containing extracted properties.
        """
        with open(file_path, "r") as f:
            record = json.load(f)["Record"]

        data_dict = {
            "CID": record.get("RecordNumber"),
            "Title": record.get("RecordTitle"),
        }

        for section in record.get("Section", []):
            if section.get("TOCHeading") == "Chemical and Physical Properties":
                for subsection in section.get("Section", []):
                    for subsubsection in subsection.get("Section", []):
                        self.extract_info(subsubsection, data_dict)
                break

        return pd.DataFrame([data_dict])


    def parse_all_pugviews(self) -> pd.DataFrame:
        """
        Parse all downloaded PUG-View JSON files in the local directory
        into a single concatenated DataFrame.

        Returns:
            pandas.DataFrame:
                Aggregated table with one row per compound.
        """
        json_files = glob.glob(os.path.join(self.pugview_dir, "*.json"))
        dfs = [self.parse_pubchem_json(fp) for fp in tqdm(json_files, desc="Parsing PUG-View JSON files")]
        return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":

    # Initialize client
    client = PubChemClient()

    print("=== Testing PubChemClient ===")

    # --------------------------------------------------
    # Test 1: Get related compounds
    # --------------------------------------------------
    print("\n[TEST] get_related_compounds")
    related = client.get_related_compounds("ivermectin")
    first_cid = list(related.keys())[0]
    print(f"Found CIDs: {list(related.keys())[:5]}")

    # --------------------------------------------------
    # Test 2: Fetch one PUG-View JSON
    # --------------------------------------------------
    print("\n[TEST] fetch_pubchem_pugview")
    out_path = client.fetch_pubchem_pugview(first_cid)
    print(f"Saved PUG-View JSON to: {out_path}")

    # --------------------------------------------------
    # Test 3: Parse a single file
    # --------------------------------------------------
    print("\n[TEST] parse_pubchem_json")
    df_single = client.parse_pubchem_json(out_path)
    print("Single-record DataFrame:")
    print(df_single.head())

    # --------------------------------------------------
    # Test 4: Parse all files in directory
    # --------------------------------------------------
    print("\n[TEST] parse_all_pugviews")
    df_all = client.parse_all_pugviews()
    print("All records shape:", df_all.shape)
    print(df_all.head())

    print("\n=== All tests completed successfully ===")
