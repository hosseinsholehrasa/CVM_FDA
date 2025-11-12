import os
import requests
import zipfile
# create a bar to show the progress of the download
from tqdm import tqdm


def download_and_extract_files(
        url: str = "https://api.fda.gov/download.json",
        save_path: str = "../data/json_files/"
) -> None:

    # Make a folder if not exist to save the files
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Open FDA download lists
    response = requests.get(url)
    response = response.json()
    cvm_data = response['results']['animalandveterinary']['event']['partitions']

    for data in tqdm(cvm_data):
        # Download each CVM file and save it as zip
        file_name = data['display_name']
        file_url = data['file']
        file = requests.get(file_url).content
        with open(save_path + file_name + ".zip", 'wb') as f:
            f.write(file)

        # Extract each zip file. It will save a json in a folder for each extracted file to avoid duplicated json names
        with zipfile.ZipFile(save_path + file_name + ".zip", 'r') as zip_ref:
            zip_ref.extractall(f'{save_path}/{file_name}/')

        # Remove the zip file after extracting
        os.remove(save_path + file_name + ".zip")


def merge_jsons(json_dir: str = "../data/json_files/") -> None:
    """
    Merge all json files in a directory to a single json file with the key of the folder name
    :param json_dir:
    :return:
    """
    # create a json file to save the merged jsons with key all_data: []
    with open('data/all_data.json', 'w') as f:
        f.write('{"all_data": []}')

    all_data = {}
    # Open the json file to append the data
    with open('../data/all_data.json', 'r') as merged_json:
        all_data = eval(merged_json.read())

    for folder_name in tqdm(sorted(os.listdir(json_dir))):
        # check if it is a folder
        if not os.path.isdir(f'{json_dir}/{folder_name}'):
            continue

        # Open each json file and append the data to the all_data key
        with open(f'{json_dir}/{folder_name}/animalandveterinary-event-0001-of-0001.json', 'r') as json_file:
            data = json_file.read()
            all_data['all_data'].append({folder_name: eval(data)})
            print(f"Data from {folder_name} added to the merged json")

    with open('../data/all_data.json', 'w') as merged_json:
        # Save the merged json to the json file
        merged_json.write(str(all_data))


if __name__ == '__main__':
      download_and_extract_files()
      print("Files downloaded and extracted successfully")
      merge_jsons()

    # with open('data/all_data.json', 'r') as merged_json:
    #     all_data = eval(merged_json.read())
    #     print(all_data['all_data'][0])
    #     print("Data loaded successfully")
