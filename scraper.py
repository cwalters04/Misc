import os
import re

import requests


def download_data(norad_cat_id, obs_id):
    """
    Recieves a satellite id and retrieves the data from satnogs network.
    The data is partitioned into satelite id -> transmitter -> audios & images


    Args:
        norad_cat_id (The id of the satelite)
    """
    # Define the base path for data storage
    if norad_cat_id:
        base_path = "data/{}".format(norad_cat_id)
    else:
        base_path = "data/test_specific_id"
    # Define the API endpoint URL
    if norad_cat_id:
        api_url = f"https://network.satnogs.org/api/observations/?satellite__norad_cat_id={norad_cat_id}"
    else:
        api_url = f"https://network.satnogs.org/api/observations/?id={obs_id}"

    next_page_url = api_url
    # Loop to get fetch all pages
    while next_page_url:

        # Make the GET request to the API endpoint
        response = requests.get(next_page_url)
        print("Fetching url: ", next_page_url)

        # Check if the request was successful
        if response.status_code == 200:  # Parse the JSON response
            data = response.json()

            # Iterate over the observations and download the images
            for observation in data:

                # Create path with observation
                transmitter_description = observation["transmitter_description"]

                transmitter_path = os.path.join(base_path, transmitter_description)
                images_path = os.path.join(transmitter_path, "images")
                audios_path = os.path.join(transmitter_path, "audios")
                decoded_path = os.path.join(transmitter_path, "decoded")

                if not os.path.exists(images_path):
                    os.makedirs(images_path)
                if not os.path.exists(audios_path):
                    os.makedirs(audios_path)
                if not os.path.exists(decoded_path):
                    os.makedirs(decoded_path)

                # Extract the URL of the waterfall image
                waterfall_url = observation["waterfall"]
                audio_url = observation["archive_url"]
                if "demoddata" in observation:
                    decoded_urls = observation["demoddata"]
                else:
                    decoded_urls = []

                # Extract the waterfall_file_name from the URL
                if waterfall_url:
                    waterfall_file_name = waterfall_url.split("/")[-1]
                if audio_url:
                    audio_file_name = audio_url.split("/")[-1]

                # Download the image
                if waterfall_url:
                    print("Image reponse url: ", waterfall_url)
                    image_response = requests.get(waterfall_url)
                    if image_response.status_code == 200:
                        # Save the image to the folder
                        with open(
                            os.path.join(images_path, waterfall_file_name), "wb"
                        ) as f:
                            f.write(image_response.content)
                        print(f"Downloaded waterfall: {waterfall_file_name}")
                    else:
                        print(f"Failed to download {waterfall_file_name}")

                # Download audio
                if audio_url:
                    print("Audio reponse url: ", audio_url)
                    audio_response = requests.get(audio_url)
                    if audio_response.status_code == 200:
                        # Save the image to the folder
                        with open(
                            os.path.join(audios_path, audio_file_name), "wb"
                        ) as f:
                            f.write(audio_response.content)
                        print(f"Downloaded audio: {audio_file_name}")
                    else:
                        print(f"Failed to download {audio_file_name}")

                # Decoded data
                print("Decoded data reponse urls: ", decoded_urls)

                # Decoded urls extration from list
                decoded_urls_clean = []
                for d in decoded_urls:
                    print("d is: ", d)
                    decoded_urls_clean.append(d["payload_demod"])
                decoded_urls = decoded_urls_clean
                print(decoded_urls)

                # Download  decoded data
                if decoded_urls:
                    for url in decoded_urls:
                        decoded_file_name = url.split("/")[-1]
                        decoded_response = requests.get(url)
                        if decoded_response.status_code == 200:
                            # Extract the file name from the URL
                            with open(
                                os.path.join(decoded_path, decoded_file_name), "wb"
                            ) as f:
                                f.write(decoded_response.content)
                            print(f"Downloaded decoded data: {decoded_file_name}")
                        else:
                            print(f"Failed to download {decoded_file_name}")

        else:
            print("Failed to fetch data from the API")

        # Pattern to extract the next page url
        pattern = r'<(https://[^>]+)>; rel="next"'
        response_head = requests.head(api_url)
        headers = response_head.headers

        if headers.get("link"):
            next_page_url_raw = headers["link"]
            next_page_url = re.search(pattern, next_page_url_raw).group(1)
            print("Next page url is: ", next_page_url)
        else:
            next_page_url = None

        if response_head.status_code != 200:
            break


# Get the NORAD Catalog ID from the user
norad_cat_id = input("Please enter the NORAD Catalog ID, example 98901: ")
if norad_cat_id:
    norad_cat_id = int(norad_cat_id)
obs_id = input("Please enter the observation ID, example 9636487: ")
if obs_id:
    obs_id = int(obs_id)


# Example  norad_cat_id = 98901


# Call the function to download images
# TODO:

download_data(norad_cat_id, obs_id)
print("Finished Extraction")
