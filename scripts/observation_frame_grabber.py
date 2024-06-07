import subprocess
from pathlib import Path
from datetime import datetime, timedelta

# Grabs a list of satellites from the satellites_to_observe file
# observe_directory - Where the satellites-to-observe file is
# Returns satellites - A list containing satellite names and norad_ids (list)
def read_satellites(observe_directory, observe_file):
    observe_path = observe_directory / observe_file

    satellites = []
    
    # / operator joins the observe_path and the name
    with open(observe_path, mode="r") as f:
        # Skips the header comment 
        f.readline()

        for line in f:
            line = line.strip().split(" ")

            # Parses the satellite name and norad_id into a dictionary
            satellite = {"satellite_name": line[0], "norad_id": line[1]}
        
            satellites.append(satellite)

    return satellites


# Gets the frames from each satellite, fetching it into a location
# satellite - The satellite that is tested
# decoder_path - Path of fetch_frames_from_network.py of the SatNOGS_decoder
# frames_path - Where to download the frames
# start, end - Period of time when the frames will be fetched 
def get_frames(satellite, decoder_path, frames_path, start, end):
    satellite_name = satellite["satellite_name"]

    # Makes a new directory for the satellite along with when the frames were fetched from the network
    new_dir_name = f'{satellite_name}_{start}--{end}'
    target_path = f'{frames_path}/{new_dir_name}'

    # Define new commands to run
    
    mkdir_command = f'mkdir -p {target_path}'    
    fetch_frames_command = f'{decoder_path} {satellite["norad_id"]} {start} {end} {target_path}'

    commands = f'{mkdir_command} && {fetch_frames_command}'

    # Run the list of commands simultanously for each satellite
    process = subprocess.Popen(commands, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Outputs how many frames feteched for each satellite
    for line in process.stdout:
        print(satellite_name)
        print(line)


def main():
    search_directory = Path.home().joinpath("ground-station/") # Directory to search
    decoder_path = Path(next(search_directory.rglob("fetch_frames_from_network.py"))) # Path of the SatNOGS_decoder
    frames_path = Path(next(search_directory.rglob("autovetter/frames/"))) # Where to download the frames
    
    # Adjust when to grab observations
    rewind = timedelta(days=-2) # How far back to grab frames from (Days)
    start = (datetime.now() + rewind).strftime("%Y-%m-%d") # Today's date (formatted)
    end = datetime.now().strftime("%Y-%m-%d") # How far in the future to get good observations (formatted)

    # Grab list of satellites
    observe_directory = Path(next(search_directory.rglob("Downlinkfiles/"))) 
    satellites = read_satellites(observe_directory, observe_file="satellites_to_observe.txt")

    for satellite in satellites:
        get_frames(satellite, decoder_path, frames_path, start, end)


if __name__ == "__main__":
    main()