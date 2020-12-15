#!/bin/bash -l
if [ $# -eq 0 ]; then
    echo "Download path not provided"
    exit 1
fi

for actor_id in {01..24}
do
    wget https://zenodo.org/record/1188976/files/Video_Speech_Actor_${actor_id}.zip
    unzip Video_Speech_Actor_${actor_id}.zip -d $(realpath "$1")
    rm Video_Speech_Actor_${actor_id}.zip
done

wget https://zenodo.org/record/3255102/files/FacialTracking_Actors_01-24.zip
unzip FacialTracking_Actors_01-24.zip -d $(realpath "$1")/landmarks
rm FacialTracking_Actors_01-24.zip
