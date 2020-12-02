for actor_id in {01..24}
do
    wget https://zenodo.org/record/1188976/files/Video_Speech_Actor_${actor_id}.zip
    unzip Video_Speech_Actor_${actor_id}.zip
    rm Video_Speech_Actor_${actor_id}.zip
done

wget https://zenodo.org/record/3255102/files/FacialTracking_Actors_01-24.zip
unzip FacialTracking_Actors_01-24.zip
rm FacialTracking_Actors_01-24.zip
mv FacialTracking_Actors_01-24 landmarks
