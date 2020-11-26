#!/bin/bash
fileid="1Hg_s8dH7lLfXElHewsKPFRbdoBDj1IC9"
filename="checkpoints.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
rm cookie
unzip checkpoints.zip
rm checkpoints.zip
