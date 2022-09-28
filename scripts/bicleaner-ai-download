#!/bin/bash

usage() {
    echo "Script to download Bicleaner AI language packs."
    echo "It will try to download {lite,full}-lang1-lang2 and if it does not exist it will try {lite,full}-lang2-lang1 ."
    echo
    echo "Usage: `basename $0` <lang1> <lang2> <download_path>"
    echo "      <lang1>         Language 1."
    echo "      <lang2>         Language 2."
    echo "      {lite,full}     Download lite or full model."
    echo "      <download_path> Path where downloaded language pack should be placed. Will be ignored for full models."
}

invalid_url(){
    wget -S --spider -o - $1 | grep -q '404 Not Found'
}

if [[ $# -lt 3 ]]
then
    echo "Wrong number of arguments: $@" >&2
    usage >&2
    exit 1
fi

URL="https://github.com/bitextor/bicleaner-ai-data/releases/latest/download"
L1=$1
L2=$2
if [ "$3" != "lite" ] && [ "$3" != "full" ]; then
    echo "Model type must be 'lite' or 'full' not '$3'" 1>&2
    usage >&2
    exit 1
fi
TYPE=$3
if [ "$4" != "" ]; then
    DOWNLOAD_PATH=$4
else
    DOWNLOAD_PATH="."
fi

if [ "$TYPE" == "full" ]; then
    # Download from HF Hub
    bicleaner-ai-download-hf bitextor/bicleaner-ai-full-$L1-$L2
else
    # Download from github bitextor/bicleaner-ai-data
    # and decompress tgz in the desired directory
    if invalid_url $URL/$TYPE-$L1-$L2.tgz
    then
        >&2 echo $L1-$L2 language pack does not exist, trying $L2-$L1...
        if invalid_url $URL/$TYPE-$L2-$L1.tgz
        then
            >&2 echo $L2-$L1 language pack does not exist
        else
            wget -P $DOWNLOAD_PATH $URL/$TYPE-$L2-$L1.tgz
            tar xvf $DOWNLOAD_PATH/$TYPE-$L2-$L1.tgz -C $DOWNLOAD_PATH
            rm $DOWNLOAD_PATH/$TYPE-$L2-$L1.tgz
        fi
    else
        wget -P $DOWNLOAD_PATH $URL/$TYPE-$L1-$L2.tgz
        tar xvf $DOWNLOAD_PATH/$TYPE-$L1-$L2.tgz -C $DOWNLOAD_PATH
        rm $DOWNLOAD_PATH/$TYPE-$L1-$L2.tgz
    fi
fi

echo Finished
