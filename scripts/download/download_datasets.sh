#!/usr/bin/env bash

# This script is used to download datasets for model evaluation.
# Usage:
#     cd <TecoGAN-PyTorch_dir>
#     bash ./scripts/download/download_datasets <degradation_mode>



function download_large_file() {
    local DATA_DIR=$1
    local FID=$2
    local FNAME=$3
    
    wget --load-cookies ${DATA_DIR}/cookies.txt -O ${DATA_DIR}/${FNAME}.zip "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ${DATA_DIR}/cookies.txt --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?export=download&id=${FID}" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FID}" && rm -rf ${DATA_DIR}/cookies.txt
}

function download_small_file() {
    local DATA_DIR=$1
    local FID=$2
    local FNAME=$3
    
    wget --no-check-certificate -O ${DATA_DIR}/${FNAME}.zip "https://drive.google.com/uc?export=download&id=${FID}"
}

function check_md5() {
    local FPATH=$1
    local MD5=$2
    
    if [ ${MD5} != $(md5sum ${FPATH} | cut -d " " -f1) ]; then
        echo "!!! Fail to match MD5 sum for: ${FPATH}"
        echo "!!! Please try downloading it again"
        exit 1
    fi
}


# Vid4 GT
if [ ! -d "./data/Vid4/GT" ]; then
    DATA_DIR="./data/Vid4"
    FID="1T8TuyyOxEUfXzCanH5kvNH2iA8nI06Wj"
    MD5="d2850eccf30092418f15afe4a7ea27e5"
    
    echo ">>> Start to download [Vid4 GT] dataset"
    
	mkdir -p ${DATA_DIR}
	download_large_file ${DATA_DIR} ${FID} GT
	check_md5 ${DATA_DIR}/GT.zip ${MD5}
	unzip ${DATA_DIR}/GT.zip -d ${DATA_DIR} && rm ${DATA_DIR}/GT.zip
fi

sleep 1s

# ToS3 GT
if [ ! -d "./data/ToS3/GT" ]; then
    DATA_DIR="./data/ToS3"
    FID="1XoR_NVBR-LbZOA8fXh7d4oPV0M8fRi8a"
    MD5="56eb9e8298a4e955d618c1658dfc89c9"
    
    echo ">>> Start to download [ToS3 GT] dataset"
    
	mkdir -p ${DATA_DIR}
	download_large_file ${DATA_DIR} ${FID} GT
	check_md5 ${DATA_DIR}/GT.zip ${MD5}
	unzip ${DATA_DIR}/GT.zip -d ${DATA_DIR} && rm ${DATA_DIR}/GT.zip
fi


if [ $1 == BD ]; then
	# Vid4 LR BD
	if [ ! -d "./data/Vid4/Gaussian4xLR" ]; then
	    DATA_DIR="./data/Vid4"
	    FID="1-5NFW6fEPUczmRqKHtBVyhn2Wge6j3ma"
	    MD5="3b525cb0f10286743c76950d9949a255"
	    
	    echo ">>> Start to download [Vid4 LR] dataset (BD degradation)"
	    
		download_small_file ${DATA_DIR} ${FID} Gaussian4xLR
		check_md5 ${DATA_DIR}/Gaussian4xLR.zip ${MD5}
		unzip ${DATA_DIR}/Gaussian4xLR.zip -d ${DATA_DIR} && rm ${DATA_DIR}/Gaussian4xLR.zip
	fi
	
	sleep 1s
	
	# ToS3 LR BD
	if [ ! -d "./data/ToS3/Gaussian4xLR" ]; then
	    DATA_DIR="./data/ToS3"
	    FID="1rDCe61kR-OykLyCo2Ornd2YgPnul2ffM"
	    MD5="803609a12453a267eb9c78b68e073e81"
	    
	    echo ">>> Start to download [ToS3 LR] dataset (BD degradation)"
	    
		download_large_file ${DATA_DIR} ${FID} Gaussian4xLR
		check_md5 ${DATA_DIR}/Gaussian4xLR.zip ${MD5}
		unzip ${DATA_DIR}/Gaussian4xLR.zip -d ${DATA_DIR} && rm ${DATA_DIR}/Gaussian4xLR.zip
	fi

elif [ $1 == BI ]; then
	# Vid4 LR BI
	if [ ! -d "./data/Vid4/Bicubic4xLR" ]; then
	    DATA_DIR="./data/Vid4"
	    FID="1Kg0VBgk1r9I1c4f5ZVZ4sbfqtVRYub91"
	    MD5="35666bd16ce582ae74fa935b3732ae1a"
	    
	    echo ">>> Start to download [Vid4 LR] dataset (BI degradation)"
	    
		download_small_file ${DATA_DIR} ${FID} Bicubic4xLR
		check_md5 ${DATA_DIR}/Bicubic4xLR.zip ${MD5}
		unzip ${DATA_DIR}/Bicubic4xLR.zip -d ${DATA_DIR} && rm ${DATA_DIR}/Bicubic4xLR.zip
	fi
	
	sleep 1s
	
	# ToS3 LR BI
	if [ ! -d "./data/ToS3/Bicubic4xLR" ]; then
	    DATA_DIR="./data/ToS3"
	    FID="1FNuC0jajEjH9ycqDkH4cZQ3_eUqjxzzf"
	    MD5="3b165ffc8819d695500cf565bf3a9ca2"
	    
	    echo ">>> Start to download [ToS3 LR] dataset (BI degradation)"
	    
		download_large_file ${DATA_DIR} ${FID} Bicubic4xLR
		check_md5 ${DATA_DIR}/Bicubic4xLR.zip ${MD5}
		unzip ${DATA_DIR}/Bicubic4xLR.zip -d ${DATA_DIR} && rm ${DATA_DIR}/Bicubic4xLR.zip
	fi

else
	echo Unknown Degradation Type: $1 \(Currently supported: \"BD\" or \"BI\"\)
fi
