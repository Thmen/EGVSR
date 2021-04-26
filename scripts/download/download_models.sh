#!/usr/bin/env bash

# This script is used to download pretrained models
# Usage:
#     cd <TecoGAN-PyTorch_dir>
#     bash ./scripts/download/download_models <degradation_mode> <model_name>


function download_small_file() {
    local FPATH=$1
    local FID=$2
    local MD5=$3
    
    wget --no-check-certificate -O ${FPATH} "https://drive.google.com/uc?export=download&id=${FID}"
    
    if [ ${MD5} != $(md5sum ${FPATH} | cut -d " " -f1) ]; then
        echo "!!! Fail to match MD5 sum for: ${FPATH}"
        echo "!!! Please try downloading it again"
        exit 1
    fi
}



if [ $1 == BD -a $2 == TecoGAN ]; then
	FPATH="./pretrained_models/TecoGAN_BD_iter500000.pth"
	if [ ! -f $FPATH ]; then
	    FID="13FPxKE6q7tuRrfhTE7GB040jBeURBj58"
	    MD5="13d826c9f066538aea9340e8d3387289"
	    
		echo "Start to download model [TecoGAN BD]"
		download_small_file ${FPATH} ${FID} ${MD5}
	fi

elif [ $1 == BD -a $2 == FRVSR ]; then
	FPATH="./pretrained_models/FRVSR_BD_iter400000.pth"
	if [ ! -f $FPATH ]; then
	    FID="11kPVS04a3B3k0SD-mKEpY_Q8WL7KrTIA"
	    MD5="77d33c58b5cbf1fc68a1887be80ed18f"
	    
		echo "Start to download model [FRVSR BD]"
		download_small_file ${FPATH} ${FID} ${MD5}
	fi

elif [ $1 == BI -a $2 == TecoGAN ]; then
	FPATH="./pretrained_models/TecoGAN_BI_iter500000.pth"
	if [ ! -f $FPATH ]; then
		FID="1ie1F7wJcO4mhNWK8nPX7F0LgOoPzCwEu"
	    MD5="4955b65b80f88456e94443d9d042d1e6"
	    
		echo "Start to download model [TecoGAN BI]"
		download_small_file ${FPATH} ${FID} ${MD5}
	fi

elif [ $1 == BI -a $2 == FRVSR ]; then
	FPATH="./pretrained_models/FRVSR_BI_iter400000.pth"
	if [ ! -f $FPATH ]; then
	    FID="1wejMAFwIBde_7sz-H7zwlOCbCvjt3G9L"
	    MD5="ad6337d934ec7ca72441082acd80c4ae"
	    
		echo "Start to download model [FRVSR BI]"
		download_small_file ${FPATH} ${FID} ${MD5}
	fi

else
	echo Unknown combination: $1, $2
fi
