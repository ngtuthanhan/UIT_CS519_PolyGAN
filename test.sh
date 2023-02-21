#!/bin/bash
for file in `ls ./data/train/image`
do
    
    python test.py --stage Stage3 --datamode train --model_image $file --reference_image ${file:0:7}1.jpg
done