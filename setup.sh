#!/bin/bash
mkdir gemma3-270m output

pip install ms-swift

git clone https://github.com/Wiscarce/gemma3-logo-homework.git

git clone \
https://github.com/roboticcam/logo-detailed-prompt

modelscope download \
--model google/gemma-3-270m-it \
--local_dir ./gemma3-270m

echo "Done!"