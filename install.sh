git clone https://github.com/akimi2259/dl_lecture_competition_pub.git
cd dl_lecture_competition
git checkout MEG-competition

gsutil cp gs://dl-common/2024/MEG/data-omni.zip ./data/
gsutil cp gs://dl-common/2024/MEG/images.zip ./data/
unzip data/data-omni.zip -d data/
unzip data/images.zip -d data/

