

conda create -n ditra python=3.12

for whisper-offiline-diarization. Install this way because nemo installation in requirements txt installs pytorch 2.7 with normal command "pip3 install torch torchvision torchaudio
". it must be avoided:

conda install nvidia/label/cuda-12.9.0::cuda-toolkit
pip3 install torch torchvision torchaudio

pip install cython


git clone https://github.com/MahmoudAshraf97/whisper-diarization.git
cd whisper-diarization/
pip install -c constraints.txt -r requirements.txt


can ignore this cudnn installation

conda info --envs
example:
ctb                  * /home/thulasi/.conda/envs/ctb

install tarball from 
https://developer.nvidia.com/cudnn-archive

tarball installation:
https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-890/install-guide/index.html

wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.10.1.4_cuda12-archive.tar.xz

tar -xvf cudnn-linux-x86_64-9.10.1.4_cuda12-archive.tar.xz

cp cudnn-*-archive/include/cudnn*.h /home/thulasi/.conda/envs/ctb/include/
cp -P cudnn-*-archive/lib/libcudnn* /home/thulasi/.conda/envs/ctb/lib/
chmod a+r /home/thulasi/.conda/envs/ctb/include/cudnn*.h /home/thulasi/.conda/envs/ctb/lib/libcudnn*

export LD_LIBRARY_PATH=/home/thulasi/.conda/envs/ctb/lib:$LD_LIBRARY_PATH



for realtme whisper:

pip install -U openai-whisper


pip install "fastapi[standard]"
pip install pydub
pip install google-genai


pytorch 2.7 causes error in whisper. downgrade to 2.5:

conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia

check compatible cudnn version with 
python -c "import torch; print('cuDNN version:', torch.backends.cudnn.version())"
and isntall it from 
https://developer.nvidia.com/cudnn-archive

wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.1.0.70_cuda12-archive.tar.xz

tar -xf cudnn-linux-x86_64-9.1.0.70_cuda12-archive.tar.xz

https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-890/install-guide/index.html



cp cudnn-*-archive/include/cudnn*.h /home/thulasi/.conda/envs/ctb/include/
cp -P cudnn-*-archive/lib/libcudnn* /home/thulasi/.conda/envs/ctb/lib/
chmod a+r /home/thulasi/.conda/envs/ctb/include/cudnn*.h /home/thulasi/.conda/envs/ctb/lib/libcudnn*

export LD_LIBRARY_PATH=/home/thulasi/.conda/envs/ctb/lib:$LD_LIBRARY_PATH


my laptop

cp cudnn-*-archive/include/cudnn*.h /home/shamila/anaconda3/envs/ditra/include/
cp -P cudnn-*-archive/lib/libcudnn* /home/shamila/anaconda3/envs/ditra/lib/
chmod a+r /home/shamila/anaconda3/envs/ditra/include/cudnn*.h /home/shamila/anaconda3/envs/ditra/lib/libcudnn*


export LD_LIBRARY_PATH=/home/shamila/anaconda3/envs/ditra/lib:$LD_LIBRARY_PATH




pip uninstall -y numpy
pip install "numpy<2.0"


try this in whisper-diarization repo
python diarize.py -a audio.wav
python diarize.py -a received_audio.wav

whisper audio.wav --model tiny

in case processes get frozen
lsof -i :7007
lsof -i :8000
lsof -i :9000

ps aux | grep 8000

kill 97563


mkdir -p ~/ngrok
tar -xvzf ngrok-v3-stable-linux-amd64.tgz -C ~/ngrok
export PATH="$HOME/ngrok:$PATH" 
source ~/.bashrc 
