# Conda Environment Setup


```bash
conda create -n new_diffusers python=3.10 pip jupyter jupyterlab
conda activate new_diffusers
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia

# Install diffusers from source code (preferrable for attend and excite)
# cd diffusers
# pip install -e ".[torch]"
# pip install transformers

# Install diffusers from remote 
#pip install --upgrade diffusers[torch]
pip install transformers==4.26 # higher than 4.27 is not compatabile with tifascore
pip install pyrallis
pip install opencv-python
pip install einops
pip install matplotlib pandas
pip install scipy
pip install scikit-learn
pip install git+https://github.com/openai/CLIP.git
pip install natsort
# For evaluation
pip install salesforce-lavis
pip install tifascore
```