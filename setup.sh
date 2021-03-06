# Assuming Nvidia and CUDA is installed (May require older version due to older theano and keras)
# Note that it is tested with CUDA 8 and nvidia driver 384.111
# sudo apt-get install graphviz

pip install theano==0.8.2 --user
pip install keras==1.1.1 --user
pip install scipy==0.19.0 --user
pip install scikit-learn==0.18.1 --user
pip install nltk==3.2.2 --user
pip install pydot==1.1.0 --user
pip install h5py==2.6.0 --user
pip install matplotlib==1.5.3 --user

cat > ~/.keras/keras.json << EOF
{
    "image_dim_ordering": "tf", 
    "epsilon": 1e-07, 
    "floatx": "float32", 
    "backend": "theano"
}
EOF

