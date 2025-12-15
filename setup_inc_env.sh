rm -rf /tmp/neural-compressor
cd /tmp
git clone https://github.com/yangulei/neural-compressor.git
cd neural-compressor
git checkout -b fsdpa origin/fsdpa
export INC_PT_ONLY=1
pip uninstall -y neural_compressor_pt
python setup.py install
