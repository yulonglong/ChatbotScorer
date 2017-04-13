# THEANO_FLAGS="device=gpu0,mode=FAST_RUN,floatX=float32" python main.py -o output-test -tr data/Joker.xml -e 100
python main.py \
-o output \
-tr data/TickTock.xml \
-lt mean -t svm \
