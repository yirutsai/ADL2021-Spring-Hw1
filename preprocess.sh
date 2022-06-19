if [ ! -f glove.840B.300d.txt ]; then
  wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O glove.840B.300d.zip
  unzip glove.840B.300d.zip
fi
python preprocess_intent.py --vocab_size 3000
python preprocess_slot.py --vocab_size 2500
