# keyboard listener

train a model on the sound of your own keyboard, then unplug it and type with just the mic listening.

watch the video on it here: https://youtu.be/bLTFr5CAW_Q

## how it works

- **record mode** - keyboard stays plugged in so every press gets labeled automatically. it finds where the click starts and saves a short clip for that key
- **train** - each clip gets turned into 294 numbers (mfccs, the spectrum of the first hit, loudness, etc). every clip also gets 4 synthetic copies (shifted, louder/quieter, some noise) so you don't have to press every key 200 times. then 4 models (svm, logistic regression, knn, random forest) get cross validated and the best one gets saved
- **test mode** - unplug the keyboard, it listens for clicks and guesses the key. `--type` actually types it out

right now it does one key at a time: letters, numbers, space, enter, backspace and `. , ' ; -`. no shift so it just capitalizes the start of sentences

## setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## usage

```
python main.py meter                  # make sure the mic hears your keys
python main.py record --target 40     # ESC to stop, you can run it more than once
python main.py train
python main.py test --type
```

`python main.py stats` shows how many samples each key has. `python main.py devices` if its using the wrong mic, then add `--device N`

## tips

- don't move the mic between recording and testing, it messes everything up
- around 40 presses per key works well. if train says two keys keep getting mixed up, record more of those
- bad clip? just delete the wav in `data/samples/<key>/` and retrain

## todo

- tune the model settings
- dictionary/autocorrect using the top 3 guesses
- fast typing
