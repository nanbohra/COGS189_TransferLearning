# EEG Motor Imagery Transfer Learning
This project investigates how well EEG-based motor imagery (MI) classifiers trained on a group of subjects generalizes to an unseen subject, and explores whether signal alignment techniques can improve generalization across subjects.


## Dataset
Dataset Link: [PhysioNet EEG Motor Movement/Imagery Database](https://physionet.org/content/eegmmidb/)

* 64-channel EEG from 109 subjects performing imagined left/right fist movements
* Used 50 subjects: subjects 1-49 for training, subject 50 as held-out test
* Extracted runs 4, 8, and 12 (correspond to imagined left/right fist task)


## Preprocessing Steps
File: `preprocess.py`

* Extracting central and frontal electrode channels: FC3, FCz, FC4, C5, C3, C1, Cz, C2, C4, C6
* Bandpass filtering; 8-30 Hz (mu & beta rhythms, IIR)
* Epoching: T1 (left fist) & T2 (right fist), 0-2s post-stimulus
* Signal Alignment: optional, applied to each subject before CSP (see below for more info)
* CSP: fit on combined training subjects, then applied to all subjects
* **Outputs**: CSVs for each subject in `processed_data/`, along with `train.csv` and `test.csv`


## Alignment Methods
File: `alignment_methods.py`

* Euclidean Alignment: normalizes each subject's epoch covariance to a common reference by applying the inverse square root of the mean covariance matrix
* Riemannian Alignment: TODO
* Shared Response Model: TODO


## How to Run
### Install Libraries + Dependencies
```
pip install -r requirements.txt
```
### Organizing Raw Data
Place raw EDF files in `./raw_data/` with this structure:
```
raw_data/
├── S001/
│   ├── S001R04.edf
│   ├── S001R08.edf
│   └── S001R12.edf
├── S002/
│   ├── S002R04.edf
│   ├── S002R08.edf
│   └── S002R12.edf
 ...
```

### Preprocess Data
For default arguments:
```
python preprocess.py
```
For custom arguments, follow this format:
```
python preprocess.py --n_subjects 50 --held_out_subject 50 --csp_components 4 --alignment euclidean
```
