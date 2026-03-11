# import libraries & packages
import pandas as pd
import numpy as np
import mne
from mne.decoding import CSP, get_spatial_filter_from_estimator

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

from alignment_methods import euclidean_alignment, riemannian_alignment

# preprocessing EEG data
def load_subject(subject_id, data_dir="./raw_data", use_all_channels=False):
    all_epochs = []
    
    # loop through only runs with Task 2 (imagined left/right fist)
    for run in [4, 8, 12]:
        # set vars
        subject_str = str(subject_id).zfill(3)
        run_str = str(run).zfill(2)
        edf_path = f"{data_dir}/S{subject_str}/S{subject_str}R{run_str}.edf"

        # load raw EDF data
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

        # keep all channels if feeding into EEGNet
        # only filter to 10 channels for CSP / LDA pipeline
        if not use_all_channels:
            # extract channels we want (central + frontal electrodes) for CSP / LDA
            raw.pick(['Fc3.', 'Fcz.', 'Fc4.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..'])
    
            # rename electrode channels to standard
            rename = {
                'Fc3.': 'FC3', 'Fcz.': 'FCz', 'Fc4.': 'FC4',
                'C5..': 'C5',  'C3..': 'C3',  'C1..': 'C1',
                'Cz..': 'Cz',  'C2..': 'C2',  'C4..': 'C4',
                'C6..': 'C6'
            }
            raw.rename_channels(rename)
            
            # set standard montage manually
            montage = mne.channels.make_standard_montage('standard_1005')
            raw.set_montage(montage, match_case=False, on_missing='ignore')
            
        # bandpass filter 8-30 Hz to capture motor imagery (mu + beta rhythms)
        # IIR filters forward-backward via filtfilt()
        raw.filter(8, 30, method='iir', verbose=False)
        
        # extract event times + labels from annotations
        events, event_ids = mne.events_from_annotations(raw, verbose=False)
        
        # epoch around T1 and T2 (ignore T0, which is rest)
        epochs = mne.Epochs(raw, events,
                            event_id={'T1': 2, 'T2': 3},
                            tmin=0, tmax=2,
                            baseline=None, preload=True, verbose=False)
        
        # collect this run's epochs
        all_epochs.append(epochs)

    # concatenate epochs from all 3 runs (4, 8, 12) into single object
    combined = mne.concatenate_epochs(all_epochs, verbose=False)

    # extract EEG data as np array
    X = combined.get_data()

    # extract labels: 1 if T1 (left fist), 0 if T2 (right fist); event codes: 2 -> T1, 3 -> T2
    y = (combined.events[:, -1] == 2).astype(int)
    
    return X, y, combined.info

if __name__ == "__main__":
    # INITIALIZE ARGUMENTS/VARS FROM CALL
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_subjects", type=int, default=50)
    parser.add_argument("--held_out_subject", type=int, default=None)
    parser.add_argument("--csp_components", type=int, default=4)
    parser.add_argument("--alignment", type=str, default=None)
    parser.add_argument("--mode", type=str, default="csp",
                        choices=["csp", "raw"],
                        help="csp: extract CSP features for LDA // raw: return raw epochs for EEGNet") 
    args = parser.parse_args()

    # set vars
    n_subjects = args.n_subjects
    held_out_subject = args.held_out_subject if args.held_out_subject is not None else n_subjects
    csp_components = args.csp_components
    use_all_channels = (args.mode == 'raw')
    
    alignment_methods = {
        "euclidean": euclidean_alignment,
        "riemannian": riemannian_alignment,
    }
    if args.alignment is not None and args.alignment not in alignment_methods:
        raise ValueError(f"unknown alignment method! alignment methods: {list(alignment_methods.keys())}")
    alignment_function = alignment_methods.get(args.alignment)


    # LOADING DATA
    all_X = {}
    all_y = {}

    for subject_id in range(1, n_subjects + 1):
        # apply preproc function to all subjects
        X, y, info = load_subject(subject_id, use_all_channels=use_all_channels)
        all_X[subject_id] = X
        all_y[subject_id] = y

        # save once
        if subject_id == 1:
            eeg_info = info
        
        # set vars
        subject_str = str(subject_id).zfill(3)
        left_count = sum(y == 1)
        right_count = sum(y == 0)
    
        # display shape of data + # epochs of left/right fist
        print(f"Subject {subject_str}: {X.shape}, left fist: {left_count}, right fist: {right_count}")


    # APPLYING ALIGNMENT IF NEEDED
    if alignment_function is not None:
        for subject_id in range(1, n_subjects + 1):
            all_X[subject_id] = alignment_function(all_X[subject_id])

    # fit CSP on train subjects (1-49) only
    if args.mode == 'csp':
        # UPDATED HERE: hold out specified subject ID and not just last subject by default
        train_X = np.concatenate([all_X[i] for i in range(1, n_subjects + 1) if i != held_out_subject], axis=0)
        train_y = np.concatenate([all_y[i] for i in range(1, n_subjects + 1) if i != held_out_subject], axis=0)
        
        # fit CSP
        csp = CSP(n_components=csp_components, log=True)
        csp.fit(train_X, train_y)
    
        os.makedirs("plots", exist_ok=True)
        fig = get_spatial_filter_from_estimator(csp, info=eeg_info).plot_filters(components=list(range(csp_components)), show=False)
        fig.savefig("plots/csp_filters.png", dpi=150, bbox_inches='tight')
    
    
        # SAVING DATA
        os.makedirs("processed_data", exist_ok=True)
        
        # save per-subject CSP features    
        for subject_id in range(1, n_subjects + 1):
            # apply CSP to all subjects
            X_csp = csp.transform(all_X[subject_id])
            y = all_y[subject_id]
            
            # combine X and y into one df
            df = pd.DataFrame(X_csp, columns=[f"CSP_{i+1}" for i in range(csp_components)])
            df['label'] = y
            df.to_csv("processed_data/S" + str(subject_id).zfill(3) + ".csv", index=False)
        
        # save combined train data (subjects 1-49)
        train_X_csp = csp.transform(train_X)
        df_train = pd.DataFrame(train_X_csp, columns=[f"CSP_{i+1}" for i in range(csp_components)])
        df_train['label'] = train_y
        df_train.to_csv("processed_data/train.csv", index=False)
        
        # save held out test set (subject 50)
        test_X_csp = csp.transform(all_X[held_out_subject])
        df_test = pd.DataFrame(test_X_csp, columns=[f"CSP_{i+1}" for i in range(csp_components)])
        df_test['label'] = all_y[held_out_subject]
        df_test.to_csv("processed_data/test.csv", index=False)

    else:
        os.makedirs("processed_data/eegnet", exist_ok=True)
        for subject_id in range(1, n_subjects + 1):
            np.save(f"processed_data/eegnet/S{str(subject_id).zfill(3)}_X.npy",all_X[subject_id])
            np.save(f"processed_data/eegnet/S{str(subject_id).zfill(3)}_y.npy",all_y[subject_id])
        print(f"Saved raw epochs to processed_data/eegnet/")
        
