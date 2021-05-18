from datetime import datetime

import numpy as np
import pyedflib


def write(signal, annot_sym, annot_idx, output_path, annot_sign=None):

    n_deriv = signal.shape[0]

    # Build headers
    headers = []
    for i in range(n_deriv):
        tmp = {
            'label': 'ECG_{}'.format(i+1),
            'dimension': 'uV',
            'sample_rate': 250,
            'physical_max': 32767.0,
            'physical_min': -32768.0,
            'digital_max': 32767,
            'digital_min': -32768,
            'prefilter': '',
            'transducer': ''
        }
        headers.append(tmp)

    head = {
             'technician': '',
             'recording_additional': '',
             'patientname': 'X',
             'patient_additional': '',
             'patientcode': '',
             'equipment': '',
             'admincode': '',
             'gender': '',
             'startdate': datetime(2020, 9, 22, 9, 32, 35),
             'birthdate': '',
             'annotations': []
            }
    # Add annotations
    for i in range(len(annot_sym)):

        time_idx = np.asarray(annot_idx[i]/250)
        tmp = [time_idx, 1, annot_sym[i]]
        head['annotations'].append(tmp)

    if annot_sign is not None:
        new_signal = np.zeros((signal.shape[0]+1, signal.shape[1]))
        new_signal[0:signal.shape[0], :] = signal
        new_signal[signal.shape[0], :] = annot_sign * 500
        signal = new_signal
        tmp = {
            'label': 'R-annot-conf',
            'dimension': 'uV',
            'sample_rate': 250,
            'physical_max': 32767.0,
            'physical_min': -32768.0,
            'digital_max': 32767,
            'digital_min': -32768,
            'prefilter': '',
            'transducer': ''
        }
        headers.append(tmp)

    # Write the file
    try:
        pyedflib.highlevel.write_edf(output_path, signal, headers, head)
        print('{} successfully exported to EDF file format'.format(output_path))
    except:
        print('ERROR during file exportation')
