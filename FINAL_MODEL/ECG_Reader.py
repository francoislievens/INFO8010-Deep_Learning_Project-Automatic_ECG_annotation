import wfdb
import pyedflib
import Utils


def read_file(path):


    # Identify MIT-file
    if '.hea' in path:
        return read_mit(path)

    if '.edf' in path:
        return read_edf(path)



def read_mit(path):

    file_path = path.replace('.hea', '')
    signal = wfdb.rdrecord(file_path)
    signal = signal.p_signal.T
    signal = Utils.filtering(signal, freq_adapter=True)
    signal *= 1000

    return signal

def read_edf(path):

    signal, _, _ = pyedflib.highlevel.read_edf(path)
    return signal

