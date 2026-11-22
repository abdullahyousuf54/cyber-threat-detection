from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd

FEATURE_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERIC = [c for c in FEATURE_NAMES if c not in CATEGORICAL]

# NSL-KDD attack names grouped into the standard five project classes.
DOS = {
    "back", "land", "neptune", "pod", "smurf", "teardrop", "apache2",
    "udpstorm", "processtable", "mailbomb"
}
PROBE = {"ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"}
R2L = {
    "ftp_write", "guess_passwd", "imap", "multihop", "phf", "warezmaster",
    "warezclient", "spy", "named", "sendmail", "xlock", "xsnoop",
    "snmpgetattack", "snmpguess", "worm", "httptunnel"
}
U2R = {
    "buffer_overflow", "loadmodule", "perl", "rootkit", "ps", "sqlattack",
    "xterm"
}

TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
TEST_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

def download_dataset(data_dir="data/nsl_kdd"):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "KDDTrain+.txt"
    test_path = data_dir / "KDDTest+.txt"

    if not train_path.exists():
        print("Downloading KDDTrain+.txt...")
        urlretrieve(TRAIN_URL, train_path)

    if not test_path.exists():
        print("Downloading KDDTest+.txt...")
        urlretrieve(TEST_URL, test_path)

    return train_path, test_path

def load_nsl_file(path):
    columns = FEATURE_NAMES + ["label", "difficulty"]
    df = pd.read_csv(path, header=None, names=columns)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    return df

def map_attack_family(label):
    if label == "normal":
        return "Normal"
    if label in DOS:
        return "DoS"
    if label in PROBE:
        return "Probe"
    if label in R2L:
        return "R2L"
    if label in U2R:
        return "U2R"
    # Keep unknown future/rare labels as Other rather than silently calling them normal.
    return "Other"

def add_project_label(df):
    out = df.copy()
    out["attack_family"] = out["label"].map(map_attack_family)
    return out
