import sys, json
sys.stdout.reconfigure(encoding='utf-8')
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

datasets = {
    "awsaf49/asvpoof-2019-dataset": "ASVspoof2019",
    "abdallamohamed312/in-the-wild-audio-deepfake": "In-the-Wild",
    "walimuhammadahmad/fakeaudio": "WaveFake",
    "kkijjaa/asvspoof2021-la": "ASVspoof2021-LA",
    "serjkalinovskiy/asvspoof2021-df": "ASVspoof2021-DF",
    "riosgonzalo/musan-rirs": "MUSAN-RIRs",
    "nhattruongdev/rirs-noises": "RIRs-Noise",
}

for slug, name in datasets.items():
    print(f"\n{'='*60}")
    print(f"DATASET: {slug} ({name})")
    print(f"{'='*60}")
    try:
        files = api.dataset_list_files(slug).files
        print(f"  Total files returned: {len(files)}")
        
        # Show directory structure (top 2 levels)
        dirs = {}
        file_types = {}
        for f in files:
            parts = f.name.split('/')
            if len(parts) >= 2:
                top = parts[0]
                if top not in dirs:
                    dirs[top] = set()
                if len(parts) >= 3:
                    dirs[top].add(parts[1])
            
            ext = f.name.rsplit('.', 1)[-1] if '.' in f.name else 'none'
            file_types[ext] = file_types.get(ext, 0) + 1
        
        print(f"  File types: {file_types}")
        print(f"  Top-level dirs/files:")
        for d in sorted(dirs.keys()):
            subs = sorted(dirs[d])[:5]
            print(f"    {d}/ -> {subs}{'...' if len(dirs[d]) > 5 else ''}")
        
        # Check for protocol/metadata files
        proto_files = [f.name for f in files if f.name.endswith('.txt') or f.name.endswith('.csv') or 'README' in f.name or 'LICENSE' in f.name or 'ANNOTATION' in f.name]
        if proto_files:
            print(f"  Protocol/metadata files: {proto_files[:10]}")
        else:
            print(f"  WARNING: No protocol/metadata files found in first 20 files!")
            
    except Exception as e:
        print(f"  ERROR: {e}")
