import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Extract only weights from a checkpoint to avoid PyTorch 2.6 weights_only=True issues')
    parser.add_argument('in_file', help='input checkpoint file')
    parser.add_argument('out_file', help='output checkpoint file with only weights')
    return parser.args()

if __name__ == '__main__':
    # Patch torch load to bypass exactly as in test.py
    _original_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_load(*args, **kwargs)
    torch.load = _patched_load

    import sys
    if len(sys.argv) != 3:
        print("Usage: python tools/extract_weights.py <in_file> <out_file>")
        sys.exit(1)

    in_file = sys.argv[1]
    out_file = sys.argv[2]
    
    print(f"Loading {in_file}...")
    checkpoint = torch.load(in_file, map_location='cpu')
    
    clean_ckpt = {}
    clean_ckpt['state_dict'] = checkpoint.get('state_dict', checkpoint)
    if 'meta' in checkpoint:
        clean_ckpt['meta'] = checkpoint['meta']
        
    print(f"Saving cleaned weights to {out_file}...")
    torch.save(clean_ckpt, out_file)
    print("Done!")
