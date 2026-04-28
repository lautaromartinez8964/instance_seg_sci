import argparse
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Watch checkpoint directory and export DT maps when target epochs appear.')
    parser.add_argument('config', help='Model config file.')
    parser.add_argument('--checkpoint-dir', required=True, help='Directory containing epoch checkpoints.')
    parser.add_argument('--epochs', nargs='+', type=int, required=True, help='Epoch numbers to watch.')
    parser.add_argument('--sample-manifest', required=True, help='Sample manifest json file.')
    parser.add_argument('--distance-root', required=True, help='Distance transform root.')
    parser.add_argument('--out-dir', required=True, help='Directory to store DT visualization exports.')
    parser.add_argument('--device', default='cuda:0', help='Inference device.')
    parser.add_argument('--poll-interval', type=int, default=60, help='Polling interval in seconds.')
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pending_epochs = sorted(set(args.epochs))
    exported_epochs = set()

    script_path = Path(__file__).with_name('export_gravel_dt_progress.py')

    while True:
        for epoch in pending_epochs:
            if epoch in exported_epochs:
                continue
            checkpoint_path = checkpoint_dir / f'epoch_{epoch}.pth'
            if not checkpoint_path.exists():
                continue

            cmd = [
                sys.executable,
                str(script_path),
                args.config,
                '--checkpoint-dir', str(checkpoint_dir),
                '--epochs', str(epoch),
                '--sample-manifest', args.sample_manifest,
                '--distance-root', args.distance_root,
                '--out-dir', str(out_dir),
                '--device', args.device,
            ]
            print(f'[watch] exporting epoch {epoch} from {checkpoint_path.as_posix()}', flush=True)
            subprocess.run(cmd, check=True)
            exported_epochs.add(epoch)
            print(f'[watch] finished epoch {epoch}', flush=True)

        if len(exported_epochs) == len(pending_epochs):
            print(f'[watch] all target epochs exported: {sorted(exported_epochs)}', flush=True)
            return

        time.sleep(max(args.poll_interval, 10))


if __name__ == '__main__':
    main()