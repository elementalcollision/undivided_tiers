#!/usr/bin/env python3

import argparse
import logging
import os
import sys

from vua.core import VUA, VUAConfig

def main():
    parser = argparse.ArgumentParser(description="Repair missing or broken symlinks in a VUA cache directory.")
    parser.add_argument("cache_dir", help="Path to the VUA cache directory (root of hash directories)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set logging level")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format='%(asctime)s [%(levelname)s] %(message)s')

    if not os.path.isdir(args.cache_dir):
        print(f"Error: {args.cache_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    vua = VUA(VUAConfig, args.cache_dir)
    vua.repair_symlinks()
    print("Symlink repair complete.")

if __name__ == "__main__":
    main() 