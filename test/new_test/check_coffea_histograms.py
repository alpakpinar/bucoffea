#!/usr/bin/env python

from coffea.util import load
import sys
from pprint import pprint

def main():
    # Input coffea file to be checked
    infile = sys.argv[1]

    acc = load(infile)
    pprint(acc.keys())

if __name__ == '__main__':
    main()