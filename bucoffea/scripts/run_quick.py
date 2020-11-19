#!/usr/bin/env python

from bucoffea.helpers.dataset import extract_year
from bucoffea.processor.executor import run_uproot_job_nanoaod
from bucoffea.helpers.cutflow import print_cutflow
from coffea.util import save
import coffea.processor as processor
import argparse

def parse_commandline():

    parser = argparse.ArgumentParser()
    parser.add_argument('processor', type=str, help='The processor to be run. (monojet or vbfhinv)')
    args = parser.parse_args()

    return args

def main():

    fileset = {
        "DYJetsToLL_M-50_HT-100to200-MLM_ext1_2017" : [
            "root://cmsxrootd.fnal.gov//store/user/aakpinar/nanopost/25Aug20_splitJEC_v2/DYJetsToLL_M-50_HT-100to200_TuneCP5_13TeV-madgraphMLM-pythia8/DYJetsToLL_M-50_HT-100to200-MLM_ext1_2017/201026_231539/0000/tree_1.root"
        ]
    }

    years = list(set(map(extract_year, fileset.keys())))
    assert(len(years)==1)

    args = parse_commandline()
    processor_class = args.processor

    if processor_class == 'monojet':
        from bucoffea.monojet import monojetProcessor
        processorInstance = monojetProcessor(years[0])
    elif processor_class == 'vbfhinv':
        from bucoffea.vbfhinv import vbfhinvProcessor
        processorInstance = vbfhinvProcessor(years[0])
    elif processor_class == 'lhe':
        from bucoffea.gen.lheVProcessor import lheVProcessor
        processorInstance = lheVProcessor()
    elif args.processor == 'purity':
        from bucoffea.photon_purity import photonPurityProcessor
        processorInstance = photonPurityProcessor()
    elif args.processor == 'sumw':
        from bucoffea.gen import mcSumwProcessor
        processorInstance = mcSumwProcessor()
    elif args.processor == 'zmumu':
        from bucoffea.zmumu.zmumuProcessor import zmumuProcessor
        processorInstance = zmumuProcessor()

    for dataset, filelist in fileset.items():
        newlist = []
        for file in filelist:
            if file.startswith("/store/"):
                newlist.append("root://cms-xrd-global.cern.ch//" + file)
            else: newlist.append(file)
        fileset[dataset] = newlist

    for dataset, filelist in fileset.items():
        tmp = {dataset:filelist}
        output = run_uproot_job_nanoaod(tmp,
                                    treename='Runs' if args.processor=='sumw' else 'Events',
                                    processor_instance=processorInstance,
                                    executor=processor.futures_executor,
                                    executor_args={'workers': 4, 'flatten': True},
                                    chunksize=500000,
                                    )
        save(output, f"{processor_class}_{dataset}.coffea")
        # Debugging / testing output
        # debug_plot_output(output)
        print_cutflow(output, outfile=f'{processor_class}_cutflow_{dataset}.txt')

if __name__ == "__main__":
    main()
