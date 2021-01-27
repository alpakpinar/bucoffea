import copy
import coffea.processor as processor
import re
import numpy as np

from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray

from bucoffea.helpers import object_overlap, weight_shape
from bucoffea.helpers.dataset import extract_year, is_data
from bucoffea.helpers.gen import setup_lhe_cleaned_genjets

Hist = hist.Hist
Bin = hist.Bin
Cat = hist.Cat

def trigger_selection(selection, df):
    '''HLT_PFJet40 requirement.'''
    trigger='HLT_PFJet40'
    selection.add('jet_trig', df[trigger])
    return selection

def jetht_accumulator():
    dataset_ax = Cat("dataset", "Primary dataset")
    region_ax = Cat("region", "Selection region")

    ht_ax = Bin("ht", r"$H_{T}$ (GeV)", 50, 0, 5000)
    htmiss_ax = Bin("htmiss", r"$H_{T}^{miss}$ (GeV)", 250, 0, 1000)

    jet_pt_ax = Bin("jetpt", r"$p_{T}$ (GeV)", 50, 0, 1000)
    jet_eta_ax = Bin("jeteta", r"$\eta$", 50, -5, 5)
    jet_eta_ax_coarse = Bin("jeteta", r"$\eta$", 20, -5, 5)
    jet_phi_ax = Bin("jetphi", r"$\phi$", 50,-np.pi, np.pi)

    items = {}
    items["htmiss"] = Hist("Counts", dataset_ax, region_ax, htmiss_ax)
    items["ht"] = Hist("Counts", dataset_ax, region_ax, ht_ax)

    items["ak4_pt"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)
    items["ak4_eta"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax)
    items["ak4_phi"] = Hist("Counts", dataset_ax, region_ax, jet_phi_ax)

    items["ak4_pt0"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)
    items["ak4_eta0"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax)
    items["ak4_phi0"] = Hist("Counts", dataset_ax, region_ax, jet_phi_ax)
    items["ak4_pt1"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)
    items["ak4_eta1"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax)
    items["ak4_phi1"] = Hist("Counts", dataset_ax, region_ax, jet_phi_ax)

    # 2D histograms
    items["htmiss_ht"] = Hist("Counts", dataset_ax, region_ax, htmiss_ax, ht_ax)
    items["ak4_eta_phi"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax_coarse, jet_phi_ax)

    return processor.dict_accumulator(items)

def jetht_regions():
    regions = {}
    regions['inclusive'] = ['inclusive']
    regions['trig_pass'] = ['inclusive', 'jet_trig']
    regions['high_htmiss_loose'] = ['inclusive', 'jet_trig', 'high_htmiss_loose']
    regions['high_htmiss_tight'] = ['inclusive', 'jet_trig', 'high_htmiss_tight']

    return regions

def setup_jets(df):
    if extract_year(df['dataset']) != 2018:
        # 2016, 2017 data
        jes_suffix = ''
    else:
        # 2018 data
        jes_suffix = '_nom'

    ak4 = JaggedCandidateArray.candidatesfromcounts(
        df['nJet'],
        pt=df[f'Jet_pt{jes_suffix}'],
        eta=df['Jet_eta'],
        abseta=np.abs(df['Jet_eta']),
        phi=df['Jet_phi'],
        mass=np.zeros_like(df['Jet_pt']),
        looseId=(df['Jet_jetId']&2) == 2, # bitmask: 1 = loose, 2 = tight, 3 = tight + lep veto
        tightId=(df['Jet_jetId']&2) == 2, # bitmask: 1 = loose, 2 = tight, 3 = tight + lep veto
    )

    return ak4

class jethtProcessor(processor.ProcessorABC):
    def __init__(self, blind=False):
        self._accumulator = jetht_accumulator()

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        if not df.size:
            return self.accumulator.identity()
        dataset = df['dataset']
        df['is_data'] = is_data(dataset)
        
        ak4 = setup_jets(df)

        # Calculate HT and HTmiss
        htmiss = ak4[ak4.pt>30].p4.sum().pt
        ht = ak4[ak4.pt>30].pt.sum()
        
        selection = processor.PackedSelection()
        pass_all = np.zeros(df.size) == 0
        selection.add('inclusive', pass_all)

        selection = trigger_selection(selection, df)

        selection.add('high_htmiss_loose', htmiss>100)
        selection.add('high_htmiss_tight', htmiss>200)

        # Fill histograms
        output = self.accumulator.identity()

        regions = jetht_regions()
        for region, cuts in regions.items():
            mask = selection.all(*cuts)

            def ezfill(name, **kwargs):
                """Helper function to make filling easier."""
                output[name].fill(
                                  dataset=dataset,
                                  region=region,
                                  **kwargs
                                  )

            ezfill('htmiss', htmiss=htmiss[mask])
            ezfill('ht', ht=ht[mask])

            ezfill('ak4_pt',     jetpt=ak4[mask].pt.flatten())
            ezfill('ak4_eta',    jeteta=ak4[mask].eta.flatten())
            ezfill('ak4_phi',    jetphi=ak4[mask].phi.flatten())

            ezfill('ak4_eta_phi',  jeteta=ak4[mask].eta.flatten(),   jetphi=ak4[mask].phi.flatten())
            ezfill('htmiss_ht', htmiss=htmiss[mask], ht=ht[mask])

        return output

    def postprocess(self, accumulator):
        return accumulator

