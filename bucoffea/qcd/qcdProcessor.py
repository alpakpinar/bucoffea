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

def qcd_accumulator():
    dataset_ax = Cat("dataset", "Primary dataset")
    region_ax = Cat("region", "Selection region")

    met_ax = Bin("met", r"$p_{T}^{miss}$ (GeV)", 75, 0, 1500)
    ht_ax = Bin("ht", r"$H_{T}$ (GeV)", 50, 0, 4000)
    htmiss_ax = Bin("htmiss", r"$H_{T}^{miss}$ (GeV)", 75, 0, 1500)

    # HTmiss coarse binning for 2D histograms
    htmiss_coarse_binning = list(range(0,400,100)) + list(range(400,1400,200))
    htmiss_ax_coarse = Bin("htmiss", r"$H_{T}^{miss}$ (GeV)", htmiss_coarse_binning)

    jet_pt_ax = Bin("jetpt", r"$p_{T}$ (GeV)", 50, 0, 1000)
    jet_eta_ax = Bin("jeteta", r"$\eta$", 50, -5, 5)
    jet_eta_ax_coarse = Bin("jeteta", r"$\eta$", 10, -5, 5)
    jet_phi_ax = Bin("jetphi", r"$\phi$", 50,-np.pi, np.pi)

    met_phi_ax = Bin("metphi", r"MET $\phi$", 50,-np.pi, np.pi)

    items = {}

    items['sumw'] = processor.defaultdict_accumulator(float)
    items['sumw2'] = processor.defaultdict_accumulator(float)

    items["ak4_pt"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)
    items["ak4_eta"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax)
    items["ak4_phi"] = Hist("Counts", dataset_ax, region_ax, jet_phi_ax)

    items["ak4_pt0"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)
    items["ak4_eta0"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax)
    items["ak4_phi0"] = Hist("Counts", dataset_ax, region_ax, jet_phi_ax)
    
    items["ak4_pt1"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)
    items["ak4_eta1"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax)
    items["ak4_phi1"] = Hist("Counts", dataset_ax, region_ax, jet_phi_ax)

    items["genmet_pt"] = Hist("Counts", dataset_ax, region_ax, met_ax)
    items["genmet_phi"] = Hist("Counts", dataset_ax, region_ax, met_phi_ax)
    
    items["gen_ht"] = Hist('Counts', dataset_ax, region_ax, ht_ax)
    items["gen_htmiss"] = Hist('Counts', dataset_ax, region_ax, htmiss_ax)

    # 2D histograms
    items["htmiss_ht"] = Hist("Counts", dataset_ax, region_ax, htmiss_ax_coarse, ht_ax)
    items["htmiss_met"] = Hist("Counts", dataset_ax, region_ax, htmiss_ax_coarse, met_ax)

    return processor.dict_accumulator(items)

def qcd_regions():
    regions = {}
    regions['inclusive'] = ['inclusive']

    # Regions based on the eta of leading dijet
    # For now: Two categories
    # 1. Trk-trk events
    # 2. Others
    regions['trk_trk'] = ['inclusive', 'trk_trk']
    regions['not_trk_trk'] = ['inclusive', 'not_trk_trk']

    return regions

class qcdProcessor(processor.ProcessorABC):
    def __init__(self, blind=False):
        self._accumulator = qcd_accumulator()

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        if not df.size:
            return self.accumulator.identity()
        dataset = df['dataset']
        df['is_data'] = is_data(dataset)

        # Setup gen jets
        genjets = setup_lhe_cleaned_genjets(df)

        # Calculate some relevant quatities (i.e. HT, HTmiss)
        mht_p4 = genjets[genjets.pt>30].p4.sum()
        htmiss = mht_p4.pt

        ht = genjets[genjets.pt>30].pt.sum()

        # GenMET pt and phi
        genmet_pt = df['GenMET_pt']
        genmet_phi = df['GenMET_phi']

        # Leading two gen-jets
        digenjet = genjets[:,:2].distincts()

        # Categorize based on the eta of leading dijet
        ak4_eta0 = digenjet.i0.eta
        ak4_eta1 = digenjet.i1.eta

        selection = processor.PackedSelection()
        pass_all = np.ones(df.size)==1
        selection.add('inclusive', pass_all)

        trk_trk = ((ak4_eta0 < 2.5) & (ak4_eta1 < 2.5)).any()
        selection.add('trk_trk', trk_trk)
        selection.add('not_trk_trk', ~trk_trk)

        # Fill histograms
        output = self.accumulator.identity()

        output['sumw'][dataset] +=  df['genEventSumw']
        output['sumw2'][dataset] +=  df['genEventSumw2']

        regions = qcd_regions()

        for region, cuts in regions.items():
            mask = selection.all(*cuts)
            
            def ezfill(name, **kwargs):
                """Helper function to make filling easier."""
                output[name].fill(
                                    dataset=dataset,
                                    region=region,
                                    **kwargs
                                    )

            
            w_alljets = weight_shape(genjets[mask].pt, df['Generator_weight'][mask])
            ezfill('ak4_pt',  jetpt=genjets[mask].pt.flatten(),  weight=w_alljets)
            ezfill('ak4_eta', jeteta=genjets[mask].eta.flatten(), weight=w_alljets)
            ezfill('ak4_phi', jetphi=genjets[mask].phi.flatten(), weight=w_alljets)

            w_one_per_event = df['Generator_weight'][mask]

            ezfill('genmet_pt',   met=genmet_pt[mask],      weight=w_one_per_event)
            ezfill('genmet_phi',  metphi=genmet_phi[mask],  weight=w_one_per_event)
            ezfill('gen_ht',      ht=ht[mask],           weight=w_one_per_event)
            ezfill('gen_htmiss',  htmiss=htmiss[mask],   weight=w_one_per_event)

            # 2D histograms
            ezfill('htmiss_ht',   htmiss=htmiss[mask],  ht=ht[mask],         weight=w_one_per_event)
            ezfill('htmiss_met',  htmiss=htmiss[mask],  met=genmet_pt[mask], weight=w_one_per_event)

        return output

    def postprocess(self, accumulator):
        return accumulator
