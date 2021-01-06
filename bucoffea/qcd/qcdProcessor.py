import coffea.processor as processor
import re
import numpy as np

from dynaconf import settings as cfg
from bucoffea.helpers import dphi, weight_shape
from bucoffea.helpers.dataset import extract_year
from coffea.analysis_objects import JaggedCandidateArray
from coffea import hist

Hist = hist.Hist
Bin = hist.Bin
Cat = hist.Cat

# List of triggers we are considering
trigger_list = [
    'HLT_AK4PFJet30',
    'HLT_AK4PFJet50',
    'HLT_AK4PFJet80',
    'HLT_AK4PFJet100',
    'HLT_AK4PFJet120',
]

def qcd_regions():
    '''In each region, require a specific jet trigger.'''
    regions = {}

    for trigger in trigger_list:
        regions[f'r_{trigger}'] = ['inclusive', trigger]

    return regions

def setup_candidates_for_qcd(df):
    if extract_year(df['dataset']) != 2018:
        # 2016, 2017 data
        jes_suffix = ''
        jes_suffix_met = ''
    else:
        # 2018 data
        jes_suffix = '_nom'
        jes_suffix_met = '_nom'
    
    # Read AK4 jet candidates
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

    # Read MET pt and phi values
    if extract_year(df['dataset']) == 2017:
        met_branch = 'METFixEE2017'
    else:
        met_branch = 'MET'

    met_pt = df[f'{met_branch}_pt{jes_suffix_met}']
    met_phi = df[f'{met_branch}_phi{jes_suffix_met}']

    return met_pt, met_phi, ak4

class qcdProcessor(processor.ProcessorABC):
    def __init__(self):
        # Histogram setup
        dataset_ax = Cat("dataset", "Primary dataset")
        region_ax = Cat("region", "Selection region")

        mjj_ax = Bin("mjj",r"$m(jj)$ (GeV)", 75, 0, 7500)
        deta_ax = Bin("deta", r"$\Delta\eta_{jj}$", 50, 0, 10)
        dphi_ax = Bin("dphi", r"$\Delta\phi_{jj}$", 50, 0, 3.5)
        
        jet_pt_ax = Bin("jetpt",r"Jet $p_{T}$ (GeV)", 50, 0, 2000)
        jet_eta_ax = Bin("jeteta", r"Jet $\eta$", 50, -5, 5)
        jet_phi_ax = Bin("jetphi", r"Jet $\phi$", 50,-np.pi, np.pi)
        multiplicity_ax = Bin("multiplicity", r"$N_{jet}$", 10, -0.5, 9.5)
        
        met_ax = Bin("met", r"$p_{T}^{miss}$ (GeV)", 200, 0, 2000)
        phi_ax = Bin("phi", r"$\phi$", 50,-np.pi, np.pi)

        items = {}
        items["ak4_mult"] = Hist("Counts", dataset_ax, region_ax, multiplicity_ax)
        
        items["ak4_pt"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)
        items["ak4_eta"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax)
        items["ak4_phi"] = Hist("Counts", dataset_ax, region_ax, jet_phi_ax)
        items["ak4_pt0"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)
        items["ak4_eta0"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax)
        items["ak4_phi0"] = Hist("Counts", dataset_ax, region_ax, jet_phi_ax)
        items["ak4_pt1"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)
        items["ak4_eta1"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax)
        items["ak4_phi1"] = Hist("Counts", dataset_ax, region_ax, jet_phi_ax)

        items["mjj"] = Hist("Counts", dataset_ax, region_ax, mjj_ax)
        items["detajj"] = Hist("Counts", dataset_ax, region_ax, deta_ax)
        items["dphijj"] = Hist("Counts", dataset_ax, region_ax, dphi_ax)
        
        items["met"] = Hist("Counts", dataset_ax, region_ax, met_ax)
        items["met_phi"] = Hist("Counts", dataset_ax, region_ax, phi_ax)

        self._accumulator = processor.dict_accumulator(items)

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        output = self.accumulator.identity()
        dataset = df['dataset']

        met_pt, met_phi, ak4 = setup_candidates_for_qcd(df)

        # Dijets
        diak4 = ak4[:,:2].distincts()
        df['mjj'] = diak4.mass.max()
        df['detajj'] = np.abs(diak4.i0.eta - diak4.i1.eta).max()
        df['dphijj'] = dphi(diak4.i0.phi.min(), diak4.i1.phi.max())

        # Trigger selection
        selection = processor.PackedSelection()
        
        pass_all = np.ones(df.size)==1
        selection.add('inclusive', pass_all)

        for trigger in trigger_list:
            selection.add(trigger, df[trigger])

        regions = qcd_regions()

        for region, cuts in regions.items():
            # Fill output histograms
            weight = np.ones(df.size)
            mask = selection.all(*cuts)

            output['ak4_mult'].fill(
                dataset=dataset,
                region=region,
                multiplicity=ak4[mask].counts,
                weight=weight[mask]
            )
    
            def ezfill(name, **kwargs):
                """Helper function to make filling easier."""
                output[name].fill(
                                    dataset=dataset,
                                    region=region,
                                    **kwargs
                                    )
    
            # This is a workaround to create a weight array of the right dimension
            w_alljets = weight_shape(ak4[mask].eta, weight[mask])
    
            ezfill('ak4_pt',   jetpt=ak4[mask].pt.flatten(),     weight=w_alljets)
            ezfill('ak4_eta',  jeteta=ak4[mask].eta.flatten(),   weight=w_alljets)
            ezfill('ak4_phi',  jetphi=ak4[mask].phi.flatten(),   weight=w_alljets)
    
            # Leading ak4
            w_diak4 = weight_shape(diak4.pt[mask], weight[mask])
            ezfill('ak4_pt0',    jetpt=diak4.i0.pt[mask].flatten(),       weight=w_diak4)
            ezfill('ak4_eta0',   jeteta=diak4.i0.eta[mask].flatten(),     weight=w_diak4)
            ezfill('ak4_phi0',   jetphi=diak4.i0.phi[mask].flatten(),     weight=w_diak4)
    
            # Trailing ak4
            ezfill('ak4_pt1',    jetpt=diak4.i1.pt[mask].flatten(),       weight=w_diak4)
            ezfill('ak4_eta1',   jeteta=diak4.i1.eta[mask].flatten(),     weight=w_diak4)
            ezfill('ak4_phi1',   jetphi=diak4.i1.phi[mask].flatten(),     weight=w_diak4)
    
            # MET
            ezfill('met',        met=met_pt[mask],       weight=weight[mask])
            ezfill('met_phi',    phi=met_phi[mask],      weight=weight[mask])
    
            ezfill('dphijj',     dphi=df["dphijj"][mask],   weight=weight[mask])
            ezfill('detajj',     deta=df["detajj"][mask],   weight=weight[mask])
            ezfill('mjj',        mjj=df["mjj"][mask],       weight=weight[mask])

        return output

    def postprocess(self, accumulator):
        return accumulator