import copy
from coffea import hist

Hist = hist.Hist
Bin = hist.Bin
Cat = hist.Cat

from coffea.analysis_objects import JaggedCandidateArray
import coffea.processor as processor
from awkward import JaggedArray
import numpy as np
from bucoffea.helpers import object_overlap
from bucoffea.helpers.paths import bucoffea_path
from bucoffea.helpers.gen import find_first_parent
from bucoffea.monojet.definitions import accu_int, defaultdict_accumulator_of_empty_column_accumulator_float16, defaultdict_accumulator_of_empty_column_accumulator_int64,defaultdict_accumulator_of_empty_column_accumulator_bool
from pprint import pprint

def zmumu_accumulator(cfg):
    dataset_ax = Cat("dataset", "Primary dataset")
    region_ax = Cat("region", "Selection region")
    type_ax = Cat("type", "Type")

    met_ax = Bin("met", r"$p_{T}^{miss}$ (GeV)", 200, 0, 2000)
    recoil_ax = Bin("recoil", r"Recoil (GeV)", 200, 0, 2000)

    jet_pt_ax = Bin("jetpt", r"$p_{T}$ (GeV)", 100, 0, 1000)
    jet_eta_ax = Bin("jeteta", r"$\eta$", 50, -5, 5)
    jet_eta_ax_coarse = Bin("jeteta", r"$\eta$", 20, -5, 5)
    jet_phi_ax = Bin("jetphi", r"$\phi$", 50,-np.pi, np.pi)
    pt_ax = Bin("pt", r"$p_{T}$ (GeV)", 100, 0, 1000)
    eta_ax = Bin("eta", r"$\eta$", 50, -5, 5)
    phi_ax = Bin("phi", r"$\phi$", 50,-np.pi, np.pi)

    dphi_ax = Bin("dphi", r"$\Delta\phi$", 50, 0, 3.5)
    deta_ax = Bin("deta", r"$\Delta\eta$", 50, 0, 10)
    dilepton_mass_ax = Bin("dilepton_mass", r"$M(\ell\ell)$ (GeV)", 100,50,150)
    ptfrac_ax = Bin('ptfrac',r'$p_T$ fraction', 50, -0.5, 0.5)
    frac_ax = Bin('frac','Fraction', 50, 0, 1)
    
    weight_type_ax = Cat("weight_type", "Weight type")
    weight_ax = Bin("weight_value", "Weight",100,0.5,1.5)

    items = {}
    items["met"] = Hist("Counts", dataset_ax, region_ax, met_ax)
    items["met_phi"] = Hist("Counts", dataset_ax, region_ax, phi_ax)
    items["recoil"] = Hist("Counts", dataset_ax, region_ax, recoil_ax)
    items["recoil_phi"] = Hist("Counts", dataset_ax, region_ax, phi_ax)

    # Leading jet
    items["ak4_pt0"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)
    items["ak4_eta0"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax)
    items["ak4_phi0"] = Hist("Counts", dataset_ax, region_ax, jet_phi_ax)
    items["ak4_chf0"] = Hist("Counts", dataset_ax, region_ax, frac_ax)
    items["ak4_nhf0"] = Hist("Counts", dataset_ax, region_ax, frac_ax)
    items["ak4_nef0"] = Hist("Counts", dataset_ax, region_ax, frac_ax)

    # 2D pt-eta histogram for the leading jet
    items["ak4_pt0_eta0"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax, jet_eta_ax_coarse)

    # Dimuon system
    items["muon_pt0"] = Hist("Counts", dataset_ax, region_ax, pt_ax)
    items["muon_eta0"] = Hist("Counts", dataset_ax, region_ax, eta_ax)
    items["muon_phi0"] = Hist("Counts", dataset_ax, region_ax, phi_ax)
    items["muon_pt1"] = Hist("Counts", dataset_ax, region_ax, pt_ax)
    items["muon_eta1"] = Hist("Counts", dataset_ax, region_ax, eta_ax)
    items["muon_phi1"] = Hist("Counts", dataset_ax, region_ax, phi_ax)

    items['dphi_z_jet'] = Hist("Counts", dataset_ax, region_ax, dphi_ax)
    items['z_pt_over_jet_pt'] = Hist("Counts", dataset_ax, region_ax, ptfrac_ax)
    items['dimuon_mass'] = Hist("Counts", dataset_ax, region_ax, dilepton_mass_ax)

    items['sumw'] = processor.defaultdict_accumulator(float)
    items['sumw2'] = processor.defaultdict_accumulator(float)
    items['sumw_pileup'] = processor.defaultdict_accumulator(float)
    items['weights'] = Hist("Weights", dataset_ax, region_ax, weight_type_ax, weight_ax)

    return processor.dict_accumulator(items)

def zmumu_regions(cfg, variations=['']):
    two_mu_cuts = [
        'single_mu_trig', 
        'two_muons', 
        'at_least_one_tight_mu', 
        'dimuon_mass', 
        'mu_pt_trig_safe',
        'dimuon_charge', 
        'veto_ele', 
        'lead_ak4_pt_eta', 
        'lead_ak4_id', 
        'z_pt_eta',
        'dphi_z_jet', 
        'met_pt', 
        'z_pt_over_jet_pt',
        'veto_b',
        'hemveto',
        'filt_met'
        ]

    regions = {}
    for var in variations:
        regions[f'cr_2m_noEmEF{var}'] = two_mu_cuts
        regions[f'cr_2m_withEmEF{var}'] = two_mu_cuts + ['ak4_neEmEF']

    # Regions with prefire weights varied
    regions['cr_2m_noEmEF_prefireUp'] = two_mu_cuts
    regions['cr_2m_withEmEF_prefireUp'] = two_mu_cuts + ['ak4_neEmEF']
    regions['cr_2m_noEmEF_prefireDown'] = two_mu_cuts
    regions['cr_2m_withEmEF_prefireDown'] = two_mu_cuts + ['ak4_neEmEF']

    # Regions with pileup weights varied
    regions['cr_2m_noEmEF_pileupUp'] = two_mu_cuts
    regions['cr_2m_withEmEF_pileupUp'] = two_mu_cuts + ['ak4_neEmEF']
    regions['cr_2m_noEmEF_pileupDown'] = two_mu_cuts
    regions['cr_2m_withEmEF_pileupDown'] = two_mu_cuts + ['ak4_neEmEF']

    # Regions with no prefire weight applied
    regions['cr_2m_noEmEF_no_prefire'] = two_mu_cuts
    regions['cr_2m_withEmEF_no_prefire'] = two_mu_cuts + ['ak4_neEmEF']

    return regions