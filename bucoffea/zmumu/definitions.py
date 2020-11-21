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

def zmumu_regions(cfg):
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
    regions['cr_2m_noEmEF'] = two_mu_cuts
    regions['cr_2m_withEmEF'] = two_mu_cuts + ['ak4_neEmEF']

    # Regions with prefire weights varied
    regions['cr_2m_noEmEF_prefireUp'] = two_mu_cuts
    regions['cr_2m_withEmEF_prefireUp'] = two_mu_cuts + ['ak4_neEmEF']
    regions['cr_2m_noEmEF_prefireDown'] = two_mu_cuts
    regions['cr_2m_withEmEF_prefireDown'] = two_mu_cuts + ['ak4_neEmEF']

    # Regions with no prefire weight applied
    regions['cr_2m_noEmEF_no_prefire'] = two_mu_cuts
    regions['cr_2m_withEmEF_no_prefire'] = two_mu_cuts + ['ak4_neEmEF']

    # Regions categorized by jet eta
    if cfg.RUN.EFF_STUDY.SPLIT_JET_ETA:
        regions['cr_2m_noEmEF_jeteta_lt_2_3'] = two_mu_cuts + ['jet_eta_lt_2_3']
        regions['cr_2m_noEmEF_jeteta_jet_eta_gt_2_3_lt_2_7'] = two_mu_cuts + ['jet_eta_gt_2_3_lt_2_7']
        regions['cr_2m_noEmEF_jeteta_jet_eta_gt_2_7_lt_3_0'] = two_mu_cuts + ['jet_eta_gt_2_7_lt_3_0']
        regions['cr_2m_noEmEF_jeteta_gt_3_0'] = two_mu_cuts + ['jet_eta_gt_3_0']

    if cfg.RUN.EFF_STUDY.ENDCAP_ONLY_REGIONS:
        # Leading jet in the positive endcap: i.e. 2.5 < eta < 3.0
        regions['cr_2m_noEmEF_ak40_in_pos_endcap'] = two_mu_cuts + ['ak40_in_pos_endcap']
        regions['cr_2m_withEmEF_ak40_in_pos_endcap'] = two_mu_cuts + ['ak40_in_pos_endcap', 'ak4_neEmEF']
        # Leading jet in the negative endcap: i.e. -3.0 < eta < -2.5
        regions['cr_2m_noEmEF_ak40_in_neg_endcap'] = two_mu_cuts + ['ak40_in_neg_endcap']
        regions['cr_2m_withEmEF_ak40_in_neg_endcap'] = two_mu_cuts + ['ak40_in_neg_endcap', 'ak4_neEmEF']
        # Leading jet in endcap (both sides): 2.5 < |eta| < 3.0
        regions['cr_2m_noEmEF_ak40_in_endcap'] = two_mu_cuts + ['ak40_in_endcap']
        regions['cr_2m_withEmEF_ak40_in_endcap'] = two_mu_cuts + ['ak40_in_endcap', 'ak4_neEmEF']

    # Regions with tighter selections
    if cfg.RUN.EFF_STUDY.TIGHTCUTS:
        regions['cr_2m_noEmEF_tightBalCut'] = two_mu_cuts + ['z_pt_over_jet_pt_tight']
        regions['cr_2m_noEmEF_tightBalCut'].remove('z_pt_over_jet_pt')
        regions['cr_2m_withEmEF_tightBalCut'] = two_mu_cuts + ['z_pt_over_jet_pt_tight', 'ak4_neEmEF']
        regions['cr_2m_withEmEF_tightBalCut'].remove('z_pt_over_jet_pt')
    
        regions['cr_2m_noEmEF_tightMassCut'] = two_mu_cuts + ['dimuon_mass_tight']
        regions['cr_2m_noEmEF_tightMassCut'].remove('dimuon_mass')
        regions['cr_2m_withEmEF_tightMassCut'] = two_mu_cuts + ['dimuon_mass_tight', 'ak4_neEmEF']
        regions['cr_2m_withEmEF_tightMassCut'].remove('dimuon_mass')
    
        regions['cr_2m_noEmEF_tight'] = copy.deepcopy(regions['cr_2m_noEmEF_tightBalCut']) + ['dimuon_mass_tight']
        regions['cr_2m_noEmEF_tight'].remove('dimuon_mass')
        regions['cr_2m_withEmEF_tight'] = copy.deepcopy(regions['cr_2m_withEmEF_tightBalCut']) + ['dimuon_mass_tight']
        regions['cr_2m_withEmEF_tight'].remove('dimuon_mass')
        
        regions['cr_2m_noEmEF_very_tight'] = copy.deepcopy(regions['cr_2m_noEmEF_tight']) + ['z_pt_over_jet_pt_very_tight']
        regions['cr_2m_noEmEF_very_tight'].remove('z_pt_over_jet_pt_tight')
        regions['cr_2m_withEmEF_very_tight'] = copy.deepcopy(regions['cr_2m_withEmEF_tight']) + ['z_pt_over_jet_pt_very_tight']
        regions['cr_2m_withEmEF_very_tight'].remove('z_pt_over_jet_pt_tight')
    
    return regions