import copy
from collections import OrderedDict

import coffea.processor as processor
import numpy as np
from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray

from bucoffea.helpers import object_overlap, sigmoid, exponential
from bucoffea.helpers.dataset import extract_year

Hist = hist.Hist
Bin = hist.Bin
Cat = hist.Cat

def empty_column_accumulator_int():
    return processor.column_accumulator(np.array([],dtype=np.uint64))

def empty_column_accumulator_float():
    return processor.column_accumulator(np.array([],dtype=np.float64))

def accu_int():
    return processor.defaultdict_accumulator(int)

def defaultdict_accumulator_of_empty_column_accumulator_float():
    return processor.defaultdict_accumulator(empty_column_accumulator_float)

def monojet_accumulator(cfg, variations):
    dataset_ax = Cat("dataset", "Primary dataset")
    region_ax = Cat("region", "Selection region")
    type_ax = Cat("type", "Type")
    wppass_ax = Bin("wppass", "WP Pass",2,-0.5,1.5)

    vpt_ax = Bin("vpt",r"$p_{T}^{V}$ (GeV)", 50, 0, 2000)

    met_ax = Bin("met", r"$p_{T}^{miss}$ (GeV)", 100, 0, 500)
    recoil_ax = Bin("recoil", r"Recoil (GeV)", 200, 0, 2000)

    jet_pt_ax = Bin("jetpt", r"$p_{T}$ (GeV)", 50, 0, 1000)
    jet_eta_ax = Bin("jeteta", r"$\eta$", 50, -5, 5)
    jet_eta_ax_coarse = Bin("jeteta", r"$\eta$", 10, -5, 5)
    jet_phi_ax = Bin("jetphi", r"$\phi$", 50,-np.pi, np.pi)

    jet_mass_ax = Bin("mass", r"$M_{jet}$ (GeV)", 100,0,300)

    dpfcalo_ax = Bin("dpfcalo", r"$(CaloMET-PFMET) / Recoil$", 20, -1, 1)
    btag_ax = Bin("btag", r"B tag discriminator", 20, 0, 1)
    multiplicity_ax = Bin("multiplicity", r"multiplicity", 10, -0.5, 9.5)
    dphi_ax = Bin("dphi", r"$\Delta\phi$", 50, 0, 3.5)
    dr_ax = Bin("dr", r"$\Delta R$", 50, 0, 2)

    dxy_ax = Bin("dxy", r"$d_{xy}$", 20, 0, 0.5)
    dz_ax = Bin("dz", r"$d_{z}$", 20, 0, 0.5)
    id_ax = Bin("id", r"ID bool", 2,-0.5,1.5)

    pt_ax = Bin("pt", r"$p_{T}$ (GeV)", 50, 0, 1000)
    ht_ax = Bin("ht", r"$H_{T}$ (GeV)", 50, 0, 4000)
    mt_ax = Bin("mt", r"$M_{T}$ (GeV)", 50, 0, 1000)
    eta_ax = Bin("eta", r"$\eta$", 50, -5, 5)
    eta_ax_coarse = Bin("eta", r"$\eta$", 25, -5, 5)
    phi_ax = Bin("phi", r"$\phi$", 50,-np.pi, np.pi)
    phi_ax_coarse = Bin("phi", r"$\phi$", 20,-np.pi, np.pi)

    ratio_ax = Bin("ratio", "ratio", 50,0,2)

    tau21_ax = Bin("tau21", r"Tagger", 50,0,1)
    tagger_ax = Bin("tagger", r"Tagger", 50,0,1)

    dilepton_mass_ax = Bin("dilepton_mass", r"$M(\ell\ell)$ (GeV)", 50,50,150)

    weight_type_ax = Cat("weight_type", "Weight type")
    weight_ax = Bin("weight_value", "Weight",50,0.5,1.5)

    nvtx_ax = Bin('nvtx','Number of vertices',50,-0.5,99.5)
    rho_ax = Bin('rho','Energy density',50, 0, 100)
    frac_ax = Bin('frac','Fraction', 50, 0, 1)
    Hist = hist.Hist
    items = {}
    items["genvpt_check"] = Hist("Counts", dataset_ax, type_ax, vpt_ax)
    items["vpt"] = Hist("Counts", dataset_ax, region_ax, vpt_ax)
    items["lhe_ht"] = Hist("Counts", dataset_ax, ht_ax)
    items["met"] = Hist("Counts", dataset_ax, region_ax, met_ax)
    items["met_phi"] = Hist("Counts", dataset_ax, region_ax, phi_ax)
    items["recoil"] = Hist("Counts", dataset_ax, region_ax, recoil_ax)
    items["recoil_nopog"] = Hist("Counts", dataset_ax, region_ax, recoil_ax)
    items["recoil_nopu"] = Hist("Counts", dataset_ax, region_ax, recoil_ax)
    items["recoil_notrg"] = Hist("Counts", dataset_ax, region_ax, recoil_ax)
    items["recoil_nopref"] = Hist("Counts", dataset_ax, region_ax, recoil_ax)
    items["recoil_phi"] = Hist("Counts", dataset_ax, region_ax, phi_ax)
    items["ak4_pt0_over_recoil"] = Hist("Counts", dataset_ax, region_ax, ratio_ax)
    items["ak4_pt0"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)
    items["ak4_ptraw0"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)
    items["ak4_eta0"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax)
    items["ak4_phi0"] = Hist("Counts", dataset_ax, region_ax, jet_phi_ax)
    items["ak4_chf0"] = Hist("Counts", dataset_ax, region_ax, frac_ax)
    items["ak4_nhf0"] = Hist("Counts", dataset_ax, region_ax, frac_ax)

    items["ak4_pt0_eta0"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax,jet_eta_ax_coarse)

    items["ak4_pt"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)
    items["ak4_eta"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax)
    items["ak4_phi"] = Hist("Counts", dataset_ax, region_ax, jet_phi_ax)
    items["ak4_eta_phi"] = Hist("Counts", dataset_ax, region_ax, eta_ax_coarse, phi_ax_coarse)
    items["ak4_btag"] = Hist("Counts", dataset_ax, region_ax, btag_ax)

    items["ak8_pt0"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)
    items["ak8_eta0"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax)
    items["ak8_phi0"] = Hist("Counts", dataset_ax, region_ax, jet_phi_ax)
    items["ak8_mass0"] = Hist("Counts", dataset_ax, region_ax, jet_mass_ax)
    items["ak8_tau210"] = Hist("Counts", dataset_ax, region_ax, tau21_ax)
    items["ak8_wvsqcd0"] = Hist("Counts", dataset_ax, region_ax, tagger_ax)
    items["ak8_wvsqcdmd0"] = Hist("Counts", dataset_ax, region_ax, tagger_ax)
    items["ak8_zvsqcd0"] = Hist("Counts", dataset_ax, region_ax, tagger_ax)
    items["ak8_zvsqcdmd0"] = Hist("Counts", dataset_ax, region_ax, tagger_ax)
    items["ak8_tvsqcd0"] = Hist("Counts", dataset_ax, region_ax, tagger_ax)
    items["ak8_tvsqcdmd0"] = Hist("Counts", dataset_ax, region_ax, tagger_ax)
    items["ak8_wvstqcd0"] = Hist("Counts", dataset_ax, region_ax, tagger_ax)
    items["ak8_wvstqcdmd0"] = Hist("Counts", dataset_ax, region_ax, tagger_ax)
    items["ak8_passloose_pt0"] = Hist("Counts", dataset_ax, region_ax, wppass_ax, jet_pt_ax)
    items["ak8_passtight_pt0"] = Hist("Counts", dataset_ax, region_ax, wppass_ax, jet_pt_ax)
    items["ak8_passloosemd_pt0"] = Hist("Counts", dataset_ax, region_ax, wppass_ax, jet_pt_ax)
    items["ak8_passtightmd_pt0"] = Hist("Counts", dataset_ax, region_ax, wppass_ax, jet_pt_ax)
    items["ak8_passloose_mass0"] = Hist("Counts", dataset_ax, region_ax, wppass_ax, jet_mass_ax)
    items["ak8_passtight_mass0"] = Hist("Counts", dataset_ax, region_ax, wppass_ax, jet_mass_ax)
    items["ak8_passloosemd_mass0"] = Hist("Counts", dataset_ax, region_ax, wppass_ax, jet_mass_ax)
    items["ak8_passtightmd_mass0"] = Hist("Counts", dataset_ax, region_ax, wppass_ax, jet_mass_ax)
    items["ak8_Vmatched_pt0"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)

    items["ak8_pt"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)
    items["ak8_eta"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax)
    items["ak8_phi"] = Hist("Counts", dataset_ax, region_ax, jet_phi_ax)
    items["ak8_mass"] = Hist("Counts", dataset_ax, region_ax, jet_mass_ax)

    items["dpfcalo"] = Hist("Counts", dataset_ax, region_ax, dpfcalo_ax)
    items["dphijm"] = Hist("min(4 leading jets, MET)", dataset_ax, region_ax, dphi_ax)
    items["dphijr"] = Hist("min(4 leading jets, Recoil)", dataset_ax, region_ax, dphi_ax)

    # Multiplicity histograms
    for cand in ['ak4', 'ak8', 'bjet', 'loose_ele', 'loose_muo', 'tight_ele', 'tight_muo', 'tau', 'photon','gen_dilepton']:
        items[f"{cand}_mult"] = Hist(cand, dataset_ax, region_ax, multiplicity_ax)

    items["muon_pt"] = Hist("Counts", dataset_ax, region_ax, pt_ax)
    items["muon_eta"] = Hist("Counts", dataset_ax, region_ax, eta_ax)
    items["muon_phi"] = Hist("Counts", dataset_ax, region_ax, phi_ax)
    items["muon_eta_phi"] = Hist("Counts", dataset_ax, region_ax, eta_ax_coarse, phi_ax_coarse)
    items["muon_dxy"] = Hist("Counts", dataset_ax, region_ax, dxy_ax)
    items["muon_dz"] = Hist("Counts", dataset_ax, region_ax, dz_ax)
    items["muon_pt0"] = Hist("Counts", dataset_ax, region_ax, pt_ax)
    items["muon_eta0"] = Hist("Counts", dataset_ax, region_ax, eta_ax)
    items["muon_phi0"] = Hist("Counts", dataset_ax, region_ax, phi_ax)
    items["muon_pt1"] = Hist("Counts", dataset_ax, region_ax, pt_ax)
    items["muon_eta1"] = Hist("Counts", dataset_ax, region_ax, eta_ax)
    items["muon_phi1"] = Hist("Counts", dataset_ax, region_ax, phi_ax)
    items["muon_dxy0"] = Hist("Counts", dataset_ax, region_ax, dxy_ax)
    items["muon_dz0"] = Hist("Counts", dataset_ax, region_ax, dz_ax)
    items["muon_mt"] = Hist("Counts", dataset_ax, region_ax, mt_ax)
    items["dimuon_pt"] = Hist("Counts", dataset_ax, region_ax, pt_ax)
    items["dimuon_eta"] = Hist("Counts", dataset_ax, region_ax, eta_ax)
    items["dimuon_mass"] = Hist("Counts", dataset_ax, region_ax, dilepton_mass_ax)
    items["dimuon_dr"] = Hist("Counts", dataset_ax, region_ax, dr_ax)

    items["electron_pt"] = Hist("Counts", dataset_ax, region_ax, pt_ax)
    items["electron_eta"] = Hist("Counts", dataset_ax, region_ax, eta_ax)
    items["electron_phi"] = Hist("Counts", dataset_ax, region_ax, phi_ax)
    items["electron_eta_phi"] = Hist("Counts", dataset_ax, region_ax, eta_ax_coarse, phi_ax_coarse)
    items["electron_dxy"] = Hist("Counts", dataset_ax, region_ax, dxy_ax)
    items["electron_dz"] = Hist("Counts", dataset_ax, region_ax, dz_ax)
    items["electron_pt0"] = Hist("Counts", dataset_ax, region_ax, pt_ax)
    items["electron_eta0"] = Hist("Counts", dataset_ax, region_ax, eta_ax)
    items["electron_phi0"] = Hist("Counts", dataset_ax, region_ax, phi_ax)
    items["electron_pt1"] = Hist("Counts", dataset_ax, region_ax, pt_ax)
    items["electron_eta1"] = Hist("Counts", dataset_ax, region_ax, eta_ax)
    items["electron_phi1"] = Hist("Counts", dataset_ax, region_ax, phi_ax)
    items["electron_tightid1"] = Hist("Counts", dataset_ax, region_ax, id_ax)
    items["electron_mt"] = Hist("Counts", dataset_ax, region_ax, mt_ax)
    items["dielectron_pt"] = Hist("Counts", dataset_ax, region_ax, pt_ax)
    items["dielectron_eta"] = Hist("Counts", dataset_ax, region_ax, eta_ax)
    items["dielectron_mass"] = Hist("Counts", dataset_ax, region_ax, dilepton_mass_ax)
    items["dielectron_dr"] = Hist("Counts", dataset_ax, region_ax, dr_ax)

    items['photon_pt0'] = Hist("Counts", dataset_ax, region_ax, pt_ax)
    items['photon_eta0'] = Hist("Counts", dataset_ax, region_ax, eta_ax)
    items['photon_phi0'] = Hist("Counts", dataset_ax, region_ax, phi_ax)
    items["photon_eta_phi"] = Hist("Counts", dataset_ax, region_ax, eta_ax_coarse, phi_ax_coarse)

    items['drphotonjet'] = Hist("Counts", dataset_ax, region_ax, dr_ax)
    items['drelejet'] = Hist("Counts", dataset_ax, region_ax, dr_ax)
    items['drmuonjet'] = Hist("Counts", dataset_ax, region_ax, dr_ax)

    # One cutflow counter per region
    regions = monojet_regions(cfg, variations).keys()
    for region in regions:
        if region=="inclusive":
            continue
        items[f'cutflow_{region}']  = processor.defaultdict_accumulator(accu_int)

    items['nevents'] = processor.defaultdict_accumulator(float)
    items['sumw'] = processor.defaultdict_accumulator(float)
    items['sumw2'] = processor.defaultdict_accumulator(float)
    items['sumw_pileup'] = processor.defaultdict_accumulator(float)

    items['selected_events'] = processor.defaultdict_accumulator(empty_column_accumulator_int)
    items['kinematics'] = processor.defaultdict_accumulator(list)


    items['tree_1e'] = processor.defaultdict_accumulator(defaultdict_accumulator_of_empty_column_accumulator_float)
    items['tree_1m'] = processor.defaultdict_accumulator(defaultdict_accumulator_of_empty_column_accumulator_float)
    items['tree_2e'] = processor.defaultdict_accumulator(defaultdict_accumulator_of_empty_column_accumulator_float)
    items['tree_2m'] = processor.defaultdict_accumulator(defaultdict_accumulator_of_empty_column_accumulator_float)

    items['weights'] = Hist("Weights", dataset_ax, region_ax, weight_type_ax, weight_ax)
    items['npv'] = Hist('Number of primary vertices', dataset_ax, region_ax, nvtx_ax)
    items['npvgood'] = Hist('Number of good primary vertices', dataset_ax, region_ax, nvtx_ax)
    items['npv_nopu'] = Hist('Number of primary vertices (No PU weights)', dataset_ax, region_ax, nvtx_ax)
    items['npvgood_nopu'] = Hist('Number of good primary vertices (No PU weights)', dataset_ax, region_ax, nvtx_ax)

    items['rho_all'] = Hist(r'$\rho$ for all PF candidates', dataset_ax, region_ax, rho_ax)
    items['rho_central'] = Hist(r'$\rho$ for central PF candidates', dataset_ax, region_ax, rho_ax)
    items['rho_all_nopu'] = Hist(r'$\rho$ for all PF candidates (No PU weights)', dataset_ax, region_ax, rho_ax)
    items['rho_central_nopu'] = Hist(r'$\rho$ for central PF candidates (No PU weights)', dataset_ax, region_ax, rho_ax)

    return  processor.dict_accumulator(items)

class VarMap:
    '''An object holding relevant values/objects for each JES/JER variation.'''
    def __init__(self, variations):
        self._variations = variations
        self.ak4 = {}
        self.ak8 = {}
        self.bjets = {}
        self.diak4 = {}
        self.met = {}
        self.selection_packers = {}


    def fill_mapping(self, ak4, bjets, met, var, ak8):
        '''Fill the dictionaries for the relevant variation.'''
        self.ak4[var] = ak4 
        self.ak8[var] = ak8
        self.diak4[var] = ak4[:,:2].distincts() # Leading jet pair 
        self.bjets[var] = bjets 
        self.met[var] = met
   
    # Getter methods
    def get_ak4(self, var):
        return self.ak4[var]
    
    def get_ak8(self, var):
        return self.ak8[var]

    def get_bjets(self, var):
        return self.bjets[var]
    
    def get_diak4(self, var):
        return self.diak4[var]
    
    def get_met(self, var):
        return self.met[var]
    
    def get_selection_packer(self, var):
        return self.selection_packers[var]
    
    def set_selection_packer(self, var, sel):
        self.selection_packers[var] = sel

def setup_candidates(df, cfg, variations):
    if df['is_data'] and extract_year(df['dataset']) != 2018:
        # 2016, 2017 data
        jes_suffix = ''
    elif df['is_data']:
        # 2018 data
        jes_suffix = '_nom'
    else:
        # MC, all years
        jes_suffix = '_nom'
        if cfg.MET.JER:
            jes_suffix_met = '_jer'
        else:
            jes_suffix_met = '_nom'

    muons = JaggedCandidateArray.candidatesfromcounts(
        df['nMuon'],
        pt=df['Muon_pt'],
        eta=df['Muon_eta'],
        abseta=np.abs(df['Muon_eta']),
        phi=df['Muon_phi'],
        mass=0 * df['Muon_pt'],
        charge=df['Muon_charge'],
        looseId=df['Muon_looseId'],
        iso=df["Muon_pfRelIso04_all"],
        tightId=df['Muon_tightId'],
        dxy=df['Muon_dxy'],
        dz=df['Muon_dz']
    )

    # All muons must be at least loose
    muons = muons[muons.looseId \
                    & (muons.iso < cfg.MUON.CUTS.LOOSE.ISO) \
                    & (muons.pt > cfg.MUON.CUTS.LOOSE.PT) \
                    & (muons.abseta<cfg.MUON.CUTS.LOOSE.ETA) \
                    ]


    electrons = JaggedCandidateArray.candidatesfromcounts(
        df['nElectron'],
        pt=df['Electron_pt'],
        eta=df['Electron_eta'],
        abseta=np.abs(df['Electron_eta']),
        etasc=df['Electron_eta']+df['Electron_deltaEtaSC'],
        absetasc=np.abs(df['Electron_eta']+df['Electron_deltaEtaSC']),
        phi=df['Electron_phi'],
        mass=0 * df['Electron_pt'],
        charge=df['Electron_charge'],
        looseId=(df[cfg.ELECTRON.BRANCH.ID]>=1),
        tightId=(df[cfg.ELECTRON.BRANCH.ID]==4),
        dxy=np.abs(df['Electron_dxy']),
        dz=np.abs(df['Electron_dz']),
        barrel=np.abs(df['Electron_eta']+df['Electron_deltaEtaSC']) <= 1.4442
    )
    # All electrons must be at least loose
    pass_dxy = (electrons.barrel & (np.abs(electrons.dxy) < cfg.ELECTRON.CUTS.LOOSE.DXY.BARREL)) \
    | (~electrons.barrel & (np.abs(electrons.dxy) < cfg.ELECTRON.CUTS.LOOSE.DXY.ENDCAP))

    pass_dz = (electrons.barrel & (np.abs(electrons.dz) < cfg.ELECTRON.CUTS.LOOSE.DZ.BARREL)) \
    | (~electrons.barrel & (np.abs(electrons.dz) < cfg.ELECTRON.CUTS.LOOSE.DZ.ENDCAP))

    electrons = electrons[electrons.looseId \
                                    & (electrons.pt>cfg.ELECTRON.CUTS.LOOSE.PT) \
                                    & (electrons.absetasc<cfg.ELECTRON.CUTS.LOOSE.ETA) \
                                    & pass_dxy \
                                    & pass_dz
                                    ]

    if cfg.OVERLAP.ELECTRON.MUON.CLEAN:
        electrons = electrons[object_overlap(electrons, muons, dr=cfg.OVERLAP.ELECTRON.MUON.DR)]


    taus = JaggedCandidateArray.candidatesfromcounts(
        df['nTau'],
        pt=df['Tau_pt'],
        eta=df['Tau_eta'],
        abseta=np.abs(df['Tau_eta']),
        phi=df['Tau_phi'],
        mass=0 * df['Tau_pt'],
        decaymode=df[cfg.TAU.BRANCH.ID],
        iso=df[cfg.TAU.BRANCH.ISO],
    )

    taus = taus[ (taus.decaymode) \
                & (taus.pt > cfg.TAU.CUTS.PT)\
                & (taus.abseta < cfg.TAU.CUTS.ETA) \
                & ((taus.iso&2)==2)]

    if cfg.OVERLAP.TAU.MUON.CLEAN:
        taus = taus[object_overlap(taus, muons, dr=cfg.OVERLAP.TAU.MUON.DR)]
    if cfg.OVERLAP.TAU.ELECTRON.CLEAN:
        taus = taus[object_overlap(taus, electrons, dr=cfg.OVERLAP.TAU.ELECTRON.DR)]

    if cfg.PHOTON.BRANCH.IDV5 in df.columns:
        photon_id = cfg.PHOTON.BRANCH.IDV5
    else:
        photon_id = cfg.PHOTON.BRANCH.IDV7

    photons = JaggedCandidateArray.candidatesfromcounts(
        df['nPhoton'],
        pt=df['Photon_pt'],
        eta=df['Photon_eta'],
        abseta=np.abs(df['Photon_eta']),
        phi=df['Photon_phi'],
        mass=0*df['Photon_pt'],
        looseId=(df[photon_id]>=1) & df['Photon_electronVeto'],
        mediumId=(df[photon_id]>=2) & df['Photon_electronVeto'],
        r9=df['Photon_r9'],
        barrel=df['Photon_isScEtaEB'],
    )
    photons = photons[photons.looseId \
              & (photons.pt > cfg.PHOTON.CUTS.LOOSE.pt) \
              & (photons.abseta < cfg.PHOTON.CUTS.LOOSE.eta)
              ]

    if cfg.OVERLAP.PHOTON.MUON.CLEAN:
        photons = photons[object_overlap(photons, muons, dr=cfg.OVERLAP.PHOTON.MUON.DR)]
    if cfg.OVERLAP.PHOTON.ELECTRON.CLEAN:
        photons = photons[object_overlap(photons, electrons, dr=cfg.OVERLAP.PHOTON.ELECTRON.DR)]

    ak4 = JaggedCandidateArray.candidatesfromcounts(
        df['nJet'],
        pt=df[f'Jet_pt{jes_suffix}'] if (df['is_data'] or cfg.AK4.JER) else df[f'Jet_pt{jes_suffix}']/df['Jet_corr_JER'],
        eta=df['Jet_eta'],
        abseta=np.abs(df['Jet_eta']),
        phi=df['Jet_phi'],
        mass=np.zeros_like(df['Jet_pt']),
        looseId=(df['Jet_jetId']&2) == 2, # bitmask: 1 = loose, 2 = tight, 3 = tight + lep veto
        tightId=(df['Jet_jetId']&2) == 2, # bitmask: 1 = loose, 2 = tight, 3 = tight + lep veto
        puid=((df['Jet_puId']&2>0) | ((df[f'Jet_pt{jes_suffix}'] if (df['is_data'] or cfg.AK4.JER) else df[f'Jet_pt{jes_suffix}']/df['Jet_corr_JER'])>50)), # medium pileup jet ID
        csvv2=df["Jet_btagCSVV2"],
        deepcsv=df['Jet_btagDeepB'],
        nef=df['Jet_neEmEF'],
        nhf=df['Jet_neHEF'],
        chf=df['Jet_chHEF'],
        ptraw=df['Jet_pt']*(1-df['Jet_rawFactor']),
        nconst=df['Jet_nConstituents']
        # clean=df['Jet_cleanmask']
        # cef=df['Jet_chEmEF'],
    )
    # Get variated jet properties for each JEC variation
    for var in variations:
        if var == '':
            continue
        # For unclustered energy variations for MET, get the nominal value of jets
        if 'unclustEn' in var:
            ak4_pt = df[f'Jet_pt{jes_suffix}'] if (df['is_data'] or cfg.AK4.JER) else df[f'Jet_pt{jes_suffix}']/df['Jet_corr_JER']
        # For other variations, read off the pt values from the input tree, as calculated by tools
        else:  
            # If JER is turned off, calculate the variations of jet pt around unsmeared pt
            if not cfg.AK4.JER:
                ak4_pt = df[f'Jet_pt{var}'] / df['Jet_corr_JER']
            else:
                ak4_pt = df[f'Jet_pt{var}']

        ak4_puid = (df['Jet_puId']&2 > 0) | (ak4_pt > 50)

        # Set the jet values
        argdict = {f'pt{var}' : ak4_pt, f'puid{var}' : ak4_puid}
        ak4.add_attributes(**argdict)
        
    # Before cleaning, apply HEM veto
    hem_ak4 = ak4[ (ak4.pt>30) &
        (-3.0 < ak4.eta) &
        (ak4.eta < -1.3) &
        (-1.57 < ak4.phi) &
        (ak4.phi < -0.87)
        ]
    df['hemveto'] = hem_ak4.counts == 0

    # B jets have their own overlap cleaning,
    # so deal with them before applying filtering to jets
    btag_discriminator = getattr(ak4, cfg.BTAG.algo)
    btag_cut = cfg.BTAG.CUTS[cfg.BTAG.algo][cfg.BTAG.wp]

    ak8 = JaggedCandidateArray.candidatesfromcounts(
        df['nFatJet'],
        pt=df[f'FatJet_pt{jes_suffix}'] if (df['is_data'] or cfg.AK8.JER) else df[f'FatJet_pt{jes_suffix}']/df['FatJet_corr_JER'],
        eta=df['FatJet_eta'],
        abseta=np.abs(df['FatJet_eta']),
        phi=df['FatJet_phi'],
        mass=df[f'FatJet_msoftdrop{jes_suffix}'] if (df['is_data'] or cfg.AK8.JER) else df[f'FatJet_msoftdrop{jes_suffix}']/df['FatJet_corr_JER'],
        tightId=(df['FatJet_jetId']&2) == 2, # Tight
        csvv2=df["FatJet_btagCSVV2"],
        deepcsv=df['FatJet_btagDeepB'],
        tau1=df['FatJet_tau1'],
        tau2=df['FatJet_tau2'],
        tau21=df['FatJet_tau2']/df['FatJet_tau1'],
        wvsqcd=df['FatJet_deepTag_WvsQCD'],
        wvsqcdmd=df['FatJet_deepTagMD_WvsQCD'],
        zvsqcd=df['FatJet_deepTag_ZvsQCD'],
        zvsqcdmd=df['FatJet_deepTagMD_ZvsQCD'],
        tvsqcd=df['FatJet_deepTag_TvsQCD'],
        tvsqcdmd=df['FatJet_deepTagMD_TvsQCD'],
        wvstqcd=df['FatJet_deepTag_WvsQCD']*(1-df['FatJet_deepTag_TvsQCD'])/(1-df['FatJet_deepTag_WvsQCD']*df['FatJet_deepTag_TvsQCD']),
        wvstqcdmd=df['FatJet_deepTagMD_WvsQCD']*(1-df['FatJet_deepTagMD_TvsQCD'])/(1-df['FatJet_deepTagMD_WvsQCD']*df['FatJet_deepTagMD_TvsQCD']),
    )
    ak8 = ak8[ak8.tightId & object_overlap(ak8, muons) & object_overlap(ak8, electrons) & object_overlap(ak8, photons)]

    if extract_year(df['dataset']) == 2017:
        met_branch = 'METFixEE2017'
    else:
        met_branch = 'MET'

    met = JaggedCandidateArray.candidatesfromcounts(
        np.ones(df.size),
        pt=df[f'{met_branch}_T1Smear_pt'] if cfg.MET.JER else df[f'{met_branch}_T1_pt'],
        phi=df[f'{met_branch}_T1Smear_phi'] if cfg.MET.JER else df[f'{met_branch}_T1_phi'],
        eta=np.zeros(df.size), # dummy
        mass=np.zeros(df.size) # dummy
    )
    # Get variated MET properties for each JEC variation
    for var in variations:
        if var == '':
            continue
        # Annoying difference for unclustered energy variations:
        # These are named slightly differently + not calculated separately for T1 & T1Smear
        if 'unclustEn' in var:
            met_pt = df[f'{met_branch}_pt{var}']
            met_phi = df[f'{met_branch}_phi{var}']
        # If JER is turned off, calculate the variations of MET pt around the unsmeared (T1) MET
        else:
            if not cfg.MET.JER:
                met_pt = df[f'{met_branch}_T1_pt{var}']
                met_phi = df[f'{met_branch}_T1_phi{var}']
            else:
                met_pt = df[f'{met_branch}_T1Smear_pt{var}']
                met_phi = df[f'{met_branch}_T1Smear_phi{var}']
        # Set the MET values
        argdict = {f'pt{var}' : met_pt, f'phi{var}' : met_phi}
        met.add_attributes(**argdict)

    # Different sets of jets/b-jets/met for each 
    # JES/JER variation
    vmap = VarMap(variations=variations)
    
    for var in variations:
        ak4_pt = getattr(ak4, f'pt{var}')
        # Sort AK4 jets according to relevant pt
        _ak4 = ak4[ak4_pt.argsort()]
        ak4_pt = ak4_pt[ak4_pt.argsort()]
        
        # Choose relevant b-jets
        _bjets = _ak4[
              (_ak4.looseId) \
              & (ak4_pt > cfg.BTAG.PT) \
              & (_ak4.abseta < cfg.BTAG.ETA) \
              & (btag_discriminator > btag_cut) 
            ]
        
        # b-jet cleaning
        if cfg.OVERLAP.BTAG.MUON.CLEAN:
            _bjets = _bjets[object_overlap(_bjets, muons, dr=cfg.OVERLAP.BTAG.MUON.DR)]
        if cfg.OVERLAP.BTAG.ELECTRON.CLEAN:
            _bjets = _bjets[object_overlap(_bjets, electrons, dr=cfg.OVERLAP.BTAG.ELECTRON.DR)]
        if cfg.OVERLAP.BTAG.PHOTON.CLEAN:
            _bjets = _bjets[object_overlap(_bjets, photons, dr=cfg.OVERLAP.BTAG.PHOTON.DR)]
        
        # AK4 jets filtering/cleaning
        _ak4 = _ak4[_ak4.looseId]

        if cfg.OVERLAP.AK4.MUON.CLEAN:
            _ak4 = _ak4[object_overlap(_ak4, muons, dr=cfg.OVERLAP.AK4.MUON.DR)]
        if cfg.OVERLAP.AK4.ELECTRON.CLEAN:
            _ak4 = _ak4[object_overlap(_ak4, electrons, dr=cfg.OVERLAP.AK4.ELECTRON.DR)]
        if cfg.OVERLAP.AK4.PHOTON.CLEAN:
            _ak4 = _ak4[object_overlap(_ak4, photons, dr=cfg.OVERLAP.AK4.PHOTON.DR)]

        # Remove the jets in noisy region, in accordance with the v2 recipe
        if df['year'] == 2017:
            _ak4 = _ak4[(_ak4.ptraw>50) | (_ak4.abseta<2.65) | (_ak4.abseta>3.139)]
            _bjets = _bjets[(_bjets.ptraw>50) | (_bjets.abseta<2.65) | (_bjets.abseta>3.139)]

        # Filter w.r.t PU ID
        ak4_puid = getattr(_ak4, f'puid{var}')
        _ak4 = _ak4[ak4_puid] 

        bjets_puid = getattr(_bjets, f'puid{var}')
        _bjets = _bjets[bjets_puid] 

        vmap.fill_mapping(  ak4=_ak4,
                            ak8=ak8,
                            bjets=_bjets,
                            met=met,
                            var=var
                            )

    return vmap, muons, electrons, taus, photons

def monojet_regions(cfg, variations):
    regions = OrderedDict()

    for var in variations:
        common_cuts = [
            'filt_met',
            'hemveto',
            'veto_ele',
            'veto_muo',
            'veto_photon',
            'veto_tau',
            f'mindphijr{var}',
            f'dpfcalo{var}',
            f'recoil{var}'
        ]
        j_cuts = [
            f'leadak4_pt_eta{var}',
            f'leadak4_id{var}',
            # 'veto_vtag'
        ]
        # Test out different working point for v tagging
        # the first one is the traditional one used in 2016
        v_cuts = [
            f'leadak8_pt_eta{var}',
            f'leadak8_id{var}',
            f'leadak8_mass{var}',
            f'leadak8_tau21{var}',
        ]

        # Signal regions (v = mono-V, j = mono-jet)
        regions[f'sr_v{var}'] = ['trig_met'] + common_cuts + v_cuts
        regions[f'sr_j{var}'] = ['trig_met'] + common_cuts + j_cuts
    
        # Dimuon CR
        cr_2m_cuts = ['trig_met','two_muons', 'at_least_one_tight_mu', 'dimuon_mass', 'dimuon_charge'] + common_cuts
        cr_2m_cuts.remove('veto_muo')
    
        # regions[f'cr_2m_j{var}'] = cr_2m_cuts + j_cuts
        # regions[f'cr_2m_v{var}'] = cr_2m_cuts + v_cuts

        # Single muon CR
        # cr_1m_cuts = ['trig_met','one_muon', 'at_least_one_tight_mu', f'mt_mu{var}'] + common_cuts
        # cr_1m_cuts.remove('veto_muo')
        # regions[f'cr_1m_j{var}'] = cr_1m_cuts + j_cuts
        # regions[f'cr_1m_v{var}'] = cr_1m_cuts + v_cuts

        # Dielectron CR
        cr_2e_cuts = ['trig_ele','two_electrons', 'at_least_one_tight_el', 'dielectron_mass', 'dielectron_charge'] + common_cuts
        cr_2e_cuts.remove('veto_ele')
        regions[f'cr_2e_j{var}'] = cr_2e_cuts + j_cuts
        regions[f'cr_2e_v{var}'] = cr_2e_cuts + v_cuts


        # Dielectron CR with no recoil cut 
        regions[f'cr_2e_j_norecoil{var}'] = copy.deepcopy(regions[f'cr_2e_j{var}'])
        regions[f'cr_2e_j_norecoil{var}'].remove(f'recoil{var}')
        regions[f'cr_2e_j_norecoil{var}'].append(f'dphi_z_jet{var}')
        regions[f'cr_2e_j_norecoil{var}'].append(f'dpt_z_jet{var}')

        regions[f'cr_2e_j_norecoil_nojpt{var}'] = copy.deepcopy(regions[f'cr_2e_j{var}'])
        regions[f'cr_2e_j_norecoil_nojpt{var}'].remove(f'recoil{var}')
        regions[f'cr_2e_j_norecoil_nojpt{var}'].remove(f'leadak4_pt_eta{var}')
        regions[f'cr_2e_j_norecoil_nojpt{var}'].append(f'dphi_z_jet{var}')
        regions[f'cr_2e_j_norecoil_nojpt{var}'].append(f'dpt_z_jet{var}')

        regions[f'cr_2e_j_norecoil_jptv2{var}'] = copy.deepcopy(regions[f'cr_2e_j{var}'])
        regions[f'cr_2e_j_norecoil_jptv2{var}'].remove(f'recoil{var}')
        regions[f'cr_2e_j_norecoil_jptv2{var}'].remove(f'leadak4_pt_eta{var}')
        regions[f'cr_2e_j_norecoil_jptv2{var}'].append(f'leadak4_pt_eta_v2{var}')
        regions[f'cr_2e_j_norecoil_jptv2{var}'].append(f'dphi_z_jet{var}')
        regions[f'cr_2e_j_norecoil_jptv2{var}'].append(f'dpt_z_jet{var}')

        # Region without the HEM veto for testing
        regions[f'cr_2e_j_norecoil_nojpt_nohemveto{var}'] = copy.deepcopy(regions[f'cr_2e_j{var}'])
        regions[f'cr_2e_j_norecoil_nojpt_nohemveto{var}'].remove(f'recoil{var}')
        regions[f'cr_2e_j_norecoil_nojpt_nohemveto{var}'].remove(f'leadak4_pt_eta{var}')
        regions[f'cr_2e_j_norecoil_nojpt_nohemveto{var}'].append(f'dphi_z_jet{var}')
        regions[f'cr_2e_j_norecoil_nojpt_nohemveto{var}'].append(f'dpt_z_jet{var}')
        regions[f'cr_2e_j_norecoil_nojpt_nohemveto{var}'].remove('hemveto')

        # Single electron CR
        cr_1e_cuts = ['trig_ele','one_electron', 'at_least_one_tight_el', f'met_el{var}',f'mt_el{var}'] + common_cuts
        cr_1e_cuts.remove('veto_ele')
        # regions[f'cr_1e_j{var}'] =  cr_1e_cuts + j_cuts
        # regions[f'cr_1e_v{var}'] =  cr_1e_cuts + v_cuts

        # Photon CR
        cr_g_cuts = ['trig_photon', 'one_photon', 'at_least_one_tight_photon','photon_pt'] + common_cuts
        cr_g_cuts.remove('veto_photon')
    
        # regions[f'cr_g_j{var}'] = cr_g_cuts + j_cuts
        # regions[f'cr_g_v{var}'] = cr_g_cuts + v_cuts

        # a tt-bar populated region by removing b veto
        # regions[f'cr_nobveto_v{var}'] = copy.deepcopy(regions[f'sr_v{var}'])
        # regions[f'cr_nobveto_v{var}'].remove(f'veto_b{var}')

    ###############################
    # NOTE: No updates starting from this line
    ###############################

    # additional regions to test out deep ak8 WvsQCD tagger
    # for region in ['sr_v','cr_2m_v','cr_1m_v','cr_2e_v','cr_1e_v','cr_g_v','cr_nobveto_v']:
        # for wp in ['inclusive', 'loose', 'tight','loosemd','tightmd']:
            # # the new region name will be, for example, cr_2m_loose_v
            # newRegionName=region.replace('_v','_'+wp+'_v')
            # regions[newRegionName] = copy.deepcopy(regions[region])
            # regions[newRegionName].remove('leadak8_tau21')
            # if wp == 'inclusive':
                # regions[newRegionName].remove('leadak8_mass')

                # # add regions: cr_2m_hasmass_inclusive_v, cr_1m_hasmass_inclusive_v, cr_2e_hasmass_inclusive_v, cr_1e_hasmass_inclusive_v
                # # these are regions with mass cut but has no tagger cut
                # hasMassRegionName = region.replace('_v', '_hasmass_'+ wp + '_v')
                # regions[hasMassRegionName] = regions[newRegionName] + ['leadak8_mass']

            # else:
                # regions[newRegionName].append('leadak8_wvsqcd_'+wp)
            # # save a copy of the v-tagged regions but not applying mistag SFs, for the sake of measuring mistag SF later
            # if wp in ['loose','tight','loosemd','tightmd']:
                # noMistagRegionName = region.replace('_v', '_nomistag_'+ wp + '_v')
                # regions[noMistagRegionName]=copy.deepcopy(regions[newRegionName])


    if cfg.RUN.TRIGGER_STUDY:
        # Trigger studies
        # num = numerator, den = denominator
        # Single Mu region: Remove recoil cut, add SingleMu trigger, toggle MET trigger
        tr_1m_num_cuts = copy.deepcopy(cr_1m_cuts) + j_cuts
        tr_1m_num_cuts.remove('recoil')
        tr_1m_num_cuts.append('trig_mu')
        tr_1m_num_cuts.append('mu_pt_trig_safe')
        regions['tr_1m_num'] = tr_1m_num_cuts

        tr_1m_den_cuts = copy.deepcopy(tr_1m_num_cuts)
        tr_1m_den_cuts.remove('trig_met')
        regions['tr_1m_den'] = tr_1m_den_cuts

        # Add HLT muon requirement
        regions['tr_1m_hlt_num'] = regions['tr_1m_num'] + ['one_hlt_muon']
        regions['tr_1m_hlt_den'] = regions['tr_1m_den'] + ['one_hlt_muon']

        # Double Mu region: Remove recoil cut, toggle MET trigger
        tr_2m_num_cuts = copy.deepcopy(cr_2m_cuts) + j_cuts
        tr_2m_num_cuts.remove('recoil')
        tr_2m_num_cuts.append('trig_mu')
        tr_2m_num_cuts.append('mu_pt_trig_safe')
        regions['tr_2m_num'] = tr_2m_num_cuts

        tr_2m_den_cuts = copy.deepcopy(tr_2m_num_cuts)
        tr_2m_den_cuts.remove('trig_met')
        regions['tr_2m_den'] = tr_2m_den_cuts

        regions['tr_2m_hlt_den'] = regions['tr_2m_den'] + ['two_hlt_muons']
        regions['tr_2m_hlt_num'] = regions['tr_2m_num'] + ['two_hlt_muons']

        # Single Electron region: Remove recoil cut, toggle MET trigger
        tr_1e_num_cuts = copy.deepcopy(cr_1e_cuts) + j_cuts
        tr_1e_num_cuts.remove('recoil')
        tr_1e_num_cuts.append('trig_met')
        regions['tr_1e_num'] = tr_1e_num_cuts

        tr_1e_den_cuts = copy.deepcopy(tr_1e_num_cuts)
        tr_1e_den_cuts.remove('trig_met')
        regions['tr_1e_den'] = tr_1e_den_cuts

        # Double Electron region: Remove recoil cut, toggle MET trigger
        tr_2e_num_cuts = copy.deepcopy(cr_2e_cuts) + j_cuts
        tr_2e_num_cuts.remove('recoil')
        tr_2e_num_cuts.append('trig_met')
        regions['tr_2e_num'] = tr_2e_num_cuts

        tr_2e_den_cuts = copy.deepcopy(tr_2e_num_cuts)
        tr_2e_den_cuts.remove('trig_met')
        regions['tr_2e_den'] = tr_2e_den_cuts

        # Photon region
        tr_g_num_cuts = copy.deepcopy(cr_g_cuts) + j_cuts
        tr_g_num_cuts.remove('recoil')
        tr_g_num_cuts.remove('photon_pt')

        tr_g_den_cuts = copy.deepcopy(tr_g_num_cuts)
        tr_g_den_cuts.remove('trig_photon')

        regions[f'tr_g_notrig_num'] = copy.deepcopy(tr_g_num_cuts)
        regions[f'tr_g_notrig_den'] = copy.deepcopy(tr_g_den_cuts)

        for trgname in cfg.TRIGGERS.HT.GAMMAEFF:
            regions[f'tr_g_{trgname}_num'] = tr_g_num_cuts + [trgname]
            regions[f'tr_g_{trgname}_den'] = tr_g_den_cuts + [trgname]

            regions[f'tr_g_{trgname}_photon_pt_trig_cut_num'] = tr_g_num_cuts + [trgname, 'photon_pt_trig']
            regions[f'tr_g_{trgname}_photon_pt_trig_cut_den'] = tr_g_den_cuts + [trgname, 'photon_pt_trig']

    if not cfg.RUN.MONOV:
        keys_to_remove = [ x for x in regions.keys() if x.endswith('_v') or '_v_' in x]
        for key in keys_to_remove:
            regions.pop(key)
    if not cfg.RUN.MONOJ:
        keys_to_remove = [ x for x in regions.keys() if x.endswith('_j') or '_j_' in x]
        for key in keys_to_remove:
            regions.pop(key)


    return regions


def fitfun(x, a, b, c):
    return a * np.exp(-b * x) + c

def theory_weights_monojet(weights, df, evaluator, gen_v_pt):
    if df['is_lo_w']:
        if extract_year(df['dataset']) == 2016:
            qcd_nlo = evaluator["qcd_nlo_w_2016"](gen_v_pt)
        else:
            qcd_nlo = fitfun(gen_v_pt, 1.053, 3.163e-3, 0.746)
        theory_weights =  qcd_nlo * evaluator["ewk_nlo_w"](gen_v_pt)
    elif df['is_lo_z']:
        if df['is_lo_znunu']:
            qcd_nlo = evaluator["qcd_nlo_znn_2017"](gen_v_pt)
        else:
            qcd_nlo = evaluator["qcd_nlo_dy_2017"](gen_v_pt)
        theory_weights =  qcd_nlo * evaluator["ewk_nlo_z"](gen_v_pt)
    elif df['is_nlo_w']:
        theory_weights = evaluator["ewk_nlo_w"](gen_v_pt)
    elif df['is_nlo_z']:
        theory_weights = evaluator["ewk_nlo_z"](gen_v_pt)
    elif df['is_lo_g']:
        theory_weights = evaluator["qcd_nlo_g"](gen_v_pt) * evaluator["ewk_nlo_g"](gen_v_pt)
    else:
        theory_weights = np.ones(df.size)

    # Guard against invalid input pt
    invalid = (gen_v_pt <=0) | np.isinf(gen_v_pt) | np.isnan(gen_v_pt)
    theory_weights[invalid] = 1

    weights.add('theory', theory_weights)
    return weights

def theory_weights_vbf(weights, df, evaluator, gen_v_pt, mjj):
    if df['is_lo_w']:
        theory_weights = evaluator["qcd_nlo_w_2017_2d"](mjj, gen_v_pt) * evaluator["ewk_nlo_w"](gen_v_pt)
    elif df['is_lo_w_ewk']:
        theory_weights = evaluator["qcd_nlo_w_ewk"](gen_v_pt, mjj)
    elif df['is_lo_z']:
        w_ewk = evaluator["ewk_nlo_z"](gen_v_pt)
        if df['is_lo_znunu']:
            w_qcd = evaluator["qcd_nlo_znn_2017_2d"](gen_v_pt, mjj)
        else:
            w_qcd = evaluator["qcd_nlo_z_2017_2d"](mjj, gen_v_pt)
        theory_weights = w_ewk * w_qcd

    elif df['is_lo_z_ewk']:
        theory_weights = evaluator["qcd_nlo_z_ewk"](gen_v_pt, mjj)
    elif df['is_nlo_w']:
        theory_weights = evaluator["ewk_nlo_w"](gen_v_pt)
    elif df['is_nlo_z']:
        theory_weights = evaluator["ewk_nlo_z"](gen_v_pt)
    elif df['is_lo_g']:
        theory_weights = evaluator["qcd_nlo_g_2017_2d"](mjj, gen_v_pt) * evaluator["ewk_nlo_g"](gen_v_pt)
    else:
        theory_weights = np.ones(df.size)

    # Guard against invalid input pt
    invalid = (gen_v_pt <=0) | np.isinf(gen_v_pt) | np.isnan(gen_v_pt)
    theory_weights[invalid] = 1

    weights.add('theory', theory_weights)

    return weights

def pileup_weights(weights, df, evaluator, cfg):
    if cfg.SF.PILEUP.MODE == 'nano':
                weights.add("pileup", df['puWeight'])
    elif cfg.SF.PILEUP.MODE == 'manual':
        weights.add("pileup", evaluator['pileup'](df['Pileup_nTrueInt']))
    else:
        raise RuntimeError(f"Unknown value for cfg.PILEUP.MODE: {cfg.PILEUP.MODE}.")
    return weights

def photon_trigger_sf(weights, photons, df):
    """MC-to-data photon trigger scale factor.

    The scale factor is obtained by separately fitting the
    trigger turn-on with a sigmoid function in data and MC.
    The scale factor is then the ratio of the two sigmoid
    functions as a function of the photon pt.

    :param weights: Weights object to write information into
    :type weights: WeightsContainer
    :param photons: Photon candidates
    :type photons: JaggedCandidateArray
    :param df: Data frame
    :type df: LazyDataFrame
    """
    year = extract_year(df['dataset'])
    x = photons.pt.max()
    if year == 2016:
        sf =  np.ones(df.size)
    elif year == 2017:
        sf = sigmoid(x,0.335,217.91,0.065,0.996) / sigmoid(x,0.244,212.34,0.050,1.000)
    elif year == 2018:
        sf = sigmoid(x,1.022, 218.39, 0.086, 0.999) / sigmoid(x, 0.301,212.83,0.062,1.000)

    sf[np.isnan(sf) | np.isinf(sf)] == 1

    weights.add("trigger_photon", sf)

def candidate_weights(weights, df, evaluator, muons, electrons, photons, cfg):
    year = extract_year(df['dataset'])
    # Muon ID and Isolation for tight and loose WP
    # Function of pT, eta (Order!)
    weights.add("muon_id_tight", evaluator['muon_id_tight'](muons[df['is_tight_muon']].pt, muons[df['is_tight_muon']].abseta).prod())
    weights.add("muon_iso_tight", evaluator['muon_iso_tight'](muons[df['is_tight_muon']].pt, muons[df['is_tight_muon']].abseta).prod())
    weights.add("muon_id_loose", evaluator['muon_id_loose'](muons[~df['is_tight_muon']].pt, muons[~df['is_tight_muon']].abseta).prod())
    weights.add("muon_iso_loose", evaluator['muon_iso_loose'](muons[~df['is_tight_muon']].pt, muons[~df['is_tight_muon']].abseta).prod())

    # Electron ID and reco
    # Function of eta, pT (Other way round relative to muons!)

    # For 2017, the reco SF is split below/above 20 GeV
    if year == 2017:
        high_et = electrons.pt>20
        ele_reco_sf = evaluator['ele_reco'](electrons.etasc[high_et], electrons.pt[high_et]).prod()
        ele_reco_sf *= evaluator['ele_reco_pt_lt_20'](electrons.etasc[~high_et], electrons.pt[~high_et]).prod()
    else:
        ele_reco_sf = evaluator['ele_reco'](electrons.etasc, electrons.pt).prod()
    weights.add("ele_reco", ele_reco_sf)
    # ID/iso SF is not split
    weights.add("ele_id_tight", evaluator['ele_id_tight'](electrons[df['is_tight_electron']].etasc, electrons[df['is_tight_electron']].pt).prod())
    weights.add("ele_id_loose", evaluator['ele_id_loose'](electrons[~df['is_tight_electron']].etasc, electrons[~df['is_tight_electron']].pt).prod())

    # Photon ID and electron veto
    if cfg.SF.PHOTON.USETNP:
        weights.add("photon_id_tight", evaluator['photon_id_tight_tnp'](np.abs(photons[df['is_tight_photon']].eta)).prod())
    else:
        weights.add("photon_id_tight", evaluator['photon_id_tight'](photons[df['is_tight_photon']].eta, photons[df['is_tight_photon']].pt).prod())

    if year == 2016:
        csev_weight = evaluator["photon_csev"](photons.abseta, photons.pt).prod()
    elif year == 2017:
        csev_sf_index = 0.5 * photons.barrel + 3.5 * ~photons.barrel + 1 * (photons.r9 > 0.94) + 2 * (photons.r9 <= 0.94)
        csev_weight = evaluator['photon_csev'](csev_sf_index).prod()
    elif year == 2018:
        csev_weight = evaluator['photon_csev'](photons.pt, photons.abseta).prod()
    csev_weight[csev_weight==0] = 1
    weights.add("photon_csev", csev_weight)

    return weights

def data_driven_qcd_dataset(dataset):
    """Dataset name to use for data-driven QCD estimate"""
    year = extract_year(dataset)
    return f"QCD_data_{year}"

def photon_impurity_weights(photon_pt, year):
    """Photon impurity as a function of pt

    :param photon_pt: Photon pt
    :type photon_pt: 1D array
    :param year: Data-taking year
    :type year: int
    :return: Weights
    :rtype: 1Darray
    """
    if year == 2017:
        a = 6.35
        b = 4.61e-3
        c = 1.05
    elif year==2018:
        a = 11.92
        b = 8.28e-3
        c = 1.55
    elif year==2016:
        return np.ones(photon_pt.size)

    # Remember to multiply by 1e-2,
    # because fit is done to percentages
    return 1e-2*exponential(photon_pt, a, b, c)
