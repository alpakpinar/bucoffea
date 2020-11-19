import copy
import coffea.processor as processor
import re
import numpy as np
from dynaconf import settings as cfg

from bucoffea.helpers import (
                              bucoffea_path,
                              dphi,
                              evaluator_from_config,
                              mask_and,
                              mask_or,
                              min_dphi_jet_met,
                              mt,
                              recoil,
                              weight_shape,
                              candidates_in_hem,
                              electrons_in_hem,
                              calculate_vecB,
                              calculate_vecDPhi
                              )
from bucoffea.helpers.dataset import (
                                      extract_year,
                                      is_data,
                                      is_lo_g,
                                      is_lo_w,
                                      is_lo_z,
                                      is_lo_w_ewk,
                                      is_lo_z_ewk,
                                      is_nlo_w,
                                      is_nlo_z,
                                      is_lo_znunu
                                      )
from bucoffea.helpers.gen import (
                                  setup_gen_candidates,
                                  setup_dressed_gen_candidates,
                                  setup_lhe_cleaned_genjets,
                                  fill_gen_v_info
                                 )
from bucoffea.helpers.weights import (
                                  get_veto_weights
                                 )
from bucoffea.monojet.definitions import (
                                          candidate_weights,
                                          pileup_weights,
                                          setup_candidates,
                                          theory_weights_vbf,
                                          photon_trigger_sf,
                                          photon_impurity_weights,
                                          data_driven_qcd_dataset
                                          )
from bucoffea.vbfhinv.definitions import (
                                           vbfhinv_accumulator,
                                           vbfhinv_regions,
                                           ak4_em_frac_weights
                                         )


# Cuts for signal region
sr_cuts = [
        'veto_ele',
        'veto_muo',
        'filt_met',
        'trig_met',
        'mindphijm',
        'recoil',
        'two_jets',
        'leadak4_pt_eta',
        'leadak4_id',
        'trailak4_pt_eta',
        'trailak4_id',
        'hemisphere',
        'mjj',
        'dphijj',
        'detajj',
        'veto_photon',
        'veto_tau',
        'veto_b',
        'max_neEmEF',
        'dpfcalo_sr',
        'veto_hfhf',
        'eemitigation'
]

regions = {
    'inclusive' : ['inclusive'],
    'sr_vbf' : sr_cuts
}

def trigger_selection(selection, df, cfg):
    # MET filters / triggers
    selection.add('filt_met', mask_and(df, cfg.FILTERS.MC))
    selection.add('trig_met', mask_or(df, cfg.TRIGGERS.MET))

    return selection

class lightVbfhinvProcessor(processor.ProcessorABC):
    def __init__(self, blind=False):
        self._year=None
        self._blind=blind
        self._configure()
        self._accumulator = vbfhinv_accumulator(cfg)

    @property
    def accumulator(self):
        return self._accumulator

    def _configure(self, df=None):
        cfg.DYNACONF_WORKS="merge_configs"
        cfg.MERGE_ENABLED_FOR_DYNACONF=True
        cfg.SETTINGS_FILE_FOR_DYNACONF = bucoffea_path("config/vbfhinv.yaml")

        # Reload config based on year
        if df:
            dataset = df['dataset']
            self._year = extract_year(dataset)
            df["year"] = self._year
            cfg.ENV_FOR_DYNACONF = f"era{self._year}"
        else:
            cfg.ENV_FOR_DYNACONF = f"default"
        cfg.reload()

    def process(self, df):
        self._configure(df)
        dataset = df['dataset']
        df['is_data'] = is_data(dataset)

        # Candidates
        # Already pre-filtered!
        # All leptons are at least loose
        # Check out setup_candidates for filtering details
        met_pt, met_phi, ak4, bjets, _, muons, electrons, taus, photons = setup_candidates(df, cfg)

        # Remove jets in accordance with the noise recipe
        if df['year'] == 2017:
            ak4   = ak4[(ak4.ptraw>50) | (ak4.abseta<2.65) | (ak4.abseta>3.139)]
            bjets = bjets[(bjets.ptraw>50) | (bjets.abseta<2.65) | (bjets.abseta>3.139)]

        # Filtering ak4 jets according to pileup ID
        ak4 = ak4[ak4.puid]

        # Recoil
        df['recoil_pt'], df['recoil_phi'] = recoil(met_pt,met_phi, electrons, muons, photons)

        df["dPFCaloSR"] = (met_pt - df["CaloMET_pt"]) / met_pt
        df["dPFCaloCR"] = (met_pt - df["CaloMET_pt"]) / df["recoil_pt"]

        df["dPFTkSR"] = (met_pt - df["TkMET_pt"]) / met_pt

        df["minDPhiJetRecoil"] = min_dphi_jet_met(ak4, df['recoil_phi'], njet=4, ptmin=30, etamax=5.0)
        df["minDPhiJetMet"] = min_dphi_jet_met(ak4, met_phi, njet=4, ptmin=30, etamax=5.0)
        selection = processor.PackedSelection()

        # Triggers
        pass_all = np.ones(df.size)==1
        selection.add('inclusive', pass_all)
        selection = trigger_selection(selection, df, cfg)

        # Common selection
        selection.add('veto_ele', electrons.counts==0)
        selection.add('veto_muo', muons.counts==0)
        selection.add('veto_photon', photons.counts==0)
        selection.add('veto_tau', taus.counts==0)
        selection.add('veto_b', bjets.counts==0)
        selection.add('mindphijm',df['minDPhiJetMet'] > cfg.SELECTION.SIGNAL.MINDPHIJR)

        selection.add('dpfcalo_sr',np.abs(df['dPFCaloSR']) < cfg.SELECTION.SIGNAL.DPFCALO)

        selection.add('recoil', df['recoil_pt']>cfg.SELECTION.SIGNAL.RECOIL)
        selection.add('met_sr', met_pt>cfg.SELECTION.SIGNAL.RECOIL)

        # AK4 dijet
        diak4 = ak4[:,:2].distincts()
        leadak4_pt_eta = (diak4.i0.pt > cfg.SELECTION.SIGNAL.LEADAK4.PT) & (np.abs(diak4.i0.eta) < cfg.SELECTION.SIGNAL.LEADAK4.ETA)
        trailak4_pt_eta = (diak4.i1.pt > cfg.SELECTION.SIGNAL.TRAILAK4.PT) & (np.abs(diak4.i1.eta) < cfg.SELECTION.SIGNAL.TRAILAK4.ETA)
        hemisphere = (diak4.i0.eta * diak4.i1.eta < 0).any()
        has_track0 = np.abs(diak4.i0.eta) <= 2.5
        has_track1 = np.abs(diak4.i1.eta) <= 2.5

        leadak4_id = diak4.i0.tightId & (has_track0*((diak4.i0.chf > cfg.SELECTION.SIGNAL.LEADAK4.CHF) & (diak4.i0.nhf < cfg.SELECTION.SIGNAL.LEADAK4.NHF)) + ~has_track0)
        trailak4_id = has_track1*((diak4.i1.chf > cfg.SELECTION.SIGNAL.TRAILAK4.CHF) & (diak4.i1.nhf < cfg.SELECTION.SIGNAL.TRAILAK4.NHF)) + ~has_track1

        df['mjj'] = diak4.mass.max()
        df['dphijj'] = dphi(diak4.i0.phi.min(), diak4.i1.phi.max())
        df['detajj'] = np.abs(diak4.i0.eta - diak4.i1.eta).max()

        selection.add('two_jets', diak4.counts>0)
        selection.add('leadak4_pt_eta', leadak4_pt_eta.any())
        selection.add('trailak4_pt_eta', trailak4_pt_eta.any())
        selection.add('hemisphere', hemisphere)
        selection.add('leadak4_id',leadak4_id.any())
        selection.add('trailak4_id',trailak4_id.any())
        selection.add('mjj', df['mjj'] > cfg.SELECTION.SIGNAL.DIJET.SHAPE_BASED.MASS)
        selection.add('dphijj', df['dphijj'] < cfg.SELECTION.SIGNAL.DIJET.SHAPE_BASED.DPHI)
        selection.add('detajj', df['detajj'] > cfg.SELECTION.SIGNAL.DIJET.SHAPE_BASED.DETA)
        
        # Cleaning cuts for signal region
        max_neEmEF = np.maximum(diak4.i0.nef, diak4.i1.nef)
        selection.add('max_neEmEF', (max_neEmEF < 0.7).any())
        
        vec_b = calculate_vecB(ak4, met_pt, met_phi)
        vec_dphi = calculate_vecDPhi(ak4, met_pt, met_phi, df['TkMET_phi'])

        no_jet_in_trk = (diak4.i0.abseta>2.5).any() & (diak4.i1.abseta>2.5).any()
        no_jet_in_hf = (diak4.i0.abseta<3.0).any() & (diak4.i1.abseta<3.0).any()

        at_least_one_jet_in_hf = (diak4.i0.abseta>3.0).any() | (diak4.i1.abseta>3.0).any()
        at_least_one_jet_in_trk = (diak4.i0.abseta<2.5).any() | (diak4.i1.abseta<2.5).any()

        # Categorized cleaning cuts
        eemitigation = (
                        (no_jet_in_hf | at_least_one_jet_in_trk) & (vec_dphi < 1.0)
                    ) | (
                        (no_jet_in_trk & at_least_one_jet_in_hf) & (vec_b < 0.2)
                    )

        selection.add('eemitigation', eemitigation)

        # HF-HF veto in SR
        both_jets_in_hf = (diak4.i0.abseta > 3.0) & (diak4.i1.abseta > 3.0)
        selection.add('veto_hfhf', ~both_jets_in_hf.any())

        # Fill histograms
        output = self.accumulator.identity()

        # Only generator weights
        weights = processor.Weights(size=df.size, storeIndividual=True)
        weights.add('gen', df['Generator_weight'])

        weight = weights.partial_weight(exclude=[None])

        for region, cuts in regions.items():
            mask = selection.all(*cuts)

            def ezfill(name, **kwargs):
                """Helper function to make filling easier."""
                output[name].fill(
                                  dataset=dataset,
                                  region=region,
                                  **kwargs
                                  )
            
            # All ak4
            # This is a workaround to create a weight array of the right dimension
            w_alljets = weight_shape(ak4[mask].eta, weight[mask])

            ezfill('ak4_eta',    jeteta=ak4[mask].eta.flatten(), weight=w_alljets)
            ezfill('ak4_phi',    jetphi=ak4[mask].phi.flatten(), weight=w_alljets)
            ezfill('ak4_pt',     jetpt=ak4[mask].pt.flatten(),   weight=w_alljets)

            # Leading ak4
            w_diak4 = weight_shape(diak4.pt[mask], weight[mask])
            ezfill('ak4_eta0',      jeteta=diak4.i0.eta[mask].flatten(),    weight=w_diak4)
            ezfill('ak4_phi0',      jetphi=diak4.i0.phi[mask].flatten(),    weight=w_diak4)
            ezfill('ak4_pt0',       jetpt=diak4.i0.pt[mask].flatten(),      weight=w_diak4)
            
            # Trailing ak4
            ezfill('ak4_eta1',      jeteta=diak4.i1.eta[mask].flatten(),    weight=w_diak4)
            ezfill('ak4_phi1',      jetphi=diak4.i1.phi[mask].flatten(),    weight=w_diak4)
            ezfill('ak4_pt1',       jetpt=diak4.i1.pt[mask].flatten(),      weight=w_diak4)

            ezfill('met',                met=met_pt[mask],             weight=weight[mask] )
            ezfill('met_phi',            phi=met_phi[mask],            weight=weight[mask] )
            ezfill('gen_met',            met=df['GenMET_pt'][mask],    weight=weight[mask] )
            ezfill('gen_met_phi',        phi=df['GenMET_phi'][mask],   weight=weight[mask] )


        return output
    
    def postprocess(self, accumulator):
        return accumulator

