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
                              calculate_z_pt_eta_phi
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
                                      is_lo_znunu,
                                      is_signal
                                      )

from bucoffea.monojet.definitions import (
                                          candidate_weights,
                                          pileup_weights,
                                          setup_candidates,
                                          theory_weights_monojet
                                          )
from bucoffea.helpers.gen import (
                                  setup_gen_candidates,
                                  setup_dressed_gen_candidates,
                                  fill_gen_v_info
                                 )

from bucoffea.zmumu.definitions import zmumu_accumulator, zmumu_regions

def add_selections_for_leading_jet(selection, lead_ak4):
    lead_ak4_pt_eta = (lead_ak4.pt > 100) & (np.abs(lead_ak4.eta) < 4.7)
    selection.add('lead_ak4_pt_eta', lead_ak4_pt_eta.any())

    has_track = np.abs(lead_ak4.eta) <= 2.5
    leadak4_id = lead_ak4.tightId & (has_track * ((lead_ak4.chf > 0.1) & (lead_ak4.nhf < 0.8)) + ~has_track)
    selection.add('lead_ak4_id', leadak4_id.any())

    # Neutral EM energy fraction cut on the jet
    selection.add('ak4_neEmEF', (lead_ak4.nef < 0.7).any())

    return selection

def add_muon_selections(df, selection, dimuons, leadak4, met_pt, ak4):
    '''The set of selections to be used for selecting Z(mumu) + jet events.'''
    # Calculate Z pt and phi from the dimuons
    z_pt, z_eta, z_phi = calculate_z_pt_eta_phi(dimuons)

    z_pt_eta = (z_pt > 100) & (np.abs(z_eta) < 4.7)
    selection.add('z_pt_eta', z_pt_eta.any())

    df['dphi_z_jet'] = dphi(z_phi.min(), leadak4.phi.max())
    selection.add('dphi_z_jet', df['dphi_z_jet'] > 2.7)

    # Add balance cut for the pt of Z and the jet
    df['z_pt_over_jet_pt'] = (z_pt.max() / leadak4.pt.max()) - 1
    selection.add('z_pt_over_jet_pt', np.abs(df['z_pt_over_jet_pt']) < 0.15)
    
    # Tighter version of the same balance cut
    selection.add('z_pt_over_jet_pt_tight', np.abs(df['z_pt_over_jet_pt']) < 0.1)
    selection.add('z_pt_over_jet_pt_very_tight', np.abs(df['z_pt_over_jet_pt']) < 0.05)

    selection.add('met_pt', met_pt < 50)

    return selection

def add_trigger_selection(df, selection):
    # Add in the single muon trigger selection
    single_mu_trig = 'HLT_IsoMu27'
    selection.add('single_mu_trig', df[single_mu_trig])

    return selection

class zmumuProcessor(processor.ProcessorABC):
    def __init__(self):
        self._year=None
        self._configure()
        self._accumulator = zmumu_accumulator(cfg)

    @property
    def accumulator(self):
        return self._accumulator

    def _configure(self, df=None):
        cfg.DYNACONF_WORKS="merge_configs"
        cfg.MERGE_ENABLED_FOR_DYNACONF=True
        cfg.SETTINGS_FILE_FOR_DYNACONF = bucoffea_path("config/monojet.yaml")

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
        if not df.size:
            return self.accumulator.identity()
        self._configure(df)
        dataset = df['dataset']
        df['is_data'] = is_data(dataset)
        df['is_lo_w'] = is_lo_w(dataset)
        df['is_lo_z'] = is_lo_z(dataset)
        df['is_nlo_z'] = is_nlo_z(dataset)
        df['is_nlo_w'] = is_nlo_w(dataset)
        df['is_lo_znunu'] = is_lo_znunu(dataset)

        # Pre-filtered candidates
        met_pt, met_phi, ak4, bjets, _, muons, electrons, taus, photons = setup_candidates(df, cfg)

        # Remove jets in accordance with the noise recipe
        if df['year'] == 2017:
            ak4   = ak4[(ak4.ptraw>50) | (ak4.abseta<2.65) | (ak4.abseta>3.139)]
            bjets = bjets[(bjets.ptraw>50) | (bjets.abseta<2.65) | (bjets.abseta>3.139)]

        # Filtering ak4 jets according to pileup ID
        ak4 = ak4[ak4.puid]

        gen_v_pt = None
        if not df['is_data']:
            gen = setup_gen_candidates(df)
        if df['is_lo_w'] or df['is_lo_z'] or df['is_nlo_z'] or df['is_nlo_w']:
            dressed = setup_dressed_gen_candidates(df)
            fill_gen_v_info(df, gen, dressed)
            gen_v_pt = df['gen_v_pt_combined']

        df['recoil_pt'], df['recoil_phi'] = recoil(met_pt, met_phi, electrons, muons, photons)

        # Muons
        df['is_tight_muon'] = muons.tightId \
                      & (muons.iso < cfg.MUON.CUTS.TIGHT.ISO) \
                      & (muons.pt > cfg.MUON.CUTS.TIGHT.PT) \
                      & (muons.abseta < cfg.MUON.CUTS.TIGHT.ETA)

        # Electrons
        df['is_tight_electron'] = electrons.tightId \
                            & (electrons.pt > cfg.ELECTRON.CUTS.TIGHT.PT) \
                            & (electrons.absetasc < cfg.ELECTRON.CUTS.TIGHT.ETA)

        df['is_tight_photon'] = photons.mediumId & photons.barrel
        
        selection = processor.PackedSelection()

        # HEM veto for 2018
        pass_all = np.ones(df.size)==1
        if df['year'] == 2018:
            selection.add('hemveto', df['hemveto'])
        else:
            selection.add('hemveto', pass_all)

        # Electron and b vetoes for 2m CR
        selection.add('veto_ele', electrons.counts==0)
        selection.add('veto_b', bjets.counts==0)

        # Selections for leading jet
        leadak4_index = ak4.pt.argmax()
        leadak4 = ak4[leadak4_index]
        selection = add_selections_for_leading_jet(selection, leadak4)

        selection = add_trigger_selection(df, selection)

        # Selections for dimuons
        dimuons = muons.distincts()
        selection = add_muon_selections(df, selection, dimuons, leadak4, met_pt, ak4)

        leadmuon_index=muons.pt.argmax()
        
        selection.add('at_least_one_tight_mu', df['is_tight_muon'].any())
        selection.add('dimuon_mass', ((dimuons.mass > cfg.SELECTION.CONTROL.DOUBLEMU.MASS.MIN) \
                                    & (dimuons.mass < cfg.SELECTION.CONTROL.DOUBLEMU.MASS.MAX)).any())

        selection.add('dimuon_mass_tight', ((dimuons.mass > 75) \
                                    & (dimuons.mass < 105)).any())

        dimuon_charge = dimuons.i0['charge'] + dimuons.i1['charge']
        selection.add('dimuon_charge', (dimuon_charge==0).any())

        selection.add('two_muons', muons.counts==2)
        selection.add('mu_pt_trig_safe', muons.pt.max() > 30)

        # Eta cuts for the leading jet
        selection.add('jet_eta_lt_2_3', (leadak4.abseta < 2.3).any())
        selection.add('jet_eta_gt_2_3_lt_2_7', ((leadak4.abseta >= 2.3) & (leadak4.abseta < 2.7)).any() )
        selection.add('jet_eta_gt_2_7_lt_3_0', ((leadak4.abseta >= 2.7) & (leadak4.abseta < 3.0)).any() )
        selection.add('jet_eta_gt_3_0', (leadak4.abseta >= 3.0).any())

        # Start to fill output
        output = self.accumulator.identity()
        
        # Weights
        evaluator = evaluator_from_config(cfg)
        weights = processor.Weights(size=df.size, storeIndividual=True)

        if not df['is_data']:
            weights.add('gen', df['Generator_weight'])

            weights = candidate_weights(weights, df, evaluator, muons, electrons, photons, cfg)
            weights = pileup_weights(weights, df, evaluator, cfg)
            if not (gen_v_pt is None):
                weights = theory_weights_monojet(weights, df, evaluator, gen_v_pt)

        if not df['is_data']:
            output['sumw'][dataset] +=  df['genEventSumw']
            output['sumw2'][dataset] +=  df['genEventSumw2']
            output['sumw_pileup'][dataset] +=  weights._weights['pileup'].sum()

        regions = zmumu_regions(cfg)

        for region, cuts in regions.items():
            exclude = [None]
            region_weights = copy.deepcopy(weights)

            if not df['is_data']:
                # For specific regions, include prefire weight up/down variations, instead of the central value
                if re.match('^.*EmEF.*Up$', region):
                    try:
                        region_weights.add('prefire', df['PrefireWeight_Up'])
                    except KeyError:
                        region_weights.add('prefire', np.ones(df.size))
                    
                elif re.match('^.*EmEF.*Down$', region):
                    try:
                        region_weights.add('prefire', df['PrefireWeight_Down'])
                    except KeyError:
                        region_weights.add('prefire', np.ones(df.size))
                    
                # Apply regular prefire weights to other regions
                else:
                    try:
                        region_weights.add('prefire', df['PrefireWeight'])
                    except KeyError:
                        region_weights.add('prefire', np.ones(df.size))

            mask = selection.all(*cuts)

            def ezfill(name, **kwargs):
                """Helper function to make filling easier."""
                if not ('dataset' in kwargs):
                    kwargs['dataset'] = dataset
                output[name].fill(
                                  region=region,
                                  **kwargs
                                  )
            # Leading ak4
            rweight = region_weights.partial_weight(exclude=exclude)
            w_leadak4 = weight_shape(leadak4.eta[mask], rweight[mask])

            ezfill('ak4_eta0',   jeteta=leadak4.eta[mask].flatten(),    weight=w_leadak4)
            ezfill('ak4_phi0',   jetphi=leadak4.phi[mask].flatten(),    weight=w_leadak4)
            ezfill('ak4_pt0',    jetpt=leadak4.pt[mask].flatten(),      weight=w_leadak4)
            ezfill('ak4_chf0',    frac=leadak4.chf[mask].flatten(),     weight=w_leadak4)
            ezfill('ak4_nhf0',    frac=leadak4.nhf[mask].flatten(),     weight=w_leadak4)
            ezfill('ak4_nef0',    frac=leadak4.nef[mask].flatten(),     weight=w_leadak4)

            ezfill('dphi_z_jet',  dphi=df['dphi_z_jet'][mask], weight=rweight[mask])
            ezfill('dimuon_mass', dilepton_mass=dimuons.mass[mask].flatten(), weight=rweight[mask])

            ezfill('met',                met=met_pt[mask],            weight=rweight[mask] )
            ezfill('met_phi',            phi=met_phi[mask],           weight=rweight[mask] )
            ezfill('recoil',             recoil=df['recoil_pt'][mask],      weight=rweight[mask] )
            ezfill('recoil_phi',         phi=df['recoil_phi'][mask],        weight=rweight[mask] )

            # Leading muon
            w_dimu = weight_shape(dimuons.pt[mask], rweight[mask])
            ezfill('muon_pt0',      pt=dimuons.i0.pt[mask].flatten(),           weight=w_dimu)
            ezfill('muon_pt1',      pt=dimuons.i1.pt[mask].flatten(),           weight=w_dimu)
            ezfill('muon_eta0',     eta=dimuons.i0.eta[mask].flatten(),         weight=w_dimu)
            ezfill('muon_eta1',     eta=dimuons.i1.eta[mask].flatten(),         weight=w_dimu)
            ezfill('muon_phi0',     phi=dimuons.i0.phi[mask].flatten(),         weight=w_dimu)
            ezfill('muon_phi1',     phi=dimuons.i1.phi[mask].flatten(),         weight=w_dimu)
        
        return output

    def postprocess(self, accumulator):
        return accumulator
