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
                              max_dphi_jet_met,
                              mt,
                              recoil,
                              weight_shape,
                              candidates_in_hem,
                              calculate_HT,
                              calculate_HTmiss,
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
                                           vbfhinv_regions
                                         )

def trigger_selection(selection, df, cfg):
    pass_all = np.zeros(df.size) == 0
    pass_none = ~pass_all
    dataset = df['dataset']

    if df['is_data']:
        selection.add('filt_met', mask_and(df, cfg.FILTERS.DATA))
    else:
        selection.add('filt_met', mask_and(df, cfg.FILTERS.MC))
    selection.add('trig_met', mask_or(df, cfg.TRIGGERS.MET))

    # Electron trigger overlap
    if df['is_data']:
        if "SinglePhoton" in dataset:
            # Backup photon trigger, but not main electron trigger
            trig_ele = mask_or(df, cfg.TRIGGERS.ELECTRON.SINGLE_BACKUP) & (~mask_or(df, cfg.TRIGGERS.ELECTRON.SINGLE))
        elif "SingleElectron" in dataset:
            # Main electron trigger, no check for backup
            trig_ele = mask_or(df, cfg.TRIGGERS.ELECTRON.SINGLE)
        elif "EGamma" in dataset:
            # 2018 has everything in one stream, so simple OR
            trig_ele = mask_or(df, cfg.TRIGGERS.ELECTRON.SINGLE_BACKUP) | mask_or(df, cfg.TRIGGERS.ELECTRON.SINGLE)
        else:
            trig_ele = pass_none
    else:
        trig_ele = mask_or(df, cfg.TRIGGERS.ELECTRON.SINGLE_BACKUP) | mask_or(df, cfg.TRIGGERS.ELECTRON.SINGLE)

    selection.add('trig_ele', trig_ele)

    # Photon trigger:
    if (not df['is_data']) or ('SinglePhoton' in dataset) or ('EGamma' in dataset):
        trig_photon = mask_or(df, cfg.TRIGGERS.PHOTON.SINGLE)
    else:
        trig_photon = pass_none
    selection.add('trig_photon', trig_photon)

    for trgname in cfg.TRIGGERS.HT.GAMMAEFF:
        if (not df['is_data']) or ('JetHT' in dataset):
            selection.add(trgname, mask_or(df,[trgname]))
        else:
            selection.add(trgname, np.ones(df.size)==1)

    # Muon trigger
    selection.add('trig_mu', mask_or(df, cfg.TRIGGERS.MUON.SINGLE))

    return selection






class vbfhinvProcessor(processor.ProcessorABC):
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
        if not df.size:
            return self.accumulator.identity()
        self._configure(df)
        dataset = df['dataset']
        df['is_lo_w'] = is_lo_w(dataset)
        df['is_lo_z'] = is_lo_z(dataset)
        df['is_lo_znunu'] = is_lo_znunu(dataset)
        df['is_lo_w_ewk'] = is_lo_w_ewk(dataset)
        df['is_lo_z_ewk'] = is_lo_z_ewk(dataset)
        df['is_lo_g'] = is_lo_g(dataset)
        df['is_nlo_z'] = is_nlo_z(dataset)
        df['is_nlo_w'] = is_nlo_w(dataset)
        df['has_lhe_v_pt'] = df['is_lo_w'] | df['is_lo_z'] | df['is_nlo_z'] | df['is_nlo_w'] | df['is_lo_g'] | df['is_lo_w_ewk'] | df['is_lo_z_ewk']
        df['is_data'] = is_data(dataset)

        gen_v_pt = None
        if df['is_lo_w'] or df['is_lo_z'] or df['is_nlo_z'] or df['is_nlo_w'] or df['is_lo_z_ewk'] or df['is_lo_w_ewk']:
            gen = setup_gen_candidates(df)
            dressed = setup_dressed_gen_candidates(df)
            fill_gen_v_info(df, gen, dressed)
            gen_v_pt = df['gen_v_pt_combined']
        elif df['is_lo_g']:
            gen = setup_gen_candidates(df)
            all_gen_photons = gen[(gen.pdg==22)]
            prompt_mask = (all_gen_photons.status==1)&(all_gen_photons.flag&1==1)
            stat1_mask = (all_gen_photons.status==1)
            gen_photons = all_gen_photons[prompt_mask | (~prompt_mask.any()) & stat1_mask ]
            gen_photon = gen_photons[gen_photons.pt.argmax()]

            gen_v_pt = gen_photon.pt.max()

        # Generator-level leading dijet mass
        if df['has_lhe_v_pt']:
            genjets = setup_lhe_cleaned_genjets(df)
            digenjet = genjets[:,:2].distincts()
            df['mjj_gen'] = digenjet.mass.max()
            df['mjj_gen'] = np.where(df['mjj_gen'] > 0, df['mjj_gen'], 0)


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

        # Muons
        df['is_tight_muon'] = muons.tightId \
                      & (muons.iso < cfg.MUON.CUTS.TIGHT.ISO) \
                      & (muons.pt>cfg.MUON.CUTS.TIGHT.PT) \
                      & (muons.abseta<cfg.MUON.CUTS.TIGHT.ETA)

        dimuons = muons.distincts()
        dimuon_charge = dimuons.i0['charge'] + dimuons.i1['charge']

        df['MT_mu'] = ((muons.counts==1) * mt(muons.pt, muons.phi, met_pt, met_phi)).max()

        # Electrons
        df['is_tight_electron'] = electrons.tightId \
                            & (electrons.pt > cfg.ELECTRON.CUTS.TIGHT.PT) \
                            & (electrons.absetasc < cfg.ELECTRON.CUTS.TIGHT.ETA)

        dielectrons = electrons.distincts()
        dielectron_charge = dielectrons.i0['charge'] + dielectrons.i1['charge']

        df['MT_el'] = ((electrons.counts==1) * mt(electrons.pt, electrons.phi, met_pt, met_phi)).max()

        # ak4
        leadak4_index=ak4.pt.argmax()

        elejet_pairs = ak4[:,:1].cross(electrons)
        df['dREleJet'] = np.hypot(elejet_pairs.i0.eta-elejet_pairs.i1.eta , dphi(elejet_pairs.i0.phi,elejet_pairs.i1.phi)).min()
        muonjet_pairs = ak4[:,:1].cross(muons)
        df['dRMuonJet'] = np.hypot(muonjet_pairs.i0.eta-muonjet_pairs.i1.eta , dphi(muonjet_pairs.i0.phi,muonjet_pairs.i1.phi)).min()

        # Recoil
        df['recoil_pt'], df['recoil_phi'] = recoil(met_pt,met_phi, electrons, muons, photons)

        df["dPFCaloSR"] = (met_pt - df["CaloMET_pt"]) / met_pt
        df["dPFCaloCR"] = (met_pt - df["CaloMET_pt"]) / df["recoil_pt"]

        df["dPFTkSR"] = (met_pt - df["TkMET_pt"]) / met_pt

        df["minDPhiJetRecoil"] = min_dphi_jet_met(ak4, df['recoil_phi'], njet=4, ptmin=30, etamax=5.0)
        df["minDPhiJetMet"] = min_dphi_jet_met(ak4, met_phi, njet=4, ptmin=30, etamax=5.0)
        df["maxDPhiJetMet"] = max_dphi_jet_met(ak4, met_phi, njet=4, ptmin=30, etamax=5.0)

        # Calculate additional jetMET quantities if they are specified in the config file
        if 'HT' in cfg.RUN.SAVE.VARIABLES:
            # Calculate HT for several cases, by default the jet filtering is pt>30 GeV
            df['HT_jetsInclusive'] = calculate_HT(ak4) 
            df['HT_jetsInHF'] = calculate_HT(ak4, selection='jetsInHF') 
            df['HT_jetsInBarrel'] = calculate_HT(ak4, selection='jetsInBarrel') 
            df['HT_jetsInEndcap'] = calculate_HT(ak4, selection='jetsInEndcap') 
            df['HT_jetsNotInEndcap'] = calculate_HT(ak4, selection='jetsNotInEndcap') 

        if 'HTmiss' in cfg.RUN.SAVE.VARIABLES:
            # Calculate HTmiss for several cases, by default the jet filtering is pt>30 GeV
            df['HTmiss_jetsInclusive_pt'], df['HTmiss_jetsInclusive_phi'] = calculate_HTmiss(ak4)
            df['HTmiss_jetsInHF_pt'], df['HTmiss_jetsInHF_phi'] = calculate_HTmiss(ak4, selection='jetsInHF') 
            df['HTmiss_jetsInBarrel_pt'], df['HTmiss_jetsInBarrel_phi'] = calculate_HTmiss(ak4, selection='jetsInBarrel') 
            df['HTmiss_jetsInEndcap_pt'], df['HTmiss_jetsInEndcap_phi'] = calculate_HTmiss(ak4, selection='jetsInEndcap') 
            df['HTmiss_jetsNotInEndcap_pt'], df['HTmiss_jetsNotInEndcap_phi'] = calculate_HTmiss(ak4, selection='jetsNotInEndcap') 

        selection = processor.PackedSelection()

        # Triggers
        pass_all = np.ones(df.size)==1
        selection.add('inclusive', pass_all)
        selection = trigger_selection(selection, df, cfg)

        selection.add('mu_pt_trig_safe', muons.pt.max() > 30)

        # Common selection
        selection.add('veto_ele', electrons.counts==0)
        selection.add('veto_muo', muons.counts==0)
        selection.add('veto_photon', photons.counts==0)
        selection.add('veto_tau', taus.counts==0)
        selection.add('at_least_one_tau', taus.counts>0)
        selection.add('veto_b', bjets.counts==0)
        selection.add('mindphijr',df['minDPhiJetRecoil'] > cfg.SELECTION.SIGNAL.MINDPHIJR)
        selection.add('mindphijm',df['minDPhiJetMet'] > cfg.SELECTION.SIGNAL.MINDPHIJR)

        selection.add('dpfcalo_sr',np.abs(df['dPFCaloSR']) < cfg.SELECTION.SIGNAL.DPFCALO)
        selection.add('dpfcalo_cr',np.abs(df['dPFCaloCR']) < cfg.SELECTION.SIGNAL.DPFCALO)

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

        # Delta phi between leading/trailing jet and MET
        df['dPhiLeadingJetMet'] = dphi(diak4.i0.phi.min(), df['recoil_phi'])
        df['dPhiTrailingJetMet'] = dphi(diak4.i1.phi.min(), df['recoil_phi'])

        # Calculate dphi between the more forward/central jet from the leading dijet pair and MET
        leading_jet_more_central = diak4.i0.abseta <= diak4.i1.abseta
        trailing_jet_more_central = diak4.i0.abseta > diak4.i1.abseta

        df['dPhiMoreCentralJetMet'] = leading_jet_more_central.any() * df['dPhiLeadingJetMet'] + trailing_jet_more_central.any() * df['dPhiTrailingJetMet'] 
        df['dPhiMoreForwardJetMet'] = trailing_jet_more_central.any() * df['dPhiLeadingJetMet'] + leading_jet_more_central.any() * df['dPhiTrailingJetMet'] 

        leading_jet_in_horn = ((diak4.i0.abseta<3.2) & (diak4.i0.abseta>2.8)).any()
        trailing_jet_in_horn = ((diak4.i1.abseta<3.2) & (diak4.i1.abseta>2.8)).any()

        selection.add('hornveto', (df['dPFTkSR'] < 0.8) | ~(leading_jet_in_horn | trailing_jet_in_horn))

        if df['year'] == 2018:
            selection.add("metphihemextveto", ((-1.8 > met_phi)|(met_phi>-0.6)))
            selection.add('no_el_in_hem', electrons[candidates_in_hem(electrons)].counts==0)
        else:
            selection.add("metphihemextveto", pass_all)
            selection.add('no_el_in_hem', pass_all)

        vec_b = calculate_vecB(ak4, met_pt, met_phi)
        vec_dphi = calculate_vecDPhi(ak4, met_pt, met_phi, df['TkMET_phi'])

        no_jet_in_trk = (diak4.i0.abseta>2.5).any() & (diak4.i1.abseta>2.5).any()
        no_jet_in_hf = (diak4.i0.abseta<3.0).any() & (diak4.i1.abseta<3.0).any()

        at_least_one_jet_in_hf = (diak4.i0.abseta>3.0).any() | (diak4.i1.abseta>3.0).any()
        at_least_one_jet_in_trk = (diak4.i0.abseta<2.5).any() | (diak4.i1.abseta<2.5).any()

        # Categorized cleaning cuts
        eemitigation_v1 = (
                        (no_jet_in_hf | at_least_one_jet_in_trk) & (vec_dphi < 1.0)
                    ) | (
                        (no_jet_in_trk & at_least_one_jet_in_hf) & (vec_b < 0.2)
                    )

        selection.add('eemitigation_v1', eemitigation_v1)

        # Another possible mitigation cut
        pt_over_met_leading_jet = diak4.i0.pt / met_pt
        eemitigation_v2 = (diak4.i0.pt > 100) & (diak4.i0.abseta > 2.9) & (diak4.i0.abseta < 3.3) \
                    & (dphi(diak4.i0.phi, met_phi) > 2.9) & (pt_over_met_leading_jet > 0.7) & (pt_over_met_leading_jet < 1.3)

        selection.add('eemitigation_v2', ~eemitigation_v2.any())

        eemitigation_v3 = (diak4.i0.pt > 100) & (diak4.i0.abseta > 2.9) & (diak4.i0.abseta < 3.3) \
                    & (dphi(diak4.i0.phi, met_phi) > 2.5)

        selection.add('eemitigation_v3', ~eemitigation_v3.any())

        # Selection to identify events that pass "v3" selection, but fail the "v1" one
        pass_v3_fail_v1 = (~eemitigation_v3.any()) & (~eemitigation_v1)
        selection.add('passv3_failv1', pass_v3_fail_v1)

        # Mask if both leading jets are in HF
        two_jets_in_hf = (diak4.i0.abseta>3.0) & (diak4.i1.abseta>3.0)
        selection.add('veto_hfhf', ~two_jets_in_hf.any())

        # Define cut on maximum of the neutral EM fraction of two leading jets
        max_neEmEF = np.maximum(diak4.i0.nef, diak4.i1.nef)
        selection.add('max_neEmEF', (max_neEmEF<0.7).any())

        selection.add('low_EmEF', (diak4.i0.nef<0.1).any())
        selection.add('high_EmEF', (diak4.i0.nef>0.9).any())
        selection.add('low_HEF', (diak4.i0.nhf<0.1).any())
        selection.add('high_HEF', (diak4.i0.nhf>0.9).any())

        selection.add('two_jets', diak4.counts>0)
        selection.add('leadak4_pt_eta', leadak4_pt_eta.any())
        selection.add('trailak4_pt_eta', trailak4_pt_eta.any())
        selection.add('hemisphere', hemisphere)
        selection.add('leadak4_id',leadak4_id.any())
        selection.add('trailak4_id',trailak4_id.any())
        selection.add('mjj', df['mjj'] > cfg.SELECTION.SIGNAL.DIJET.SHAPE_BASED.MASS)
        selection.add('dphijj', df['dphijj'] < cfg.SELECTION.SIGNAL.DIJET.SHAPE_BASED.DPHI)
        selection.add('detajj', df['detajj'] > cfg.SELECTION.SIGNAL.DIJET.SHAPE_BASED.DETA)

        # Divide into three categories for trigger study
        if cfg.RUN.TRIGGER_STUDY:
            two_central_jets = (np.abs(diak4.i0.eta) <= 2.4) & (np.abs(diak4.i1.eta) <= 2.4)
            two_forward_jets = (np.abs(diak4.i0.eta) > 2.4) & (np.abs(diak4.i1.eta) > 2.4)
            one_jet_forward_one_jet_central = (~two_central_jets) & (~two_forward_jets)
            selection.add('two_central_jets', two_central_jets.any())
            selection.add('two_forward_jets', two_forward_jets.any())
            selection.add('one_jet_forward_one_jet_central', one_jet_forward_one_jet_central.any())

        # Dimuon CR
        leadmuon_index=muons.pt.argmax()
        selection.add('at_least_one_tight_mu', df['is_tight_muon'].any())
        selection.add('dimuon_mass', ((dimuons.mass > cfg.SELECTION.CONTROL.DOUBLEMU.MASS.MIN) \
                                    & (dimuons.mass < cfg.SELECTION.CONTROL.DOUBLEMU.MASS.MAX)).any())
        selection.add('dimuon_charge', (dimuon_charge==0).any())
        selection.add('two_muons', muons.counts==2)

        # Single muon CR
        selection.add('one_muon', muons.counts==1)
        selection.add('mt_mu', df['MT_mu'] < cfg.SELECTION.CONTROL.SINGLEMU.MT)

        # Diele CR
        leadelectron_index=electrons.pt.argmax()


        selection.add('one_electron', electrons.counts==1)
        selection.add('two_electrons', electrons.counts==2)
        selection.add('at_least_one_tight_el', df['is_tight_electron'].any())


        selection.add('dielectron_mass', ((dielectrons.mass > cfg.SELECTION.CONTROL.DOUBLEEL.MASS.MIN)  \
                                        & (dielectrons.mass < cfg.SELECTION.CONTROL.DOUBLEEL.MASS.MAX)).any())
        selection.add('dielectron_charge', (dielectron_charge==0).any())

        # Single Ele CR
        selection.add('met_el', met_pt > cfg.SELECTION.CONTROL.SINGLEEL.MET)
        selection.add('mt_el', df['MT_el'] < cfg.SELECTION.CONTROL.SINGLEEL.MT)

        # Photon CR
        leadphoton_index=photons.pt.argmax()

        df['is_tight_photon'] = photons.mediumId & photons.barrel

        selection.add('one_photon', photons.counts==1)
        selection.add('at_least_one_tight_photon', df['is_tight_photon'].any())
        selection.add('photon_pt', photons.pt.max() > cfg.PHOTON.CUTS.TIGHT.PT)
        selection.add('photon_pt_trig', photons.pt.max() > cfg.PHOTON.CUTS.TIGHT.PTTRIG)

        # Fill histograms
        output = self.accumulator.identity()

        # Gen
        if df['has_lhe_v_pt']:
            output['genvpt_check'].fill(vpt=gen_v_pt,type="Nano", dataset=dataset)

        if 'LHE_Njets' in df:
            output['lhe_njets'].fill(dataset=dataset, multiplicity=df['LHE_Njets'])
        if 'LHE_HT' in df:
            output['lhe_ht'].fill(dataset=dataset, ht=df['LHE_HT'])
        if 'LHE_HTIncoming' in df:
            output['lhe_htinc'].fill(dataset=dataset, ht=df['LHE_HTIncoming'])

        # Weights
        evaluator = evaluator_from_config(cfg)

        weights = processor.Weights(size=df.size, storeIndividual=True)
        if not df['is_data']:
            weights.add('gen', df['Generator_weight'])

            try:
                weights.add('prefire', df['PrefireWeight'])
            except KeyError:
                weights.add('prefire', np.ones(df.size))

            weights = candidate_weights(weights, df, evaluator, muons, electrons, photons, cfg)
            weights = pileup_weights(weights, df, evaluator, cfg)
            if not (gen_v_pt is None):
                weights = theory_weights_vbf(weights, df, evaluator, gen_v_pt, df['mjj_gen'])

        # Save per-event values for synchronization
        if cfg.RUN.KINEMATICS.SAVE:
            for event in cfg.RUN.KINEMATICS.EVENTS:
                mask = df['event'] == event
                if not mask.any():
                    continue
                output['kinematics']['event'] += [event]
                output['kinematics']['met'] += [met_pt[mask]]
                output['kinematics']['met_phi'] += [met_phi[mask]]
                output['kinematics']['recoil'] += [df['recoil_pt'][mask]]
                output['kinematics']['recoil_phi'] += [df['recoil_phi'][mask]]

                output['kinematics']['ak4pt0'] += [ak4[leadak4_index][mask].pt]
                output['kinematics']['ak4eta0'] += [ak4[leadak4_index][mask].eta]
                output['kinematics']['leadbtag'] += [ak4.pt.max()<0][mask]

                output['kinematics']['nLooseMu'] += [muons.counts[mask]]
                output['kinematics']['nTightMu'] += [muons[df['is_tight_muon']].counts[mask]]
                output['kinematics']['mupt0'] += [muons[leadmuon_index][mask].pt]
                output['kinematics']['mueta0'] += [muons[leadmuon_index][mask].eta]

                output['kinematics']['nLooseEl'] += [electrons.counts[mask]]
                output['kinematics']['nTightEl'] += [electrons[df['is_tight_electron']].counts[mask]]
                output['kinematics']['elpt0'] += [electrons[leadelectron_index][mask].pt]
                output['kinematics']['eleta0'] += [electrons[leadelectron_index][mask].eta]

                output['kinematics']['nLooseGam'] += [photons.counts[mask]]
                output['kinematics']['nTightGam'] += [photons[df['is_tight_photon']].counts[mask]]
                output['kinematics']['gpt0'] += [photons[leadphoton_index][mask].pt]
                output['kinematics']['geta0'] += [photons[leadphoton_index][mask].eta]


        # Sum of all weights to use for normalization
        # TODO: Deal with systematic variations
        output['nevents'][dataset] += df.size
        if not df['is_data']:
            output['sumw'][dataset] +=  df['genEventSumw']
            output['sumw2'][dataset] +=  df['genEventSumw2']
            output['sumw_pileup'][dataset] +=  weights._weights['pileup'].sum()

        regions = vbfhinv_regions(cfg)

        veto_weights = get_veto_weights(df, evaluator, electrons, muons, taus)
        for region, cuts in regions.items():
            exclude = [None]
            region_weights = copy.deepcopy(weights)

            if not df['is_data']:
                ### Trigger weights
                if re.match(r'cr_(\d+)e.*', region):
                    p_pass_data = 1 - (1-evaluator["trigger_electron_eff_data"](electrons.etasc, electrons.pt)).prod()
                    p_pass_mc   = 1 - (1-evaluator["trigger_electron_eff_mc"](electrons.etasc, electrons.pt)).prod()
                    trigger_weight = p_pass_data/p_pass_mc
                    trigger_weight[np.isnan(trigger_weight)] = 1
                    region_weights.add('trigger', trigger_weight)
                elif re.match(r'cr_(\d+)m.*', region) or re.match('sr_.*', region):
                    region_weights.add('trigger_met', evaluator["trigger_met"](df['recoil_pt']))
                elif re.match(r'cr_g.*', region):
                    photon_trigger_sf(region_weights, photons, df)

                # Veto weights
                if re.match('.*no_veto.*', region):
                    exclude = [
                            "muon_id_tight",
                            "muon_iso_tight",
                            "muon_id_loose",
                            "muon_iso_loose",
                            "ele_reco",
                            "ele_id_tight",
                            "ele_id_loose",
                            "tau_id"
                        ]
                    region_weights.add("veto",veto_weights.partial_weight(include=["nominal"]))


            # This is the default weight for this region
            rweight = region_weights.partial_weight(exclude=exclude)

            # Blinding
            # if(self._blind and df['is_data'] and region.startswith('sr')):
                # continue

            # Cutflow plot for signal and control regions
            if any(x in region for x in ["sr", "cr", "tr"]):
                output['cutflow_' + region][dataset]['all']+=df.size
                for icut, cutname in enumerate(cuts):
                    output['cutflow_' + region][dataset][cutname] += selection.all(*cuts[:icut+1]).sum()

            mask = selection.all(*cuts)

            if cfg.RUN.SAVE.TREE:
                if re.match(cfg.RUN.SAVE.TREEREGIONS, region):
                    # General properties
                    output['tree_int64'][region]["event"]                   += processor.column_accumulator(df["event"][mask])
                    output['tree_int64'][region]["run"]                     += processor.column_accumulator(df["run"][mask])
                    output['tree_int64'][region]["lumi"]                    += processor.column_accumulator(df["luminosityBlock"][mask])

                    if region == 'inclusive':
                        for name in selection.names:
                            output['tree_bool'][region][name] += processor.column_accumulator(np.bool_(selection.all(*[name])[mask]))

                    # Save quantities for other regions 
                    else:
                        if cfg.RUN.PROCESS_UL:
                            met_pt_nojer = df['MET_pt']
                            met_phi_nojer = df['MET_phi']
                        else:
                            met_pt_nojer = df['MET_pt_nom' if df['year']==2018 else 'METFixEE2017_pt_nom']
                            met_phi_nojer = df['MET_phi_nom' if df['year']==2018 else 'METFixEE2017_phi_nom']

                        output['tree_float16'][region]["recoil_pt"]       +=  processor.column_accumulator(df["recoil_pt"][mask])
                        output['tree_float16'][region]["recoil_phi"]      +=  processor.column_accumulator(df["recoil_phi"][mask])
                        output['tree_float16'][region]["met_pt"]          +=  processor.column_accumulator(met_pt[mask])
                        output['tree_float16'][region]["met_phi"]         +=  processor.column_accumulator(met_phi[mask])
                        output['tree_float16'][region]["met_pt_nojer"]    +=  processor.column_accumulator(met_pt_nojer[mask])
                        output['tree_float16'][region]["met_phi_nojer"]   +=  processor.column_accumulator(met_phi_nojer[mask])
                        output['tree_float16'][region]["mjj"]             +=  processor.column_accumulator(df["mjj"][mask])
                        output['tree_float16'][region]["dphijj"]          +=  processor.column_accumulator(df["dphijj"][mask])
                        output['tree_float16'][region]["detajj"]          +=  processor.column_accumulator(df["detajj"][mask])
                        
                        # Save info about the cleaning cuts
                        output['tree_bool'][region]['pass_cleaning_cut']  += processor.column_accumulator(eemitigation_v1[mask])
                        output['tree_bool'][region]['two_jets_in_hf']     += processor.column_accumulator(two_jets_in_hf[mask])
                        output['tree_float16'][region]['vecB']     += processor.column_accumulator(vec_b[mask])            
                        output['tree_float16'][region]['vecDPhi']  += processor.column_accumulator(vec_dphi[mask])            

                        # Number of jets
                        good_jets = ak4[ak4.pt>30]
                        nJet = good_jets.counts
                        # Get the third jet for the events which have more than two jets 
                        # If there is no third jet with pt>30 GeV, the values will be set to -99
                        event_has_third_jet = nJet > 2
                        third_jet_pt  = np.hstack( np.where(event_has_third_jet, good_jets.pt[:,:3][:,-1:], [-99]) )
                        third_jet_eta = np.hstack( np.where(event_has_third_jet, good_jets.eta[:,:3][:,-1:], [-99]) )
                        third_jet_phi = np.hstack( np.where(event_has_third_jet, good_jets.phi[:,:3][:,-1:], [-99]) )

                        # Calculate and store delta phi between TrackMET and PF MET
                        dphi_tkMET_pfMET = dphi(met_phi, df['TkMET_phi'])

                        output['tree_float16'][region]["nJet"]            += processor.column_accumulator(nJet[mask])
                        output['tree_float16'][region]["thirdJet_pt"]     += processor.column_accumulator(third_jet_pt[mask])
                        output['tree_float16'][region]["thirdJet_eta"]    += processor.column_accumulator(third_jet_eta[mask])
                        output['tree_float16'][region]["thirdJet_phi"]    += processor.column_accumulator(third_jet_phi[mask])

                        # Information on leading jet in the event
                        output['tree_float16'][region]["leadak4_pt"]      +=  processor.column_accumulator(diak4.i0.pt.max()[mask])
                        output['tree_float16'][region]["leadak4_eta"]     +=  processor.column_accumulator(diak4.i0.eta.max()[mask])
                        output['tree_float16'][region]["leadak4_phi"]     +=  processor.column_accumulator(diak4.i0.phi.max()[mask])
                        output['tree_float16'][region]["leadak4_chHEF"]   +=  processor.column_accumulator(diak4.i0.chf.max()[mask])
                        output['tree_float16'][region]["leadak4_neHEF"]   +=  processor.column_accumulator(diak4.i0.nhf.max()[mask])
                        output['tree_float16'][region]["leadak4_neEmEF"]  +=  processor.column_accumulator(diak4.i0.nef.max()[mask])

                        # Information on trailing jet in the event
                        output['tree_float16'][region]["trailak4_pt"]     +=  processor.column_accumulator(diak4.i1.pt.max()[mask])
                        output['tree_float16'][region]["trailak4_eta"]    +=  processor.column_accumulator(diak4.i1.eta.max()[mask])
                        output['tree_float16'][region]["trailak4_phi"]    +=  processor.column_accumulator(diak4.i1.phi.max()[mask])
                        output['tree_float16'][region]["trailak4_chHEF"]  +=  processor.column_accumulator(diak4.i1.chf.max()[mask])
                        output['tree_float16'][region]["trailak4_neHEF"]  +=  processor.column_accumulator(diak4.i1.nhf.max()[mask])
                        output['tree_float16'][region]["trailak4_neEmEF"] +=  processor.column_accumulator(diak4.i1.nef.max()[mask])
        
                        output['tree_float16'][region]["mindPhiJetMet"]          +=  processor.column_accumulator(df["minDPhiJetMet"][mask])
                        output['tree_float16'][region]["maxdPhiJetMet"]          +=  processor.column_accumulator(df["maxDPhiJetMet"][mask])
                        output['tree_float16'][region]["dPhiLeadingJetMet"]      +=  processor.column_accumulator(df["dPhiLeadingJetMet"][mask])
                        output['tree_float16'][region]["dPhiTrailingJetMet"]     +=  processor.column_accumulator(df["dPhiTrailingJetMet"][mask])
                        output['tree_float16'][region]["dPhiMoreCentralJetMet"]  +=  processor.column_accumulator(df["dPhiMoreCentralJetMet"][mask])
                        output['tree_float16'][region]["dPhiMoreForwardJetMet"]  +=  processor.column_accumulator(df["dPhiMoreForwardJetMet"][mask])
                        output['tree_float16'][region]["CaloMET_pt"]             +=  processor.column_accumulator(df["CaloMET_pt"][mask])
                        output['tree_float16'][region]["CaloMET_phi"]            +=  processor.column_accumulator(df["CaloMET_phi"][mask])
                        output['tree_float16'][region]["TkMET_pt"]               +=  processor.column_accumulator(df["TkMET_pt"][mask])
                        output['tree_float16'][region]["TkMET_phi"]              +=  processor.column_accumulator(df["TkMET_phi"][mask])
                        
                        output['tree_float16'][region]["dPhi_TkMET_PFMET"]       +=  processor.column_accumulator(dphi_tkMET_pfMET[mask])

                        # HT and missing HT for jets
                        if 'HT' in cfg.RUN.SAVE.VARIABLES:
                            output['tree_float16'][region]["HT_jetsInclusive"]     +=  processor.column_accumulator(df["HT_jetsInclusive"][mask])
                            output['tree_float16'][region]["HT_jetsInHF"]          +=  processor.column_accumulator(df["HT_jetsInHF"][mask])
                            output['tree_float16'][region]["HT_jetsInBarrel"]      +=  processor.column_accumulator(df["HT_jetsInBarrel"][mask])
                            output['tree_float16'][region]["HT_jetsInEndcap"]      +=  processor.column_accumulator(df["HT_jetsInEndcap"][mask])
                            output['tree_float16'][region]["HT_jetsNotInEndcap"]   +=  processor.column_accumulator(df["HT_jetsNotInEndcap"][mask])

                        if 'HTmiss' in cfg.RUN.SAVE.VARIABLES:
                            output['tree_float16'][region]["HTmiss_jetsInclusive_pt"]     +=  processor.column_accumulator(df["HTmiss_jetsInclusive_pt"][mask])
                            output['tree_float16'][region]["HTmiss_jetsInclusive_phi"]    +=  processor.column_accumulator(df["HTmiss_jetsInclusive_phi"][mask])
                            output['tree_float16'][region]["HTmiss_jetsInHF_pt"]          +=  processor.column_accumulator(df["HTmiss_jetsInHF_pt"][mask])
                            output['tree_float16'][region]["HTmiss_jetsInHF_phi"]         +=  processor.column_accumulator(df["HTmiss_jetsInHF_phi"][mask])
                            output['tree_float16'][region]["HTmiss_jetsInBarrel_pt"]      +=  processor.column_accumulator(df["HTmiss_jetsInBarrel_pt"][mask])
                            output['tree_float16'][region]["HTmiss_jetsInBarrel_phi"]     +=  processor.column_accumulator(df["HTmiss_jetsInBarrel_phi"][mask])
                            output['tree_float16'][region]["HTmiss_jetsInEndcap_pt"]      +=  processor.column_accumulator(df["HTmiss_jetsInEndcap_pt"][mask])
                            output['tree_float16'][region]["HTmiss_jetsInEndcap_phi"]     +=  processor.column_accumulator(df["HTmiss_jetsInEndcap_phi"][mask])
                            output['tree_float16'][region]["HTmiss_jetsNotInEndcap_pt"]   +=  processor.column_accumulator(df["HTmiss_jetsNotInEndcap_pt"][mask])
                            output['tree_float16'][region]["HTmiss_jetsNotInEndcap_phi"]  +=  processor.column_accumulator(df["HTmiss_jetsNotInEndcap_phi"][mask])

                        if 'sumEt' in cfg.RUN.SAVE.VARIABLES:
                            if cfg.RUN.PROCESS_UL:
                                met_sumEt = df['MET_sumEt']
                            else:
                                met_sumEt = df['MET_sumEt' if df['year'] == 2018 else 'METFixEE2017_sumEt']
                            output['tree_float16'][region]['MET_sumEt'] += processor.column_accumulator(met_sumEt[mask])

                        output['tree_float16'][region]['PV_npvs']                += processor.column_accumulator(df["PV_npvs"][mask])
                        output['tree_float16'][region]['PV_npvsGood']            += processor.column_accumulator(df["PV_npvsGood"][mask])

                        # MC quantities
                        if not df['is_data']:
                            if gen_v_pt is not None:
                                output['tree_float16'][region]["gen_v_pt"]    +=  processor.column_accumulator(gen_v_pt[mask])
                                output['tree_float16'][region]["GenMET_pt"]   +=  processor.column_accumulator(df['GenMET_pt'][mask])
                                output['tree_float16'][region]["GenMET_phi"]  +=  processor.column_accumulator(df['GenMET_phi'][mask])
                            
                            # Information related to pileup
                            output['tree_float16'][region]['Pileup_nTrueInt']        += processor.column_accumulator(df["Pileup_nTrueInt"][mask])
                            output['tree_float16'][region]['Pileup_pudensity']       += processor.column_accumulator(df["Pileup_pudensity"][mask])
                            output['tree_float16'][region]['Pileup_gpudensity']      += processor.column_accumulator(df["Pileup_gpudensity"][mask])
                            output['tree_float16'][region]['Pileup_nPU']             += processor.column_accumulator(df["Pileup_nPU"][mask])
                            output['tree_float16'][region]['Pileup_sumEOOT']         += processor.column_accumulator(df["Pileup_sumEOOT"][mask])
                            output['tree_float16'][region]['Pileup_sumLOOT']         += processor.column_accumulator(df["Pileup_sumLOOT"][mask])
                            
                            if df['has_lhe_v_pt']:
                                output['tree_float16'][region]["gen_mjj"]     +=  processor.column_accumulator(df['mjj_gen'][mask])
                            
                            for name, w in region_weights._weights.items():
                                output['tree_float16'][region][f"weight_{name}"] += processor.column_accumulator(np.float16(w[mask]))
                    
                        # Single electron transverse mass
                        if re.match('.*_1e_.*', region):
                            output['tree_float16'][region]["mt"]   +=  processor.column_accumulator(df['MT_el'][mask])
                        # Single muon transverse mass
                        if re.match('.*_1m_.*', region):
                            output['tree_float16'][region]["mt"]   +=  processor.column_accumulator(df['MT_mu'][mask])
        
                        # Leading electron
                        if re.match('.*_(\d)e_.*', region):
                            output['tree_float16'][region]["ele0_pt"]        +=  processor.column_accumulator(electrons.pt[leadelectron_index][mask].max())
                            output['tree_float16'][region]["ele0_eta"]       +=  processor.column_accumulator(electrons.eta[leadelectron_index][mask].max())
                            output['tree_float16'][region]["ele0_phi"]       +=  processor.column_accumulator(electrons.phi[leadelectron_index][mask].max())
                            output['tree_float16'][region]["ele0_tightId"]   +=  processor.column_accumulator(electrons.tightId[leadelectron_index][mask].max())
                        # Trailing electron
                        if re.match('.*_2e_.*', region):
                            output['tree_float16'][region]["ele1_pt"]        +=  processor.column_accumulator(electrons.pt[~leadelectron_index][mask].max())
                            output['tree_float16'][region]["ele1_eta"]       +=  processor.column_accumulator(electrons.eta[~leadelectron_index][mask].max())
                            output['tree_float16'][region]["ele1_phi"]       +=  processor.column_accumulator(electrons.phi[~leadelectron_index][mask].max())
                            output['tree_float16'][region]["ele1_tightId"]   +=  processor.column_accumulator(electrons.tightId[~leadelectron_index][mask].max())
                        # Leading muon
                        if re.match('.*_(\d)m_.*', region):
                            output['tree_float16'][region]["mu0_pt"]        +=  processor.column_accumulator(muons.pt[leadmuon_index][mask].max())
                            output['tree_float16'][region]["mu0_eta"]       +=  processor.column_accumulator(muons.eta[leadmuon_index][mask].max())
                            output['tree_float16'][region]["mu0_phi"]       +=  processor.column_accumulator(muons.phi[leadmuon_index][mask].max())
                            output['tree_float16'][region]["mu0_tightId"]   +=  processor.column_accumulator(muons.tightId[leadmuon_index][mask].max())
                        # Trailing muon
                        if re.match('.*_2m_.*', region):
                            output['tree_float16'][region]["mu1_pt"]        +=  processor.column_accumulator(muons.pt[~leadmuon_index][mask].max())
                            output['tree_float16'][region]["mu1_eta"]       +=  processor.column_accumulator(muons.eta[~leadmuon_index][mask].max())
                            output['tree_float16'][region]["mu1_phi"]       +=  processor.column_accumulator(muons.phi[~leadmuon_index][mask].max())
                            output['tree_float16'][region]["mu1_tightId"]   +=  processor.column_accumulator(muons.tightId[~leadmuon_index][mask].max())
                        # Photon
                        if re.match('.*_g_.*', region):
                            output['tree_float16'][region]["photon_pt"]    +=  processor.column_accumulator(photons.pt[leadphoton_index][mask].max())
                            output['tree_float16'][region]["photon_eta"]   +=  processor.column_accumulator(photons.eta[leadphoton_index][mask].max())
                            output['tree_float16'][region]["photon_phi"]   +=  processor.column_accumulator(photons.phi[leadphoton_index][mask].max())
    
            # Save the event numbers of events passing this selection
            if cfg.RUN.SAVE.PASSING:
                output['selected_events'][region] += list(df['event'][mask])

            # Multiplicities
            def fill_mult(name, candidates):
                output[name].fill(
                                  dataset=dataset,
                                  region=region,
                                  multiplicity=candidates[mask].counts,
                                  weight=rweight[mask]
                                  )

            fill_mult('ak4_mult', ak4[ak4.pt>30])
            fill_mult('bjet_mult',bjets)
            fill_mult('loose_ele_mult',electrons)
            fill_mult('tight_ele_mult',electrons[df['is_tight_electron']])
            fill_mult('loose_muo_mult',muons)
            fill_mult('tight_muo_mult',muons[df['is_tight_muon']])
            fill_mult('tau_mult',taus)
            fill_mult('photon_mult',photons)

            def ezfill(name, **kwargs):
                """Helper function to make filling easier."""
                output[name].fill(
                                  dataset=dataset,
                                  region=region,
                                  **kwargs
                                  )

            # Monitor weights
            for wname, wvalue in region_weights._weights.items():
                ezfill("weights", weight_type=wname, weight_value=wvalue[mask])
                ezfill("weights_wide", weight_type=wname, weight_value=wvalue[mask])

            # All ak4
            # This is a workaround to create a weight array of the right dimension
            w_alljets = weight_shape(ak4[mask].eta, rweight[mask])
            w_alljets_nopref = weight_shape(ak4[mask].eta, region_weights.partial_weight(exclude=exclude+['prefire'])[mask])

            ezfill('ak4_eta',    jeteta=ak4[mask].eta.flatten(), weight=w_alljets)
            ezfill('ak4_phi',    jetphi=ak4[mask].phi.flatten(), weight=w_alljets)
            ezfill('ak4_pt',     jetpt=ak4[mask].pt.flatten(),   weight=w_alljets)

            ezfill('ak4_eta_nopref',    jeteta=ak4[mask].eta.flatten(), weight=w_alljets_nopref)
            ezfill('ak4_phi_nopref',    jetphi=ak4[mask].phi.flatten(), weight=w_alljets_nopref)
            ezfill('ak4_pt_nopref',     jetpt=ak4[mask].pt.flatten(),   weight=w_alljets_nopref)

            # Leading ak4
            w_diak4 = weight_shape(diak4.pt[mask], rweight[mask])
            ezfill('ak4_eta0',      jeteta=diak4.i0.eta[mask].flatten(),    weight=w_diak4)
            ezfill('ak4_phi0',      jetphi=diak4.i0.phi[mask].flatten(),    weight=w_diak4)
            ezfill('ak4_pt0',       jetpt=diak4.i0.pt[mask].flatten(),      weight=w_diak4)
            ezfill('ak4_ptraw0',    jetpt=diak4.i0.ptraw[mask].flatten(),   weight=w_diak4)
            ezfill('ak4_chf0',      frac=diak4.i0.chf[mask].flatten(),      weight=w_diak4)
            ezfill('ak4_nhf0',      frac=diak4.i0.nhf[mask].flatten(),      weight=w_diak4)
            ezfill('ak4_nef0',      frac=diak4.i0.nef[mask].flatten(),      weight=w_diak4)
            ezfill('ak4_nconst0',   nconst=diak4.i0.nconst[mask].flatten(), weight=w_diak4)

            # Trailing ak4
            ezfill('ak4_eta1',      jeteta=diak4.i1.eta[mask].flatten(),    weight=w_diak4)
            ezfill('ak4_phi1',      jetphi=diak4.i1.phi[mask].flatten(),    weight=w_diak4)
            ezfill('ak4_pt1',       jetpt=diak4.i1.pt[mask].flatten(),      weight=w_diak4)
            ezfill('ak4_ptraw1',    jetpt=diak4.i1.ptraw[mask].flatten(),   weight=w_diak4)
            ezfill('ak4_chf1',      frac=diak4.i1.chf[mask].flatten(),      weight=w_diak4)
            ezfill('ak4_nhf1',      frac=diak4.i1.nhf[mask].flatten(),      weight=w_diak4)
            ezfill('ak4_nef1',      frac=diak4.i1.nef[mask].flatten(),      weight=w_diak4)
            ezfill('ak4_nconst1',   nconst=diak4.i1.nconst[mask].flatten(), weight=w_diak4)

            ezfill('ak4_eta0_phi0', jeteta=diak4.i0.eta[mask].flatten(), jetphi=diak4.i0.phi[mask].flatten(), weight=w_diak4)
            ezfill('ak4_eta1_phi1', jeteta=diak4.i1.eta[mask].flatten(), jetphi=diak4.i1.phi[mask].flatten(), weight=w_diak4)

            if region in ['sr_vbf_leadak4_ee', 'sr_vbf_leadak4_ee_pt']:
                ezfill('ak4_pt0_over_met', jmet=pt_over_met_leading_jet[mask].flatten(), weight=w_diak4)
                ezfill('dphi_ak40_met', dphi=df['dPhiLeadingJetMet'][mask], weight=w_diak4)
                ezfill('dphi_ak40_met_ak4_pt0_over_met', 
                        dphi=df['dPhiLeadingJetMet'][mask], 
                        jmet=pt_over_met_leading_jet[mask].flatten(), 
                        weight=w_diak4)

            # B tag discriminator
            btag = getattr(ak4, cfg.BTAG.ALGO)
            w_btag = weight_shape(btag[mask], rweight[mask])
            ezfill('ak4_btag', btag=btag[mask].flatten(), weight=w_btag )

            # MET
            ezfill('dpfcalo_cr',            dpfcalo=df["dPFCaloCR"][mask],       weight=rweight[mask] )
            ezfill('dpfcalo_sr',            dpfcalo=df["dPFCaloSR"][mask],       weight=rweight[mask] )
            ezfill('met',                met=met_pt[mask],            weight=rweight[mask] )
            ezfill('met_phi',            phi=met_phi[mask],           weight=rweight[mask] )
            ezfill('recoil',             recoil=df["recoil_pt"][mask],      weight=rweight[mask] )
            ezfill('recoil_phi',         phi=df["recoil_phi"][mask],        weight=rweight[mask] )
            ezfill('dphijm',             dphi=df["minDPhiJetMet"][mask],    weight=rweight[mask] )
            ezfill('dphijr',             dphi=df["minDPhiJetRecoil"][mask], weight=rweight[mask] )

            ezfill('dphijj',             dphi=df["dphijj"][mask],   weight=rweight[mask] )
            ezfill('detajj',             deta=df["detajj"][mask],   weight=rweight[mask] )
            ezfill('mjj',                mjj=df["mjj"][mask],      weight=rweight[mask] )


            if gen_v_pt is not None:
                ezfill('gen_vpt', vpt=gen_v_pt[mask], weight=df['Generator_weight'][mask])
                ezfill('gen_mjj', mjj=df['mjj_gen'][mask], weight=df['Generator_weight'][mask])


            # Photon CR data-driven QCD estimate
            if df['is_data'] and re.match("cr_g.*", region) and re.match("(SinglePhoton|EGamma).*", dataset):
                w_imp = photon_impurity_weights(photons[leadphoton_index].pt.max()[mask], df["year"])
                output['mjj'].fill(
                                    dataset=data_driven_qcd_dataset(dataset),
                                    region=region,
                                    mjj=df["mjj"][mask],
                                    weight=rweight[mask] * w_imp
                                )
                output['recoil'].fill(
                                    dataset=data_driven_qcd_dataset(dataset),
                                    region=region,
                                    recoil=df["recoil_pt"][mask],
                                    weight=rweight[mask] * w_imp
                                )

            # Uncertainty variations
            if df['is_lo_z'] or df['is_nlo_z'] or df['is_lo_z_ewk']:
                theory_uncs = [x for x in cfg.SF.keys() if x.startswith('unc')]
                for unc in theory_uncs:
                    reweight = evaluator[unc](gen_v_pt)
                    w = (region_weights.weight() * reweight)[mask]
                    ezfill(
                        'mjj_unc',
                        mjj=df['mjj'][mask],
                        uncertainty=unc,
                        weight=w)

            # Two dimensional
            ezfill('recoil_mjj',         recoil=df["recoil_pt"][mask], mjj=df["mjj"][mask], weight=rweight[mask] )

            # Muons
            if '_1m_' in region or '_2m_' in region or 'no_veto' in region:
                w_allmu = weight_shape(muons.pt[mask], rweight[mask])
                ezfill('muon_pt',   pt=muons.pt[mask].flatten(),    weight=w_allmu )
                ezfill('muon_pt_abseta',pt=muons.pt[mask].flatten(),abseta=muons.eta[mask].flatten(),    weight=w_allmu )
                ezfill('muon_mt',   mt=df['MT_mu'][mask],           weight=rweight[mask])
                ezfill('muon_eta',  eta=muons.eta[mask].flatten(),  weight=w_allmu)
                ezfill('muon_phi',  phi=muons.phi[mask].flatten(),  weight=w_allmu)

            # Dimuon
            if '_2m_' in region:
                w_dimu = weight_shape(dimuons.pt[mask], rweight[mask])
                ezfill('muon_pt0',      pt=dimuons.i0.pt[mask].flatten(),           weight=w_dimu)
                ezfill('muon_pt1',      pt=dimuons.i1.pt[mask].flatten(),           weight=w_dimu)
                ezfill('muon_eta0',     eta=dimuons.i0.eta[mask].flatten(),         weight=w_dimu)
                ezfill('muon_eta1',     eta=dimuons.i1.eta[mask].flatten(),         weight=w_dimu)
                ezfill('muon_phi0',     phi=dimuons.i0.phi[mask].flatten(),         weight=w_dimu)
                ezfill('muon_phi1',     phi=dimuons.i1.phi[mask].flatten(),         weight=w_dimu)
                ezfill('dimuon_pt',     pt=dimuons.pt[mask].flatten(),              weight=w_dimu)
                ezfill('dimuon_eta',    eta=dimuons.eta[mask].flatten(),            weight=w_dimu)
                ezfill('dimuon_mass',   dilepton_mass=dimuons.mass[mask].flatten(), weight=w_dimu )

            # Electrons
            if '_1e_' in region or '_2e_' in region or 'no_veto' in region:
                w_allel = weight_shape(electrons.pt[mask], rweight[mask])
                ezfill('electron_pt',   pt=electrons.pt[mask].flatten(),    weight=w_allel)
                ezfill('electron_pt_eta',   pt=electrons.pt[mask].flatten(), eta=electrons.eta[mask].flatten(),    weight=w_allel)
                ezfill('electron_mt',   mt=df['MT_el'][mask],               weight=rweight[mask])
                ezfill('electron_eta',  eta=electrons.eta[mask].flatten(),  weight=w_allel)
                ezfill('electron_phi',  phi=electrons.phi[mask].flatten(),  weight=w_allel)

            # Dielectron
            if '_2e_' in region:
                w_diel = weight_shape(dielectrons.pt[mask], rweight[mask])
                ezfill('electron_pt0',      pt=dielectrons.i0.pt[mask].flatten(),               weight=w_diel)
                ezfill('electron_pt1',      pt=dielectrons.i1.pt[mask].flatten(),               weight=w_diel)
                ezfill('electron_eta0',     eta=dielectrons.i0.eta[mask].flatten(),             weight=w_diel)
                ezfill('electron_eta1',     eta=dielectrons.i1.eta[mask].flatten(),             weight=w_diel)
                ezfill('electron_phi0',     phi=dielectrons.i0.phi[mask].flatten(),             weight=w_diel)
                ezfill('electron_phi1',     phi=dielectrons.i1.phi[mask].flatten(),             weight=w_diel)
                ezfill('dielectron_pt',     pt=dielectrons.pt[mask].flatten(),                  weight=w_diel)
                ezfill('dielectron_eta',    eta=dielectrons.eta[mask].flatten(),                weight=w_diel)
                ezfill('dielectron_mass',   dilepton_mass=dielectrons.mass[mask].flatten(),     weight=w_diel)

            # Photon
            if '_g_' in region:
                w_leading_photon = weight_shape(photons[leadphoton_index].pt[mask],rweight[mask]);
                ezfill('photon_pt0',              pt=photons[leadphoton_index].pt[mask].flatten(),    weight=w_leading_photon)
                ezfill('photon_eta0',             eta=photons[leadphoton_index].eta[mask].flatten(),  weight=w_leading_photon)
                ezfill('photon_phi0',             phi=photons[leadphoton_index].phi[mask].flatten(),  weight=w_leading_photon)
                ezfill('photon_pt0_recoil',       pt=photons[leadphoton_index].pt[mask].flatten(), recoil=df['recoil_pt'][mask&(leadphoton_index.counts>0)],  weight=w_leading_photon)
                ezfill('photon_eta_phi',          eta=photons[leadphoton_index].eta[mask].flatten(), phi=photons[leadphoton_index].phi[mask].flatten(),  weight=w_leading_photon)

                # w_drphoton_jet = weight_shape(df['dRPhotonJet'][mask], rweight[mask])

            # Tau
            if 'no_veto' in region:
                w_all_taus = weight_shape(taus.pt[mask], rweight[mask])
                ezfill("tau_pt", pt=taus.pt[mask].flatten(), weight=w_all_taus)

            # PV
            ezfill('npv', nvtx=df['PV_npvs'][mask], weight=rweight[mask])
            ezfill('npvgood', nvtx=df['PV_npvsGood'][mask], weight=rweight[mask])

            ezfill('npv_nopu', nvtx=df['PV_npvs'][mask], weight=region_weights.partial_weight(exclude=exclude+['pileup'])[mask])
            ezfill('npvgood_nopu', nvtx=df['PV_npvsGood'][mask], weight=region_weights.partial_weight(exclude=exclude+['pileup'])[mask])

            ezfill('rho_all', rho=df['fixedGridRhoFastjetAll'][mask], weight=region_weights.partial_weight(exclude=exclude)[mask])
            ezfill('rho_central', rho=df['fixedGridRhoFastjetCentral'][mask], weight=region_weights.partial_weight(exclude=exclude)[mask])
            ezfill('rho_all_nopu', rho=df['fixedGridRhoFastjetAll'][mask], weight=region_weights.partial_weight(exclude=exclude+['pileup'])[mask])
            ezfill('rho_central_nopu', rho=df['fixedGridRhoFastjetCentral'][mask], weight=region_weights.partial_weight(exclude=exclude+['pileup'])[mask])
        return output

    def postprocess(self, accumulator):
        return accumulator
