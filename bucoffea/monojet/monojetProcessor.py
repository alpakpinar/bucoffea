import copy
import re

import numpy as np

import coffea.processor as processor

from dynaconf import settings as cfg

from bucoffea.monojet.definitions import (
                                          monojet_accumulator,
                                          setup_candidates,
                                          monojet_regions,
                                          theory_weights_monojet,
                                          pileup_weights,
                                          candidate_weights,
                                          photon_trigger_sf,
                                          photon_impurity_weights,
                                          data_driven_qcd_dataset
                                         )
from bucoffea.helpers import (
                              min_dphi_jet_met,
                              recoil,
                              mt,
                              weight_shape,
                              bucoffea_path,
                              dphi,
                              mask_and,
                              mask_or,
                              evaluator_from_config,
                              calculate_v_pt_from_dilepton
                             )

from bucoffea.helpers.dataset import (
                                      is_lo_z,
                                      is_lo_znunu,
                                      is_lo_dy,
                                      is_lo_w,
                                      is_lo_g,
                                      is_nlo_z,
                                      is_nlo_w,
                                      has_v_jet,
                                      is_data,
                                      extract_year
                                     )
from bucoffea.helpers.gen import (
                                  setup_gen_candidates,
                                  setup_dressed_gen_candidates,
                                  fill_gen_v_info
                                 )

def trigger_selection(selection, df, cfg):
    pass_all = np.zeros(df.size) == 0
    pass_none = ~pass_all
    dataset = df['dataset']
    if cfg.RUN.SYNC: # Synchronization mode
        selection.add('filt_met', pass_all)
        selection.add('trig_met', pass_all)
        selection.add('trig_ele', pass_all)
        selection.add('trig_mu',  pass_all)
        selection.add('trig_photon',  pass_all)

    else:
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

class monojetProcessor(processor.ProcessorABC):
    def __init__(self, blind=True):
        self._year=None
        self._blind=blind
        self._configure()

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
        # All the split JES uncertainties, "" represents the nominal case with no variation
        self._variations = ['', '_jerUp', '_jerDown',
                            '_jesTotalUp', '_jesTotalDown',
                            # '_unclustEnUp', '_unclustEnDown'
                            ]
        self._accumulator = monojet_accumulator(cfg, variations=self._variations)

    def process(self, df):
        if not df.size:
            return self.accumulator.identity()
        self._configure(df)
        dataset = df['dataset']
        df['is_lo_w'] = is_lo_w(dataset)
        df['is_lo_z'] = is_lo_z(dataset)
        df['is_lo_znunu'] = is_lo_znunu(dataset)
        df['is_lo_g'] = is_lo_g(dataset)
        df['is_nlo_z'] = is_nlo_z(dataset)
        df['is_nlo_w'] = is_nlo_w(dataset)
        df['has_v_jet'] = has_v_jet(dataset)
        df['has_lhe_v_pt'] = df['is_lo_w'] | df['is_lo_z'] | df['is_nlo_z'] | df['is_nlo_w'] | df['is_lo_g']
        df['is_data'] = is_data(dataset)
        
        if df['is_data']:
            return self.accumulator.identity()

        gen_v_pt = None
        if not df['is_data']:
            gen = setup_gen_candidates(df)
        if df['is_lo_w'] or df['is_lo_z'] or df['is_nlo_z'] or df['is_nlo_w']:
            dressed = setup_dressed_gen_candidates(df)
            fill_gen_v_info(df, gen, dressed)
            gen_v_pt = df['gen_v_pt_combined']
        elif df['is_lo_g']:
            gen_v_pt = gen[(gen.pdg==22) & (gen.status==1)].pt.max()

        # Candidates
        # Already pre-filtered!
        # All leptons are at least loose
        # Check out setup_candidates for filtering details
        vmap, muons, electrons, taus, photons = setup_candidates(df, cfg, variations=self._variations)

        # vmap holds information about ak4, met and selection
        # packers for each JES/JER variation 
        # Check out monojet/definitions.py for the object definition

        #################################
        # First process the part which is 
        # unrelated to JES/JER variations
        #################################
        
        # Muons
        df['is_tight_muon'] = muons.tightId \
                      & (muons.iso < cfg.MUON.CUTS.TIGHT.ISO) \
                      & (muons.pt>cfg.MUON.CUTS.TIGHT.PT) \
                      & (muons.abseta<cfg.MUON.CUTS.TIGHT.ETA)

        dimuons = muons.distincts()
        dimuon_charge = dimuons.i0['charge'] + dimuons.i1['charge']

        # Electrons
        df['is_tight_electron'] = electrons.tightId \
                            & (electrons.pt > cfg.ELECTRON.CUTS.TIGHT.PT) \
                            & (electrons.absetasc < cfg.ELECTRON.CUTS.TIGHT.ETA)

        dielectrons = electrons.distincts()
        dielectron_charge = dielectrons.i0['charge'] + dielectrons.i1['charge']

        # Selection packer for the nominal (no varition) case
        selection_nom = processor.PackedSelection() 

        # Triggers
        pass_all = np.ones(df.size)==1
        selection_nom.add('inclusive', pass_all)
        selection_nom = trigger_selection(selection_nom, df, cfg)

        selection_nom.add('mu_pt_trig_safe', muons.pt.max() > 30)

        # Common selection
        selection_nom.add('veto_ele', electrons.counts==0)
        selection_nom.add('veto_muo', muons.counts==0)
        selection_nom.add('veto_photon', photons.counts==0)
        selection_nom.add('veto_tau', taus.counts==0)

        if (cfg.MITIGATION.HEM and extract_year(df['dataset']) == 2018 and not cfg.RUN.SYNC):
            selection_nom.add('hemveto', df['hemveto'])
        else:
            selection_nom.add('hemveto', np.ones(df.size)==1)

        # Single muon CR
        selection_nom.add('one_muon', muons.counts==1)

        # Dimuon CR
        leadmuon_index=muons.pt.argmax()
        selection_nom.add('at_least_one_tight_mu', df['is_tight_muon'].any())
        selection_nom.add('dimuon_mass', ((dimuons.mass > cfg.SELECTION.CONTROL.DOUBLEMU.MASS.MIN) \
                                    & (dimuons.mass < cfg.SELECTION.CONTROL.DOUBLEMU.MASS.MAX)).any())
        selection_nom.add('dimuon_charge', (dimuon_charge==0).any())
        selection_nom.add('two_muons', muons.counts==2)

        # Diele CR
        leadelectron_index=electrons.pt.argmax()

        selection_nom.add('one_electron', electrons.counts==1)
        selection_nom.add('two_electrons', electrons.counts==2)
        selection_nom.add('at_least_one_tight_el', df['is_tight_electron'].any())

        selection_nom.add('dielectron_mass', ((dielectrons.mass > cfg.SELECTION.CONTROL.DOUBLEEL.MASS.MIN)  \
                                        & (dielectrons.mass < cfg.SELECTION.CONTROL.DOUBLEEL.MASS.MAX)).any())
        selection_nom.add('dielectron_charge', (dielectron_charge==0).any())
        selection_nom.add('two_electrons', electrons.counts==2)

        # Photon CR
        leadphoton_index=photons.pt.argmax()
        df['is_tight_photon'] = photons.mediumId & photons.barrel

        selection_nom.add('one_photon', photons.counts==1)
        selection_nom.add('at_least_one_tight_photon', df['is_tight_photon'].any())
        selection_nom.add('photon_pt', photons.pt.max() > cfg.PHOTON.CUTS.TIGHT.PT)
        selection_nom.add('photon_pt_trig', photons.pt.max() > cfg.PHOTON.CUTS.TIGHT.PTTRIG)

        # Set the nominal selection packer, other selection packers (for variated cases)
        # will copy the common selections from this one
        vmap.set_selection_packer(var='', sel=selection_nom)

        # Temp selection object to be used later for other variations
        selection_nom_to_copy = copy.deepcopy(selection_nom)

        # Process for each JES/JER variation
        for var in self._variations:
            # Get the correct objects/quantities for each variation
            # For other variations, copy the common selections and
            # add on top of those.
            if var == '':
                selection = vmap.get_selection_packer(var='')
            else:
                selection = copy.deepcopy(selection_nom_to_copy) 
                vmap.set_selection_packer(var=var, sel=selection)    

            bjets = vmap.get_bjets(var)
            ak4 = vmap.get_ak4(var)
            if cfg.RUN.MONOV: 
                ak8 = vmap.get_ak8(var) 
            met = vmap.get_met(var) 

            met_pt = getattr(met, f'pt{var}').flatten()
            met_phi = getattr(met, f'phi{var}').flatten()

            selection.add(f'veto_b{var}', bjets.counts==0)
            
            # Get relevant pts for AK4 and AK8 jets
            ak4_pt = getattr(ak4, f'pt{var}')
            if cfg.RUN.MONOV: 
                ak8_pt = getattr(ak8, f'pt{var}')

            # MT requirements for muons and electrons
            df[f'MT_mu{var}'] = ((muons.counts==1) * mt(muons.pt, muons.phi, met_pt, met_phi)).max()
            selection.add(f'mt_mu{var}', df[f'MT_mu{var}'] < cfg.SELECTION.CONTROL.SINGLEMU.MT)
            df[f'MT_el{var}'] = ((electrons.counts==1) * mt(electrons.pt, electrons.phi, met_pt, met_phi)).max()
            selection.add(f'mt_el{var}', df[f'MT_el{var}'] < cfg.SELECTION.CONTROL.SINGLEEL.MT)
            
            # MET requirement in single electrion region
            selection.add(f'met_el{var}', met_pt > cfg.SELECTION.CONTROL.SINGLEEL.MET)

            elejet_pairs = ak4[:,:1].cross(electrons)
            df[f'dREleJet{var}'] = np.hypot(elejet_pairs.i0.eta-elejet_pairs.i1.eta , dphi(elejet_pairs.i0.phi,elejet_pairs.i1.phi)).min()
            muonjet_pairs = ak4[:,:1].cross(muons)
            df[f'dRMuonJet{var}'] = np.hypot(muonjet_pairs.i0.eta-muonjet_pairs.i1.eta , dphi(muonjet_pairs.i0.phi,muonjet_pairs.i1.phi)).min()

            # Recoil
            df[f"recoil_pt{var}"], df[f'recoil_phi{var}'] = recoil(met_pt,met_phi, electrons, muons, photons)
            df[f"dPFCalo{var}"] = (met_pt - df["CaloMET_pt"]) / df[f"recoil_pt{var}"]
            df[f"minDPhiJetRecoil{var}"] = min_dphi_jet_met(ak4, df[f"recoil_phi{var}"], njet=4, ptmin=30, etamax=2.4)
            df[f"minDPhiJetMet{var}"] = min_dphi_jet_met(ak4, met_phi, njet=4, ptmin=30, etamax=2.4)

            selection.add(f'mindphijr{var}',df[f'minDPhiJetRecoil{var}'] > cfg.SELECTION.SIGNAL.MINDPHIJR)
            selection.add(f'mindphijm{var}',df[f'minDPhiJetMet{var}'] > cfg.SELECTION.SIGNAL.MINDPHIJR)
            selection.add(f'dpfcalo{var}',np.abs(df[f'dPFCalo{var}']) < cfg.SELECTION.SIGNAL.DPFCALO)
            selection.add(f'recoil{var}', df[f'recoil_pt{var}']>cfg.SELECTION.SIGNAL.RECOIL)

            # Leading AK4 Jet
            leadak4_index = ak4_pt.argmax()
            leadak4_pt_eta = (ak4_pt.max() > cfg.SELECTION.SIGNAL.leadak4.PT) \
                            & (ak4.abseta[leadak4_index] < cfg.SELECTION.SIGNAL.leadak4.ETA).any()
            selection.add(f'leadak4_pt_eta{var}', leadak4_pt_eta)
    
            selection.add(f'leadak4_id{var}',(ak4.tightId[leadak4_index] \
                                                        & (ak4.chf[leadak4_index] >cfg.SELECTION.SIGNAL.leadak4.CHF) \
                                                        & (ak4.nhf[leadak4_index]<cfg.SELECTION.SIGNAL.leadak4.NHF)).any())

            # Looser version of leading jet pt cut
            leadak4_pt_eta_v2 = (ak4_pt.max() > 50) \
                            & (ak4.abseta[leadak4_index] < cfg.SELECTION.SIGNAL.leadak4.ETA).any()

            selection.add(f'leadak4_pt_eta_v2{var}', leadak4_pt_eta_v2)

            # AK8 Jet
            if cfg.RUN.MONOV:
                leadak8_index = ak8_pt.argmax()
                leadak8_pt_eta = (ak8_pt.max() > cfg.SELECTION.SIGNAL.leadak8.PT) \
                                & (ak8.abseta[leadak8_index] < cfg.SELECTION.SIGNAL.leadak8.ETA).any()
                selection.add(f'leadak8_pt_eta{var}', leadak8_pt_eta)
        
                selection.add(f'leadak8_id{var}',(ak8.tightId[leadak8_index]).any())
    
                # Mono-V selection
                selection.add(f'leadak8_tau21{var}', ((ak8.tau2[leadak8_index] / ak8.tau1[leadak8_index]) < cfg.SELECTION.SIGNAL.LEADAK8.TAU21).any())
                selection.add(f'leadak8_mass{var}', ((ak8.mass[leadak8_index] > cfg.SELECTION.SIGNAL.LEADAK8.MASS.MIN) \
                                            & (ak8.mass[leadak8_index] < cfg.SELECTION.SIGNAL.LEADAK8.MASS.MAX)).any())
                selection.add('leadak8_wvsqcd_loosemd', ((ak8.wvsqcdmd[leadak8_index] > cfg.WTAG.LOOSEMD)
                                            & (ak8.wvsqcdmd[leadak8_index] < cfg.WTAG.TIGHTMD)).any())
                selection.add('leadak8_wvsqcd_tightmd', ((ak8.wvsqcdmd[leadak8_index] > cfg.WTAG.TIGHTMD)).any())
                selection.add('leadak8_wvsqcd_loose', ((ak8.wvsqcd[leadak8_index] > cfg.WTAG.LOOSE)
                                            & (ak8.wvsqcd[leadak8_index] < cfg.WTAG.TIGHT)).any())
                selection.add('leadak8_wvsqcd_tight', ((ak8.wvsqcd[leadak8_index] > cfg.WTAG.TIGHT)).any())
        
                selection.add(f'veto_vtag{var}', ~selection.all(f"leadak8_pt_eta{var}", f"leadak8_id{var}", f"leadak8_tau21{var}", f"leadak8_mass{var}"))
                selection.add('only_one_ak8', ak8.counts==1)

            # Photons
            # Angular distance leading photon - leading jet
            phojet_pairs = ak4[:,:1].cross(photons[:,:1])
            df[f'dRPhotonJet{var}'] = np.hypot(phojet_pairs.i0.eta-phojet_pairs.i1.eta , dphi(phojet_pairs.i0.phi,phojet_pairs.i1.phi)).min()

        # Fill histograms
        output = self.accumulator.identity()

        # Gen
        if gen_v_pt is not None:
            output['genvpt_check'].fill(vpt=gen_v_pt,type="Nano", dataset=dataset, weight=df['Generator_weight'])

        if 'LHE_HT' in df:
            output['lhe_ht'].fill(dataset=dataset, ht=df['LHE_HT'])

        # Weights
        evaluator = evaluator_from_config(cfg)

        weights = processor.Weights(size=df.size, storeIndividual=True)
        if not df['is_data']:
            weights.add('gen', df['Generator_weight'])

            try:
                weights.add('prefire', df['PrefireWeight'])
            except KeyError:
                weights.add('prefire', np.ones(df.size))

            weights = candidate_weights(weights, df, evaluator, muons, electrons, photons)
            weights = pileup_weights(weights, df, evaluator, cfg)
            if not (gen_v_pt is None):
                weights = theory_weights_monojet(weights, df, evaluator, gen_v_pt)

        # Save per-event values for synchronization
        if cfg.RUN.KINEMATICS.SAVE:
            for event in cfg.RUN.KINEMATICS.EVENTS:
                mask = df['event'] == event
                if not mask.any():
                    continue
                output['kinematics']['event'] += [event]
                output['kinematics']['met'] += [met_pt[mask].flatten()]
                output['kinematics']['met_phi'] += [met_phi[mask].flatten()]
                output['kinematics']['recoil'] += [df['recoil_pt'][mask].flatten()]
                output['kinematics']['recoil_phi'] += [df['recoil_phi'][mask].flatten()]

                output['kinematics']['ak4pt0'] += [ak4[leadak4_index][mask].pt.flatten()]
                output['kinematics']['ak4eta0'] += [ak4[leadak4_index][mask].eta.flatten()]
                output['kinematics']['leadbtag'] += [ak4.pt.max()<0][mask]

                output['kinematics']['nLooseMu'] += [muons.counts[mask]]
                output['kinematics']['nTightMu'] += [muons[df['is_tight_muon']].counts[mask].flatten()]
                output['kinematics']['mupt0'] += [muons[leadmuon_index][mask].pt.flatten()]
                output['kinematics']['mueta0'] += [muons[leadmuon_index][mask].eta.flatten()]
                output['kinematics']['muphi0'] += [muons[leadmuon_index][mask].phi.flatten()]

                output['kinematics']['nLooseEl'] += [electrons.counts[mask]]
                output['kinematics']['nTightEl'] += [electrons[df['is_tight_electron']].counts[mask].flatten()]
                output['kinematics']['elpt0'] += [electrons[leadelectron_index][mask].pt.flatten()]
                output['kinematics']['eleta0'] += [electrons[leadelectron_index][mask].eta.flatten()]

                output['kinematics']['nLooseGam'] += [photons.counts[mask]]
                output['kinematics']['nTightGam'] += [photons[df['is_tight_photon']].counts[mask].flatten()]
                output['kinematics']['gpt0'] += [photons[leadphoton_index][mask].pt.flatten()]
                output['kinematics']['geta0'] += [photons[leadphoton_index][mask].eta.flatten()]


        # Sum of all weights to use for normalization
        # TODO: Deal with systematic variations
        output['nevents'][dataset] += df.size
        if not df['is_data']:
            output['sumw'][dataset] +=  df['genEventSumw']
            output['sumw2'][dataset] +=  df['genEventSumw2']
            output['sumw_pileup'][dataset] +=  weights.partial_weight(include=['pileup']).sum()

        regions = monojet_regions(cfg, self._variations)

        for region, cuts in regions.items():
            # Get relevant variation name for each region
            if ('Up' in region) or ('Down' in region):
                if str(df["year"]) not in region:
                    var = '_' + region.split('_')[-1]
                else:
                    var = '_' + '_'.join(region.split('_')[-2:])
            else:
                var = ''

            # Get the correct objects/quantities for each variation
            selection = vmap.get_selection_packer(var)
            bjets = vmap.get_bjets(var)
            ak4 = vmap.get_ak4(var) 
            if cfg.RUN.MONOV:
                ak8 = vmap.get_ak8(var) 
            met = vmap.get_met(var)
            # Get jet pts
            ak4_pt = getattr(ak4, f'pt{var}')
            leadak4_index = ak4_pt.argmax()
            ak4_pt0 = ak4_pt[leadak4_index]

            region_weights = copy.deepcopy(weights)
            if not df['is_data']:
                if re.match(r'cr_(\d+)e.*', region):
                    p_pass_data = 1 - (1-evaluator["trigger_electron_eff_data"](electrons.etasc, electrons.pt)).prod()
                    p_pass_mc   = 1 - (1-evaluator["trigger_electron_eff_mc"](electrons.etasc, electrons.pt)).prod()
                    trigger_weight = p_pass_data/p_pass_mc
                    trigger_weight[np.isnan(trigger_weight)] = 1
                    region_weights.add('trigger', trigger_weight)
                elif re.match(r'cr_(\d+)m.*', region) or re.match('sr_.*', region):
                    region_weights.add('trigger_met', evaluator["trigger_met"](df[f'recoil_pt{var}']))
                elif re.match(r'cr_g.*', region):
                    photon_trigger_sf(region_weights, photons, df)

            if cfg.RUN.MONOV:
                if not df['is_data']:
                    genVs = gen[((gen.pdg==23) | (gen.pdg==24) | (gen.pdg==-24)) & (gen.pt>10)]
                    leadak8 = ak8[ak8.pt.argmax()]
                    leadak8_matched_mask = leadak8.match(genVs, deltaRCut=0.8)
                    matched_leadak8 = leadak8[leadak8_matched_mask]
                    unmatched_leadak8 = leadak8[~leadak8_matched_mask]
                    for wp in ['loose','loosemd','tight','tightmd']:
                        if re.match(r'.*_{wp}_v.*', region):
    
                            if (wp == 'tight') or ('nomistag' in region): # no mistag SF available for tight cut
                                matched_weights = evaluator[f'wtag_{wp}'](matched_leadak8.pt).prod()
                            else:
                                matched_weights = evaluator[f'wtag_{wp}'](matched_leadak8.pt).prod() \
                                        * evaluator[f'wtag_mistag_{wp}'](unmatched_leadak8.pt).prod()
    
                            region_weights.add('wtag_{wp}', matched_weights)

            # Blinding
            if(self._blind and df['is_data'] and region.startswith('sr')):
                continue

            # Cutflow plot for signal and control regions
            if any(x in region for x in ["sr", "cr", "tr"]):
                output['cutflow_' + region][dataset]['all']+=df.size
                for icut, cutname in enumerate(cuts):
                    output['cutflow_' + region][dataset][cutname] += selection.all(*cuts[:icut+1]).sum()

            mask = selection.all(*cuts)

            # Save information to an output nested dictionary for electron and muon CR
            if cfg.RUN.SAVE.TREE:
                # if re.match('cr_\d(e|m).*', region):
                if region in ['cr_2m_j', 'cr_2e_j']:
                    def fill_tree(variable, values):
                        treeacc = processor.column_accumulator(values)
                        name = f'tree_{region}_{variable}'
                        if dataset in output[name].keys():
                            output[name][dataset] += treeacc
                        else:
                            output[name][dataset] = treeacc
    
                    # Fill different trees for different regions
                    trees = {
                        'cr_1m_.*' : 'tree_1m',
                        'cr_1e_.*' : 'tree_1e',
                        'cr_2m_.*' : 'tree_2m',
                        'cr_2e_.*' : 'tree_2e',
                    }

                    # Pick correct tree for relevant region
                    for region_name, tree_name in trees.items():
                        if re.match(region_name, region):
                            tree = tree_name
                            break

                    output[tree][region]["event"] +=  processor.column_accumulator(df["event"][mask])
                    output[tree][region]["run"] +=  processor.column_accumulator(df["run"][mask])
                    output[tree][region]["luminosityBlock"] +=  processor.column_accumulator(df["luminosityBlock"][mask])
                    output[tree][region]["gen_v_pt"] +=  processor.column_accumulator(gen_v_pt[mask])

                    # output[tree][region]["met"] += processor.column_accumulator(getattr(met, f'pt{var}')[mask].flatten())
                    output[tree][region]["met_pt_nom"] += processor.column_accumulator(met.pt_nom[mask].flatten())
                    output[tree][region]["met_pt_jesUp"] += processor.column_accumulator(met.pt_jesup[mask].flatten())
                    output[tree][region]["met_pt_jesDown"] += processor.column_accumulator(met.pt_jesdown[mask].flatten())
                    output[tree][region]["met_pt_jerUp"] += processor.column_accumulator(met.pt_jerup[mask].flatten())
                    output[tree][region]["met_pt_jerDown"] += processor.column_accumulator(met.pt_jerdown[mask].flatten())

                    output[tree][region]["met_phi"] += processor.column_accumulator(getattr(met, f'phi{var}')[mask].flatten())
                    output[tree][region]["recoil"] +=  processor.column_accumulator(df[f"recoil_pt{var}"][mask])
                    output[tree][region]["recoil_phi"] +=  processor.column_accumulator(df[f"recoil_phi{var}"][mask])
                    # output[tree][region]["theory"] +=  processor.column_accumulator(region_weights.partial_weight(include=["theory"])[mask])
    
                    # Leading jet information
                    output[tree][region]['ak4_pt0'] += processor.column_accumulator(ak4_pt0[mask].flatten())
                    output[tree][region]['ak4_eta0'] += processor.column_accumulator(ak4[leadak4_index].eta[mask].flatten())
                    output[tree][region]['ak4_phi0'] += processor.column_accumulator(ak4[leadak4_index].phi[mask].flatten())
    
                    if 'cr_1m_j' in region or 'cr_2m_j' in region:
                        output[tree][region]['muon_pt0'] += processor.column_accumulator(muons[leadmuon_index].pt[mask].flatten())
                        output[tree][region]['muon_eta0'] += processor.column_accumulator(muons[leadmuon_index].eta[mask].flatten())
                        output[tree][region]['muon_phi0'] += processor.column_accumulator(muons[leadmuon_index].phi[mask].flatten())
                        
                        if 'cr_2m_j' in region:
                            output[tree][region]['muon_pt1'] += processor.column_accumulator(muons[~leadmuon_index].pt[mask].flatten())
                            output[tree][region]['muon_eta1'] += processor.column_accumulator(muons[~leadmuon_index].eta[mask].flatten())
                            output[tree][region]['muon_phi1'] += processor.column_accumulator(muons[~leadmuon_index].phi[mask].flatten())
    
                            output[tree][region]['dimuon_pt'] += processor.column_accumulator(dimuons.pt[mask].flatten())
                            output[tree][region]['dimuon_eta'] += processor.column_accumulator(dimuons.eta[mask].flatten())
                            output[tree][region]['dimuon_mass'] += processor.column_accumulator(dimuons.mass[mask].flatten())

                    if 'cr_1e_j' in region or 'cr_2e_j' in region:
                        output[tree][region]['electron_pt0'] += processor.column_accumulator(electrons[leadelectron_index].pt[mask].flatten())
                        output[tree][region]['electron_eta0'] += processor.column_accumulator(electrons[leadelectron_index].eta[mask].flatten())
                        output[tree][region]['electron_phi0'] += processor.column_accumulator(electrons[leadelectron_index].phi[mask].flatten())
                        
                        if 'cr_2e_j' in region:
                            output[tree][region]['electron_pt1'] += processor.column_accumulator(electrons[~leadelectron_index].pt[mask].flatten())
                            output[tree][region]['electron_eta1'] += processor.column_accumulator(electrons[~leadelectron_index].eta[mask].flatten())
                            output[tree][region]['electron_phi1'] += processor.column_accumulator(electrons[~leadelectron_index].phi[mask].flatten())

                            output[tree][region]['dielectron_pt'] += processor.column_accumulator(dielectrons.pt[mask].flatten())
                            output[tree][region]['dielectron_eta'] += processor.column_accumulator(dielectrons.eta[mask].flatten())
                            output[tree][region]['dielectron_mass'] += processor.column_accumulator(dielectrons.mass[mask].flatten())

            # Save the event numbers of events passing this selection
            if cfg.RUN.SAVE.PASSING:
                # Save only every Nth event
                save_mask = mask & ((df['event']%cfg.RUN.SAVE.PRESCALE)== 0)
                output['selected_events'][region] += processor.column_accumulator(df['event'][save_mask].astype(np.uint64))


            # Multiplicities
            def fill_mult(name, candidates):
                output[name].fill(
                                  dataset=dataset,
                                  region=region,
                                  multiplicity=candidates[mask].counts,
                                  weight=region_weights.weight()[mask]
                                  )

            if cfg.RUN.MONOV:
                fill_mult('ak8_mult', ak8)
            fill_mult('ak4_mult', ak4)
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

            # All ak4
            # This is a workaround to create a weight array of the right dimension
            w_alljets = weight_shape(ak4[mask].eta, region_weights.weight()[mask])

            ezfill('ak4_eta',     jeteta=ak4[mask].eta.flatten(), weight=w_alljets)
            ezfill('ak4_phi',     jetphi=ak4[mask].phi.flatten(), weight=w_alljets)
            ezfill('ak4_pt',      jetpt=ak4_pt[mask].flatten(),   weight=w_alljets)

            # Leading ak4
            w_leadak4 = weight_shape(ak4[leadak4_index].eta[mask], region_weights.weight()[mask])
            ezfill('ak4_eta0',   jeteta=ak4[leadak4_index].eta[mask].flatten(),    weight=w_leadak4)
            ezfill('ak4_phi0',   jetphi=ak4[leadak4_index].phi[mask].flatten(),    weight=w_leadak4)
            ezfill('ak4_pt0',    jetpt=ak4_pt0[mask].flatten(),      weight=w_leadak4)
            ezfill('ak4_ptraw0',    jetpt=ak4[leadak4_index].ptraw[mask].flatten(),      weight=w_leadak4)

            if 'cr_2e' in region:
                zpt = calculate_v_pt_from_dilepton(dielectrons)
                ezfill('vpt', vpt=zpt[mask].flatten(), weight=region_weights.weight()[mask]) 

            # AK8 jets
            if cfg.RUN.MONOV:
                if region=='inclusive' or '_v' in region:
                    # All
                    w_allak8 = weight_shape(ak8.eta[mask], region_weights.weight()[mask])
                    ak8_pt = getattr(ak8, f'pt{var}')
                    leadak8_index = ak8_pt.argmax()
                    ak8_pt0 = ak8_pt[leadak8_index]
    
                    ezfill('ak8_eta',    jeteta=ak8[mask].eta.flatten(), weight=w_allak8)
                    ezfill('ak8_phi',    jetphi=ak8[mask].phi.flatten(), weight=w_allak8)
                    ezfill('ak8_pt',     jetpt=ak8_pt[mask].flatten(),   weight=w_allak8)
                    ezfill('ak8_mass',   mass=ak8[mask].mass.flatten(),  weight=w_allak8)
    
                    # Leading
                    w_leadak8 = weight_shape(ak8[leadak8_index].eta[mask], region_weights.weight()[mask])
    
                    ezfill('ak8_eta0',       jeteta=ak8[leadak8_index].eta[mask].flatten(),    weight=w_leadak8)
                    ezfill('ak8_phi0',       jetphi=ak8[leadak8_index].phi[mask].flatten(),    weight=w_leadak8)
                    ezfill('ak8_pt0',        jetpt=ak8_pt0[mask].flatten(),      weight=w_leadak8 )
                    ezfill('ak8_mass0',      mass=ak8[leadak8_index].mass[mask].flatten(),     weight=w_leadak8)
                    ezfill('ak8_tau210',     tau21=ak8[leadak8_index].tau21[mask].flatten(),     weight=w_leadak8)
                    ezfill('ak8_wvsqcd0',    tagger=ak8[leadak8_index].wvsqcd[mask].flatten(),     weight=w_leadak8)
                    ezfill('ak8_wvsqcdmd0',  tagger=ak8[leadak8_index].wvsqcdmd[mask].flatten(),     weight=w_leadak8)
                    ezfill('ak8_zvsqcd0',    tagger=ak8[leadak8_index].zvsqcd[mask].flatten(),     weight=w_leadak8)
                    ezfill('ak8_zvsqcdmd0',  tagger=ak8[leadak8_index].zvsqcdmd[mask].flatten(),     weight=w_leadak8)
                    ezfill('ak8_tvsqcd0',    tagger=ak8[leadak8_index].tvsqcd[mask].flatten(),     weight=w_leadak8)
                    ezfill('ak8_tvsqcdmd0',    tagger=ak8[leadak8_index].tvsqcdmd[mask].flatten(),     weight=w_leadak8)
                    ezfill('ak8_wvstqcd0',    tagger=ak8[leadak8_index].wvstqcd[mask].flatten(),     weight=w_leadak8)
                    ezfill('ak8_wvstqcdmd0',    tagger=ak8[leadak8_index].wvstqcdmd[mask].flatten(),     weight=w_leadak8)
    
                    # histogram with only gen-matched lead ak8 pt
                    if not df['is_data']:
                        w_matchedleadak8 = weight_shape(matched_leadak8.eta[mask], region_weights.weight()[mask])
                        ezfill('ak8_Vmatched_pt0', jetpt=matched_leadak8.pt[mask].flatten(),      weight=w_matchedleadak8 )
    
    
                    # Dimuon specifically for deepak8 mistag rate measurement
                    if 'inclusive_v' in region:
                        ezfill('ak8_passloose_pt0', wppass=ak8[leadak8_index].wvsqcd[mask].max()>cfg.WTAG.LOOSE, jetpt=ak8[leadak8_index].pt[mask].max(),      weight=w_leadak8 )
                        ezfill('ak8_passtight_pt0', wppass=ak8[leadak8_index].wvsqcd[mask].max()>cfg.WTAG.TIGHT, jetpt=ak8[leadak8_index].pt[mask].max(),      weight=w_leadak8 )
                        ezfill('ak8_passloosemd_pt0', wppass=ak8[leadak8_index].wvsqcdmd[mask].max()>cfg.WTAG.LOOSEMD, jetpt=ak8[leadak8_index].pt[mask].max(),      weight=w_leadak8 )
                        ezfill('ak8_passtightmd_pt0', wppass=ak8[leadak8_index].wvsqcdmd[mask].max()>cfg.WTAG.TIGHTMD, jetpt=ak8[leadak8_index].pt[mask].max(),      weight=w_leadak8 )
                        ezfill('ak8_passloose_mass0', wppass=ak8[leadak8_index].wvsqcd[mask].max()>cfg.WTAG.LOOSE, mass=ak8[leadak8_index].mass[mask].max(),      weight=w_leadak8 )
                        ezfill('ak8_passtight_mass0', wppass=ak8[leadak8_index].wvsqcd[mask].max()>cfg.WTAG.TIGHT, mass=ak8[leadak8_index].mass[mask].max(),      weight=w_leadak8 )
                        ezfill('ak8_passloosemd_mass0', wppass=ak8[leadak8_index].wvsqcdmd[mask].max()>cfg.WTAG.LOOSEMD, mass=ak8[leadak8_index].mass[mask].max(),      weight=w_leadak8 )
                        ezfill('ak8_passtightmd_mass0', wppass=ak8[leadak8_index].wvsqcdmd[mask].max()>cfg.WTAG.TIGHTMD, mass=ak8[leadak8_index].mass[mask].max(),      weight=w_leadak8 )

            # MET
            met_pt = getattr(met, f'pt{var}').flatten()
            met_phi = getattr(met, f'phi{var}').flatten()

            ezfill('dpfcalo',            dpfcalo=df[f"dPFCalo{var}"][mask],       weight=region_weights.weight()[mask] )
            ezfill('met',                met=met_pt[mask],            weight=region_weights.weight()[mask] )
            ezfill('met_phi',            phi=met_phi[mask],            weight=region_weights.weight()[mask] )
            ezfill('recoil',             recoil=df[f"recoil_pt{var}"][mask],      weight=region_weights.weight()[mask] )
            ezfill('dphijm',             dphi=df[f"minDPhiJetMet{var}"][mask],    weight=region_weights.weight()[mask] )
            ezfill('dphijr',             dphi=df[f"minDPhiJetRecoil{var}"][mask],    weight=region_weights.weight()[mask] )

            # Photon CR data-driven QCD estimate
            if df['is_data'] and re.match("cr_g.*", region) and re.match("(SinglePhoton|EGamma).*", dataset):
                w_imp = photon_impurity_weights(photons[leadphoton_index].pt.max()[mask], df["year"])
                output['recoil'].fill(
                                    dataset=data_driven_qcd_dataset(dataset),
                                    region=region,
                                    recoil=df[f"recoil_pt{var}"][mask],
                                    weight=region_weights.weight()[mask] * w_imp
                                )

            if 'noveto' in region:
                continue

            # For leptons, only fill MT histograms for the variated cases,
            # fill the remaining histograms only for nominal cases

            # Muons
            if '_1m_' in region or '_2m_' in region:
                w_allmu = weight_shape(muons.pt[mask], region_weights.weight()[mask])
                ezfill('muon_mt',   mt=df[f'MT_mu{var}'][mask],   weight=region_weights.weight()[mask])

            # Electrons
            if '_1e_' in region or '_2e_' in region:
                w_allel = weight_shape(electrons.pt[mask], region_weights.weight()[mask])
                ezfill('electron_mt',   mt=df[f'MT_el{var}'][mask],   weight=region_weights.weight()[mask])

            if var != '':
                continue

            # Muons
            if '_1m_' in region or '_2m_' in region:
                ezfill('muon_pt',   pt=muons.pt[mask].flatten(),    weight=w_allmu )
                ezfill('muon_eta',  eta=muons.eta[mask].flatten(),  weight=w_allmu)

                # Leading muon
                w_leadmu = weight_shape(muons[leadmuon_index].pt[mask], region_weights.weight()[mask])
                ezfill('muon_pt0',   pt=muons[leadmuon_index].pt[mask].flatten(),    weight=w_leadmu )
                ezfill('muon_eta0',  eta=muons[leadmuon_index].eta[mask].flatten(),  weight=w_leadmu)

            # Dimuon
            if '_2m_' in region:
                w_dimu = weight_shape(dimuons.pt[mask], region_weights.weight()[mask])

                ezfill('dimuon_pt',     pt=dimuons.pt[mask].flatten(),              weight=w_dimu)
                ezfill('dimuon_eta',    eta=dimuons.eta[mask].flatten(),            weight=w_dimu)
                ezfill('dimuon_mass',   dilepton_mass=dimuons.mass[mask].flatten(), weight=w_dimu )

                ezfill('muon_pt1',   pt=muons[~leadmuon_index].pt[mask].flatten(),    weight=w_leadmu )
                ezfill('muon_eta1',  eta=muons[~leadmuon_index].eta[mask].flatten(),  weight=w_leadmu)

            # Electrons
            if '_1e_' in region or '_2e_' in region:
                w_allel = weight_shape(electrons.pt[mask], region_weights.weight()[mask])
                ezfill('electron_pt',   pt=electrons.pt[mask].flatten(),    weight=w_allel)
                ezfill('electron_eta',  eta=electrons.eta[mask].flatten(),  weight=w_allel)

                w_leadel = weight_shape(electrons[leadelectron_index].pt[mask], region_weights.weight()[mask])
                ezfill('electron_pt0',   pt=electrons[leadelectron_index].pt[mask].flatten(),    weight=w_leadel)
                ezfill('electron_eta0',  eta=electrons[leadelectron_index].eta[mask].flatten(),  weight=w_leadel)

                w_trailel = weight_shape(electrons[~leadelectron_index].pt[mask], region_weights.weight()[mask])
                ezfill('electron_tightid1',  id=electrons[~leadelectron_index].tightId[mask].flatten(),  weight=w_trailel)

            # Dielectron
            if '_2e_' in region:
                w_diel = weight_shape(dielectrons.pt[mask], region_weights.weight()[mask])
                ezfill('dielectron_pt',     pt=dielectrons.pt[mask].flatten(),                  weight=w_diel)
                ezfill('dielectron_eta',    eta=dielectrons.eta[mask].flatten(),                weight=w_diel)
                ezfill('dielectron_mass',   dilepton_mass=dielectrons.mass[mask].flatten(),     weight=w_diel)

                ezfill('electron_pt1',   pt=electrons[~leadelectron_index].pt[mask].flatten(),    weight=w_leadel)
                ezfill('electron_eta1',  eta=electrons[~leadelectron_index].eta[mask].flatten(),  weight=w_leadel)
            # Photon
            if '_g_' in region:
                w_leading_photon = weight_shape(photons[leadphoton_index].pt[mask],region_weights.weight()[mask]);
                ezfill('photon_pt0',              pt=photons[leadphoton_index].pt[mask].flatten(),    weight=w_leading_photon)
                ezfill('photon_eta0',             eta=photons[leadphoton_index].eta[mask].flatten(),  weight=w_leading_photon)

                # w_drphoton_jet = weight_shape(df['dRPhotonJet'][mask], region_weights.weight()[mask])

            # PV
            ezfill('npv', nvtx=df['PV_npvs'][mask], weight=region_weights.weight()[mask])
            ezfill('npvgood', nvtx=df['PV_npvsGood'][mask], weight=region_weights.weight()[mask])

            ezfill('npv_nopu', nvtx=df['PV_npvs'][mask], weight=region_weights.partial_weight(exclude=['pileup'])[mask])
            ezfill('npvgood_nopu', nvtx=df['PV_npvsGood'][mask], weight=region_weights.partial_weight(exclude=['pileup'])[mask])

            ezfill('rho_all', rho=df['fixedGridRhoFastjetAll'][mask], weight=region_weights.weight()[mask])
            ezfill('rho_central', rho=df['fixedGridRhoFastjetCentral'][mask], weight=region_weights.weight()[mask])
            ezfill('rho_all_nopu', rho=df['fixedGridRhoFastjetAll'][mask], weight=region_weights.partial_weight(exclude=['pileup'])[mask])
            ezfill('rho_central_nopu', rho=df['fixedGridRhoFastjetCentral'][mask], weight=region_weights.partial_weight(exclude=['pileup'])[mask])
        return output

    def postprocess(self, accumulator):
        return accumulator
