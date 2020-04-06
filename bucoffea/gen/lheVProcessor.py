import coffea.processor as processor
import numpy as np
import awkward
from coffea import hist

from bucoffea.helpers import min_dphi_jet_met, dphi
from bucoffea.helpers.dataset import (is_lo_g, is_lo_g_ewk, is_lo_w, is_lo_z,
                                      is_nlo_g,is_nlo_g_ewk, is_nlo_w, is_nlo_z,)
from bucoffea.helpers.gen import (fill_gen_v_info,
                                  setup_dressed_gen_candidates,
                                  setup_gen_candidates,
                                  setup_lhe_cleaned_genjets,
                                  setup_lhe_parton_photon_pairs)

Hist = hist.Hist
Bin = hist.Bin
Cat = hist.Cat

def vbf_selection(vphi, dijet, genjets):
    selection = processor.PackedSelection()
    selection.add(
                  'two_jets',
                  dijet.counts>0
                  )
    selection.add(
                  'leadak4_pt_eta',
                  (dijet.i0.pt.max() > 80) & (np.abs(dijet.i0.eta.max()) < 5.0)
                  )
    selection.add(
                  'trailak4_pt_eta',
                  (dijet.i1.pt.max() > 40) & (np.abs(dijet.i1.eta.max()) < 5.0)
                  )
    selection.add(
                  'hemisphere',
                  (dijet.i0.eta.max()*dijet.i1.eta.max() < 0)
                  )
    selection.add(
                  'mindphijr',
                  min_dphi_jet_met(genjets, vphi, njet=4, ptmin=30, etamax=5.0) > 0.5
                  )
    selection.add(
                  'detajj',
                  np.abs(dijet.i0.eta-dijet.i1.eta).max() > 1
                  )
    selection.add(
                  'dphijj',
                  dphi(dijet.i0.phi,dijet.i1.phi).min() < 1.5
                  )

    return selection

def get_partial_mask_vbf(selection, exclude):
    '''
    Return a mask for partial VBF requirements. In the mask,
    all cuts will be required until the "exclude" cut is seen.
    '''
    # Require all cuts if none is going to be excluded
    if exclude == 'none':
        mask = selection.all(*selection.names)
        return mask
    
    cutnames = selection.names
    requirements = {}
    for cutname in cutnames:
        if cutname != exclude:
            requirements[cutname] = True
        elif cutname == exclude:
            break

    # Get the mask out of selection object
    mask = selection.require(**requirements)
    
    return mask

def monojet_selection(vphi, genjets):
    selection = processor.PackedSelection()

    selection.add(
                  'at_least_one_jet',
                  genjets.counts>0
                  )
    selection.add(
                  'leadak4_pt_eta',
                  (genjets.pt.max() > 100) & (np.abs(genjets[genjets.pt.argmax()].eta.max()) < 2.4)
                  )
    selection.add(
                  'mindphijr',
                  min_dphi_jet_met(genjets, vphi, njet=4, ptmin=30) > 0.5
                  )

    return selection


def photon_isolation_mask(partons, hadrons, photons):
    '''
    Returns a mask for the photon isolation requirement for every photon.
    Requirement is described in https://arxiv.org/pdf/1705.04664.pdf
    '''
    # Parameters as chosen in https://arxiv.org/pdf/1705.04664.pdf
    mz = 91
    eps0 = 0.1

    # Create photon-hadron and photon-parton pairs
    hadron_photon_pairs = hadrons.cross(photons, nested=True)
    parton_photon_pairs = partons.cross(photons, nested=True)

    # Calculate dynamic radius for the cone, different for each event, depends on photon pt
    R_dyn = mz / (photons.pt * np.sqrt(eps0))
    # Following two arrays do the same calculation, but array shape is modified
    R_dyn_photon_hadron = mz / (hadron_photon_pairs.i1.pt * np.sqrt(eps0))
    R_dyn_photon_parton = mz / (parton_photon_pairs.i1.pt * np.sqrt(eps0))

    # Loop over different R values, up to maximum R_dyn value
    # NOTE: Range + implementation may be changed
    logcount = 3
    R = R_dyn / (2**logcount)
    Rph = R_dyn_photon_hadron / (2**logcount)
    Rpp = R_dyn_photon_parton / (2**logcount)

    # Initialize photon isolation mask with True values for all photons
    photon_iso_mask = R_dyn.ones_like()  

    for _ in range(logcount+1):
        # Take the hadrons and partons that are within the cone with size R, sum their pt
        hadron_mask = hadrons.match(photons, deltaRCut=Rph)
        hadron_pt_sum = hadrons[hadron_mask].pt.sum()
        parton_mask = partons.match(photons, deltaRCut=Rpp)
        parton_pt_sum = partons[parton_mask].pt.sum()

        total_pt_sum = hadron_pt_sum + parton_pt_sum

        threshold = eps0 * photons.pt * ((1-np.cos(R))/(1-np.cos(R_dyn)))
        mask = total_pt_sum <= threshold

        # Update the photon isolation mask 
        photon_iso_mask = np.logical_and(photon_iso_mask, mask)

        # Go to the next radius values        
        R = R*2
        Rph = Rph*2
        Rpp = Rpp*2

    return photon_iso_mask
    
class lheVProcessor(processor.ProcessorABC):
    def __init__(self):
        # Histogram setup
        dataset_ax = Cat("dataset", "Primary dataset")
        cut_ax = Cat("cut", "The cut to be excluded in the cutflow")

        vpt_ax = Bin("vpt",r"$p_{T}^{V}$ (GeV)", 50, 0, 2000)
        jpt_ax = Bin("jpt",r"$p_{T}^{j}$ (GeV)", 50, 0, 2000)
        jet_eta_ax = Bin("jeteta", r"$\eta$", 50, -5, 5)
        mjj_ax = Bin("mjj",r"$m(jj)$ (GeV)", 75, 0, 7500)
        res_ax = Bin("res",r"pt: dressed / stat1 - 1", 80,-0.2,0.2)
        dr_ax = Bin("dr", r"$\Delta R$", 100, 0, 10)
        dphi_ax = Bin("dphi", r"$\Delta\phi$", 50, 0, 3.5)
        deta_ax = Bin("deta", r"$\Delta\eta$", 50, 0, 10)

        items = {}
        self.distributions = {
            'vpt' : vpt_ax,
            'mjj' : mjj_ax,
            'ak4_pt0'  : jpt_ax,
            'ak4_pt1'  : jpt_ax,
            'ak4_eta0' : jet_eta_ax,
            'ak4_eta1' : jet_eta_ax,
            'detajj'   : deta_ax,
            'dphijj'   : dphi_ax
        }

        for tag in ['stat1','dress','lhe','combined']:
            items[f"gen_vpt_inclusive_{tag}"] = Hist("Counts",
                                        dataset_ax,
                                        vpt_ax)

            items[f"gen_vpt_monojet_{tag}"] = Hist("Counts",
                                    dataset_ax,
                                    jpt_ax,
                                    vpt_ax)
            items[f"gen_vpt_vbf_{tag}"] = Hist("Counts",
                                    dataset_ax,
                                    jpt_ax,
                                    mjj_ax,
                                    vpt_ax)
            
            # Histograms for all variables, with VBF selection + DR > 0.4 requirement applied
            # These histograms also have a "cut axis" to specify to which point in the cutflow 
            # the cuts are applied while filling the histogram
            # For now, these studies are only for photons, therefore create these histograms 
            # for only "stat1" pt tag.
            if tag == 'stat1':
                for dist, variable_ax in self.distributions.items():
                    items[f"gen_{dist}_vbf_{tag}_withDRreq"] = Hist("Counts",
                                                    dataset_ax,
                                                    variable_ax,
                                                    cut_ax)

                # Store minDR between partons and photon in the events
                # Fill the histogram for three cases: Completely inclusive, with VBF cuts (with and without DR cut)     
                items[f'lhe_mindr_g_parton_{tag}'] = Hist("Counts",
                                                    dataset_ax,
                                                    dr_ax)

                items[f'lhe_mindr_g_parton_{tag}_noDRreq'] = Hist("Counts",
                                                        dataset_ax,
                                                        dr_ax)

                items[f'lhe_mindr_g_parton_{tag}_inclusive'] = Hist("Counts",
                                                        dataset_ax,
                                                        dr_ax)

        items["resolution"] = Hist("Counts",
                                dataset_ax,
                                res_ax)
        items['sumw'] = processor.defaultdict_accumulator(float)
        items['sumw2'] = processor.defaultdict_accumulator(float)

        self._accumulator = processor.dict_accumulator(items)

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        output = self.accumulator.identity()
        dataset = df['dataset']

        genjets = setup_lhe_cleaned_genjets(df)

        is_photon_sample = is_lo_g(dataset) | is_nlo_g(dataset) | is_lo_g_ewk(dataset) | is_nlo_g_ewk(dataset)

        # Dilepton
        gen = setup_gen_candidates(df)
        tags = ['stat1','lhe']
        if is_lo_w(dataset) or is_nlo_w(dataset) or is_lo_z(dataset) or is_nlo_z(dataset):
            dressed = setup_dressed_gen_candidates(df)
            fill_gen_v_info(df, gen, dressed)
            tags.extend(['dress','combined'])
        elif is_photon_sample:
            photons = gen[(gen.status==1) & (gen.pdg==22)]
            prompt_photons = photons[(photons.flag&1 == 1) & (np.abs(photons.eta)<1.442)]
            # Check if a prompt photon exists in the event
            good_prompt_photons  = prompt_photons.counts > 0

            # For V-pt and phi, if there is a prompt photon take the values from that photon
            # Otherwise, use the values from a non-prompt photon (if it exists) 
            df['gen_v_pt_stat1'] = np.where(
                good_prompt_photons,
                prompt_photons.pt.max(),
                photons.pt.max()
            )

            df['gen_v_phi_stat1'] = np.where(
                good_prompt_photons,
                prompt_photons.phi[prompt_photons.pt.argmax()].max(),
                photons.phi[photons.pt.argmax()].max()
            )

            df['gen_v_pt_lhe'] = df['LHE_Vpt']
            df['gen_v_phi_lhe'] = np.zeros(df.size)

            # Get LHE level photon + parton pairs
            # Calculate minimum deltaR between them in each event
            pairs = setup_lhe_parton_photon_pairs(df)
            min_dr = pairs.i0.p4.delta_r(pairs.i1.p4).min()
            df['lhe_mindr_g_parton'] = min_dr

            # Take the partons (before showering) and hadrons (after showering) from gen-level candidates
            partons = gen[((gen.status > 70) & (gen.status < 80)) & (gen.abspdg != 22)] # do not include photons
            hadrons = gen[gen.abspdg > 100]

        # Dijet for VBF
        dijet = genjets[:,:2].distincts()
        df['mjj'] = dijet.mass.max()

        # Leading and trailing jet pt and etas
        df['ak4_pt0'], df['ak4_eta0'] = dijet.i0.pt.max(), dijet.i0.eta.max()
        df['ak4_pt1'], df['ak4_eta1'] = dijet.i1.pt.max(), dijet.i1.eta.max()

        df['detajj'] = np.abs(df['ak4_eta0'] - df['ak4_eta1'])
        df['dphijj'] = dphi(dijet.i0.phi, dijet.i1.phi).min()

        for tag in tags:
            # Selection
            vbf_sel = vbf_selection(df[f'gen_v_phi_{tag}'], dijet, genjets)
            monojet_sel = monojet_selection(df[f'gen_v_phi_{tag}'], genjets)

            nominal = df['Generator_weight']

            output[f'gen_vpt_inclusive_{tag}'].fill(
                                    dataset=dataset,
                                    vpt=df[f'gen_v_pt_{tag}'],
                                    weight=nominal
                                    )
                                    
            mask_vbf = vbf_sel.all(*vbf_sel.names) 

            # For photons, add DR > 0.4 and V-pt > 150 GeV requirements
            if is_photon_sample and tag == 'stat1':
                # Get the photon isolation mask
                photon_iso_mask = photon_isolation_mask(partons, hadrons, photons)

                dr_mask = df['lhe_mindr_g_parton'] > 0.4
                vpt_mask = df['gen_v_pt_stat1'] > 150
                full_mask_vbf = mask_vbf * dr_mask * vpt_mask 

            output[f'gen_vpt_vbf_{tag}'].fill(
                                    dataset=dataset,
                                    vpt=df[f'gen_v_pt_{tag}'][full_mask_vbf],
                                    jpt=genjets.pt.max()[full_mask_vbf],
                                    mjj = df['mjj'][full_mask_vbf],
                                    weight=nominal[full_mask_vbf]
                                    )

            # Fill histograms for deltaR distribution between photons and partons at LHE level
            # Also fill some gen-level distributions (only for photons for now)
            if is_photon_sample and tag == 'stat1':
                # Fill the DR histogram with the deltaR requirement
                output[f'lhe_mindr_g_parton_{tag}'].fill(
                                    dataset=dataset,
                                    dr=df['lhe_mindr_g_parton'][full_mask_vbf],
                                    weight=nominal[full_mask_vbf]
                                    )
                
                # Fill the DR histogram without the deltaR requirement
                output[f'lhe_mindr_g_parton_{tag}_noDRreq'].fill(
                                    dataset=dataset,
                                    dr=df['lhe_mindr_g_parton'][mask_vbf],
                                    weight=nominal[mask_vbf]
                                    )

                # Fill the DR histogram for the inclusive case
                output[f'lhe_mindr_g_parton_{tag}_inclusive'].fill(
                                    dataset=dataset,
                                    dr=df['lhe_mindr_g_parton'],
                                    weight=nominal
                                    )

                def ezfill(dist, **kwargs):
                    '''Function for easier histogram filling.'''
                    output[f'gen_{dist}_vbf_stat1_withDRreq'].fill(
                                                        dataset=dataset,
                                                        **kwargs
                                                    )

                # Fill histograms for separate points in the cutflow
                cuts_to_exclude = vbf_sel.names + ['none']
                cut_labels = ['inclusive'] + [f'up_to_{cut}' for cut in cuts_to_exclude[1:] if cut != 'none'] + ['all_cuts_applied']

                for cut, cutlabel in zip(cuts_to_exclude, cut_labels):
                    # Get partial VBF masks, also with DR > 0.4 and V-pt > 150 GeV requirements applied
                    mask = get_partial_mask_vbf(selection=vbf_sel, exclude=cut) * dr_mask * vpt_mask

                    # Fill histograms with (partial) VBF selection + DR > 0.4 + V-pt > 150 GeV requirements
                    ezfill('vpt', vpt=df['gen_v_pt_stat1'][mask], cut=cutlabel, weight=nominal[mask])
                    ezfill('mjj', mjj=df['mjj'][mask], cut=cutlabel, weight=nominal[mask])
                    
                    ezfill('ak4_pt0', jpt=df['ak4_pt0'][mask], cut=cutlabel, weight=nominal[mask])
                    ezfill('ak4_pt1', jpt=df['ak4_pt1'][mask], cut=cutlabel, weight=nominal[mask])
                    ezfill('ak4_eta0', jeteta=df['ak4_eta0'][mask], cut=cutlabel, weight=nominal[mask])
                    ezfill('ak4_eta1', jeteta=df['ak4_eta1'][mask], cut=cutlabel, weight=nominal[mask])
    
                    ezfill('detajj', deta=df['detajj'][mask], cut=cutlabel, weight=nominal[mask])
                    ezfill('dphijj', dphi=df['dphijj'][mask], cut=cutlabel, weight=nominal[mask])
                                   
            mask_monojet = monojet_sel.all(*monojet_sel.names)

            output[f'gen_vpt_monojet_{tag}'].fill(
                                    dataset=dataset,
                                    vpt=df[f'gen_v_pt_{tag}'][mask_monojet],
                                    jpt=genjets.pt.max()[mask_monojet],
                                    weight=nominal[mask_monojet]
                                    )

        # Keep track of weight sum
        output['sumw'][dataset] +=  df['genEventSumw']
        output['sumw2'][dataset] +=  df['genEventSumw2']

        return output

    def postprocess(self, accumulator):
        return accumulator
