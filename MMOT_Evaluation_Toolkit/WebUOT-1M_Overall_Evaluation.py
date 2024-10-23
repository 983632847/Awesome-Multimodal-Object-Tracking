from __future__ import absolute_import
from got10k.experiments import *
from siamfc import TrackerSiamFC

if __name__ == '__main__':
    ## Setup tracker
    # net_path = 'pretrained/siamfc/model.pth'
    # tracker = TrackerSiamFC(net_path=net_path)

    ## All Sequences
    att_names = ["All"]

    ## Scenarios
    sce_names = ['Underwater']

    for att_name in att_names:
        ## Setup experiments
        experiments = [
            # ExperimentGOT10k('data/GOT-10k', subset='test'),
            # ExperimentOTB('data/OTB', version=2013),
            # ExperimentOTB('data/OTB', version=2015),
            # ExperimentVOT('data/vot2018', version=2018),
            # ExperimentDTB70('data/DTB70/sequences'),
            # ExperimentTColor128('data/Temple-color-128'),
            # ExperimentUAV123('data/UAV123', version='UAV123'),
            # ExperimentUAV123('data/UAV123', version='UAV20L'),
            # ExperimentNfS('data/nfs', fps=30),
            # ExperimentNfS('data/nfs', fps=240),
            # ExperimentOTB('data/OTB100', version=2015),
            # ExperimentUAV123('data/UAV123', version='UAV123'),

            ## Overall Performance.
            ## The sce_name is not used in this evaluation, you can keep the default setting (sce_names[0]).
            ExperimentUWVLT('home/Datasets/WebUOT-1M/Test', att_name, sce_names[0]),
        ]

        ## Run tracking experiments (optional) and report performance
        for e in experiments:
            ## demo of SiamFC
            # e.run(tracker, visualize=False)                # run tracker
            # e.report([tracker.name])                       # report results

            #################################################
            ## Overall Performance
            # report result (one tracker)
            # e.report(["SiamFC"])

            # report results (multiple trackers)
            e.report([
                "SiamFC", "ECO", "VITAL", "ATOM", "SiamRPN++",
                "SiamBAN", "SiamCAR", "Ocean", "PrDiMP", "TrDiMP",
                "TransT", "STARK-ST50", "KeepTrack", "AutoMatch", "TCTrack",
                "ToMP-101", "AiATrack", "SimTrack-B32", "OSTrack", "MixFormerV2-B",
                "GRM", "SeqTrack-B256", "VLT_SCAR", "VLT_TT", "JointNLT",
                "CiteTracker-256", "All-in-One", "UVLTrack", "UOSTrack", "OKTrack",
                     ])


