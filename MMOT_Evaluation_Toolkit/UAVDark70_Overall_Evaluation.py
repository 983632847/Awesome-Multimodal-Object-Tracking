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
    sce_names = ['UAVDark70']

    for att_name in att_names:
        ## Setup experiments
        experiments = [
            ## Overall Performance.
            ## The sce_name is not used in this evaluation, you can keep the default setting (sce_names[0]).
            ExperimentUAVDark70('/Disk/Night_UAV_Tracking/Datasets/UAVDark70', att_name, sce_names[0]),
        ]

        ## Run tracking experiments (optional) and report performance
        for e in experiments:
            ## demo of SiamFC
            # e.run(tracker, visualize=False)                # run tracker
            # e.report([tracker.name])                       # report results

            #################################################
            ## 1.Overall Performance
            # report result (one tracker)
            # e.report(["MambaTrack"])

            # report results (multiple trackers)
            e.report([
                "MambaTrack", "CiteTracker", "VLT_SCAR", "VLT_TT", "JointNLT",
                "STARK50", "MixFormerV2", "D3S", "DaSiamRPN", "HiFT",
                "Ocean","SE-SiamFC", "SiamAPN", "SiamAPN++", "SiamBAN",
                "SiamCAR",'SiamDW_RPN_Res22', "SiamFC++", "SiamRPN++", "UDAT-BAN",
                "UDAT-CAR", "UpdateNet"
                     ])

