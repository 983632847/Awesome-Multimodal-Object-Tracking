from __future__ import absolute_import
from got10k.experiments import *
from siamfc import TrackerSiamFC

if __name__ == '__main__':
    ## Setup tracker
    # net_path = 'pretrained/siamfc/model.pth'
    # tracker = TrackerSiamFC(net_path=net_path)

    ## All Sequences
    # att_names = ["VL-SOT230"]
    att_names = ["VL-SOT270"]

    ## Scenarios
    sce_names = ['VL-SOT']

    for att_name in att_names:
        ## Setup experiments
        experiments = [

            ## Overall Performance.
            # The sce_name is not used in this evaluation, you can keep the default setting (sce_names[0]).
            ExperimentVLTOT230('/Disk/Tiny_Object_Tracking/VL-TOT/VL-SOT230', att_name, sce_names[0],
                            result_dir='results/Baseline_Results/VL-SOT230'),
            # ExperimentVLTOT270('/Disk/Tiny_Object_Tracking/VL-TOT/VL-SOT270', att_name, sce_names[0],
            #                 result_dir='results/Baseline_Results/VL-SOT270'),
        ]

        ## Run tracking experiments (optional) and report performance
        for e in experiments:
            ## demo of SiamFC
            # e.run(tracker, visualize=False)                # run tracker
            # e.report([tracker.name])                       # report results

            #################################################
            ## 1.Overall Performance
            # report result (one tracker)
            # e.report(["SiamFC"])

            # report results (multiple trackers)
            ## VL-SOT230, VL-SOT270
            e.report([
                "SiamFC", "ECO", "ATOM", "VITAL", "SiamRPN++",
                "Ocean", "DiMP", "PrDiMP", "TransT", "UDAT",
                "Aba-ViTrack", "AutoMatch", "HiFT", "LightTrack", "TCTrack",
                "SiamFC++", "SiamGAT", "STMTrack", "STARK-ST50", "KYS",
                "SiamBAN", "SiamCAR", "OSTrack", "TransInMo", "GRM",
                "ARTrack", "ZoomTrack", "SeqTrack-B256", "UVLTrack", "MMTrack",
                "JointNLT", "VLT_TT", "VLT_SCAR", "CiteTracker-256", "COST"
            ])


