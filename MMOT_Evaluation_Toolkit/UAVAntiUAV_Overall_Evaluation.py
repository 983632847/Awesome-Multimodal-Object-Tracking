from __future__ import absolute_import
from got10k.experiments import *
# from siamfc import TrackerSiamFC

if __name__ == '__main__':
    ## All Sequences
    att_names = ["All"]

    ## Attributes
    # att_names = [
    # 'Camera_Motion', 'Viewpoint_Changes', 'Partial_Occlusion', 'Full_Occlusion', 'Out_of_View',
    # 'Rotation', 'Similar_Distractors', 'Illumination_Variations', 'Motion_Blur', 'Partial_Target_Information',
    # 'Small_Object', 'Fast_Motion', 'Scale_Variation', 'Aspect_Ratio_Variation',
    # 'Length_short','Length_medium','Length_long'
    # ]

    ## Scenarios
    sce_names = ['UAVAntiUAV']

    for att_name in att_names:
        ## Setup experiments
        experiments = [
            ## Overall Performance.
            # The sce_name is not used in this evaluation, you can keep the default setting (sce_names[0]).
            ExperimentUAVAntiUAV_Overall('/mnt/Datasets/UAVAntiUAV_Tracking/Test', att_name, sce_names[0],
                            result_dir='results/Baseline_Results/UAVAntiUAV'),
        ]

        ## Run tracking experiments (optional) and report performance
        for e in experiments:
            #################################################
            ## Overall Performance
            # 1.report result (one tracker)
            # e.report(["All-in-One"])

            # 2.report results (multiple trackers)
            e.report([
                "SiamFC", "ECO", "VITAL", "ATOM", "SiamMask",
                "SiamRPN++", "SiamFC++", "SiamBAN", "SiamCAR", "LightTrack",

                "SiamGAT", "TrDiMP", "TransT", "STARK-ST50", "KeepTrack",
                "HiFT", "AutoMatch", "TCTrack", "ToMP-101", "AiATrack",

                "SimTrack-B32", "OSTrack", "SGLATrack", "GRM", "ZoomTrack",
                "Aba-ViTrack", "HIPTrack", "AQATrack", "TCTrack++", "EVPTrack",

                "SeqTrack-B256", "DropTrack", "MixFormerV2", "ARTrackV2", "MCITrack-B224",
                "ODTrack", "LORAT-B224", "MambaNUT", "ORTrack", "MambaLCT",

                "ATCTrack", "JointNLT", "VLT_TT", "MambaTrack", "All-in-One",
                "CiteTracker-256", "UVLTrack", "DUTrack-256", "SUTrack-B224", "MambaSTS",
            ])


