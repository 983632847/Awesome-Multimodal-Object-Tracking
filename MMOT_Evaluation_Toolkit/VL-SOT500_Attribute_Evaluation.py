from __future__ import absolute_import
from got10k.experiments import *
from siamfc import TrackerSiamFC

if __name__ == '__main__':
    ## Setup tracker
    # net_path = 'pretrained/siamfc/model.pth'
    # tracker = TrackerSiamFC(net_path=net_path)

    ## Attributes
    att_names = [
        'Camera_Motion', 'Viewpoint_Changes', 'Partial_Occlusion', 'Full_Occlusion', 'Out_of_View',
        'Rotation', 'Deformation', 'Similar_Distractors', 'Illumination_Variations', 'Motion_Blur',
        'Natural_Object', 'Artificial_Object', 'Partial_Target_Information',
        'Brightness_low', 'Brightness_medium', 'Brightness_high',
        'Fast_Motion', 'Scale_variation', 'Aspect_Ratio_Variation',
        'Length_short', 'Length_medium', 'Length_long']

    ## Scenarios
    sce_names = ['VL-SOT']

    for att_name in att_names:
        ## Setup experiments
        experiments = [

            ## Attribute-Based Performance.
            ## The sce_name is not used in this evaluation, you can keep the default setting (sce_names[0]).
            ExperimentVLTOT230('/Disk/Tiny_Object_Tracking/VL-TOT/Test', att_name, sce_names[0],
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
            ## 2.Attribute-Based Performance
            # report result (one tracker)
            # e.report(["TransT"])

            # report results (multiple trackers)
            # VL-SOT230 Attributes
            e.report([
                "SiamFC", "ECO", "ATOM", "VITAL", "SiamRPN++",
                "Ocean", "DiMP", "PrDiMP", "TransT", "UDAT",
                "Aba-ViTrack", "AutoMatch", "HiFT", "LightTrack", "TCTrack",
                "JointNLT", "VLT_TT", "VLT_SCAR", "CiteTracker-256", "COST",
            ])




