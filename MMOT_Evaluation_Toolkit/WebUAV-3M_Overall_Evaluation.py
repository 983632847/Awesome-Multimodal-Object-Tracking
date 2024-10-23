from __future__ import absolute_import
from got10k.experiments import *
from siamfc import TrackerSiamFC

if __name__ == '__main__':
    ## Setup tracker
    # net_path = 'pretrained/siamfc/model.pth'
    # tracker = TrackerSiamFC(net_path=net_path)

    ## All Sequences
    att_names = ["All"]

    ## Attributes
    # att_names = [
    # 'Low_resolution','Partial_occlusion','Full_occlusion','Out_of_view','Fast_motion',
    # 'Camera_motion','Viewpoint_change','Rotation','Deformation','Background_clutters',
    # 'Scale_variation','Aspect_ratio_variation','Illumination_variation','Motion_blur','Complexity_easy',
    # 'Complexity_medium','Complexity_hard','Size_small','Size_medium','Size_large',
    # 'Length_short','Length_medium','Length_long']

    ## Scenarios
    sce_names = [
        'Low_light', 'Long-term_occlusion', 'Small_targets',
        'High-speed_motion', 'Dual-dynamic_disturbances', 'Target_distortions',
        'Adversarial_examples']

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
            ExperimentWebUAV3M('annotations/Test_Set', att_name, sce_names[0]),
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
            e.report([
                "SiamFC", "ACT", "AlphaRefine", "ARCF", "ATOM",
                "AutoMatch", "AutoTrack", "BACF", "CACF", "CCOT",
                "CF2", "DaSiamRPN", "DeepSRDCF", "DiMP", "DSiam",
                "ECO", "GOTURN", "HiFT", "KCF", "KeepTrack",
                "LADCF", "LightTrack", "MCCT", "MDNet", "MetaTracker",
                "Ocean", "PrDiMP", "SiamBAN", "SiamCAR", "SiamDW",
                "SiamFC++", "SiamGAT", "SiamMask", "SiamRPN", "SiamRPN++",
                "STRCF", "TransT", "TrDiMP", "UDT", "UpdateNet",
                "UTrack", "VITAL", "RPT"
                     ])

