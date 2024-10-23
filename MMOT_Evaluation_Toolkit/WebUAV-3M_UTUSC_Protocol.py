from __future__ import absolute_import
from got10k.experiments import *
from siamfc import TrackerSiamFC

if __name__ == '__main__':
    ## Setup tracker
    # net_path = 'pretrained/siamfc/model.pth'
    # tracker = TrackerSiamFC(net_path=net_path)

    ## Attributes
    # att_names = [
    # 'Low_resolution','Partial_occlusion','Full_occlusion','Out_of_view','Fast_motion',
    # 'Camera_motion','Viewpoint_change','Rotation','Deformation','Background_clutters',
    # 'Scale_variation','Aspect_ratio_variation','Illumination_variation','Motion_blur','Complexity_easy',
    # 'Complexity_medium','Complexity_hard','Size_small','Size_medium','Size_large',
    # 'Length_short','Length_medium','Length_long']

    ## Scenarios
    att_names = ["All"]
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

            ## UTUSC Evaluation Protocal
            ExperimentWebUAV3M('annotations/Scenario_low_light', att_name, sce_names[0]),                 # 0
            ExperimentWebUAV3M('annotations/Scenario_long_term_occlusion', att_name, sce_names[1]),       # 1
            ExperimentWebUAV3M('annotations/Scenario_small_targets', att_name, sce_names[2]),             # 2
            ExperimentWebUAV3M('annotations/Scenario_high_speed_motion', att_name, sce_names[3]),         # 3
            ExperimentWebUAV3M('annotations/Scenario_dual_dynamic_disturbances', att_name, sce_names[4]), # 4
            ExperimentWebUAV3M('annotations/Scenario_target_distortions', att_name, sce_names[5]),        # 5
            # ExperimentWebUAV3M('annotations/Scenario_adversarial_examples', att_name, sce_names[6])     # 6
        ]

        ## Run tracking experiments (optional) and report performance
        for e in experiments:
            ## demo of SiamFC
            # e.run(tracker, visualize=False)                # run tracker
            # e.report([tracker.name])                       # report results

            #################################################
            ## UTUSC Evaluation Protocol
            ## Six Scenarios: long-term occlusion, target distortions, dual-dynamic disturbances, small targets, high-speed motion and low light
            # report result (one tracker)
            # e.report_scenario(["SiamFC"])

            # report result (multiple trackers)
            e.report_scenario([
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

            ## Adversarial Examples
            # report result (one tracker)
            # e.report_scenario(["SiamFC"])

            # report results (multiple trackers)
            # e.report_scenario([
            #     "SiamFC", "Ocean", "SiamRPN++", "ACT", "AutoMatch",
            #     "DaSiamRPN", "GOTURN", "HiFT", "KCF", "LightTrack",
            #     "MDNet", "MetaTracker", "SiamBAN", "SiamCAR", "SiamDW",
            #     "SiamGAT","SiamRPN", "TransT", "VITAL", "UpdateNet",
            #     "SiamFC++"])