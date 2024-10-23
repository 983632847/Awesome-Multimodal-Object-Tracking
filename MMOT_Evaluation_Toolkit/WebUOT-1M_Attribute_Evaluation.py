from __future__ import absolute_import
from got10k.experiments import *
from siamfc import TrackerSiamFC

if __name__ == '__main__':
    ## Setup tracker
    # net_path = 'pretrained/siamfc/model.pth'
    # tracker = TrackerSiamFC(net_path=net_path)

    ## Attributes
    # 10+14+15+16+3=58
    att_names = [
        'Low_resolution', 'Fast_motion', 'Scale_variation', 'Aspect_ratio_variation',
        'Size_Small', 'Size_Medium', 'Size_Big',
        'Length_Short', 'Length_Medium', 'Length_Long',

        'Camera_motion', 'Viewpoint_change', 'Partial_occlusion', 'Full_occlusion', 'Out_of_view',
        'Rotation', 'Deformation', 'Similar_distractors', 'Illumination_variation', 'Motion_blur',
        'Partial_target_information', 'Natural_object', 'Artificial_object', 'Camouflage',

        'Underwater_visibility_Low', 'Underwater_visibility_Medium', 'Underwater_visibility_High',
        'Underwater_scene_Sea', 'Underwater_scene_River', 'Underwater_scene_Pool', 'Underwater_scene_Water_Tank',
        'Underwater_scene_Fish_Tank', 'Underwater_scene_Basin', 'Underwater_scene_Bowl', 'Underwater_scene_Cup',
        'Underwater_scene_Aquaria', 'Underwater_scene_Pond', 'Underwater_scene_Puddle', 'Underwater_scene_Lake',

        'Water_color_variation_Colorless', 'Water_color_variation_Ash', 'Water_color_variation_Gray',
        'Water_color_variation_Green', 'Water_color_variation_Light_Green', 'Water_color_variation_Dark',
        'Water_color_variation_Blue_Black', 'Water_color_variation_Deep_Blue', 'Water_color_variation_Blue',
        'Water_color_variation_Light_Blue', 'Water_color_variation_Partly_Blue', 'Water_color_variation_GrayBlue',
        'Water_color_variation_Light_Yellow', 'Water_color_variation_Light_Brown', 'Water_color_variation_Cyan',
        'Water_color_variation_Light_Purple',

        'Underwater_view', 'Fish_eye_view', 'Outside_water_view',
    ]

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

            ## Attribute-Based Performance.
            ## The sce_name is not used in this evaluation, you can keep the default setting (sce_names[0]).
            ExperimentUWVLT('home/Datasets/WebUOT-1M/Test', att_name, sce_names[0]),     ##
        ]

        ## Run tracking experiments (optional) and report performance
        for e in experiments:
            ## demo of SiamFC
            # e.run(tracker, visualize=False)                # run tracker
            # e.report([tracker.name])                       # report results

            #################################################
            ## Attribute-Based Performance
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



