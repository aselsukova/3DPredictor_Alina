import logging
import sys
import os
head_folder_path = os.path.dirname(os.path.abspath(sys.argv[0]))+"/3Dpredictor"
source_path = os.path.dirname(os.path.abspath(sys.argv[0]))+"/3Dpredictor/source/"
sys.path.append(source_path)
from Predictor import Predictor
from Weight_funcs_modul import *
import numpy as np
from functools import partial
from shared import Interval,decorate_oe2obs, oe2obs, decorate_return_coordinates_after_rearrangements, return_coordinates_after_deletion
import pandas as pd
import datetime
import pickle

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%I:%M:%S', level=logging.DEBUG)

expected_folder = "/mnt/scratch/ws/psbelokopytova/202011291709Polya_data/3DPredictor/input/expected/K562/5KB/"
cell_type = 'mast_cells'
out_dir = "/mnt/scratch/ws/psbelokopytova/202011291709Polya_data/3DPredictor/out/"
# contact_type = ["oe"]
# not_cor_predictors_file = "/mnt/scratch/ws/psbelokopytova/202011291709Polya_data/3DPredictor/out/Hepat/validating_chrms_1/not_cor_predictors0.8"
# suffix = ".gz.8.1500000.50001.25000.25000.txt"

# suffix_val = ".gz.1000000.50001.500000.25000.txt"#".gz.12.1000000.50001.1117748.25000.txt"
deletion =Interval("chr5", 75852814, 75881252)
# deletion = "no_deletion"
training_files = [
    # "/mnt/scratch/ws/psbelokopytova/202011291709Polya_data/3DPredictor/out/H1/Interval_chr10_13000_133787000training.RandOn1000.contacts.gz.False.11.1500000.2001.1.270376.cont_with_CTCF2833443.1000.txt",
   "/mnt/scratch/ws/psbelokopytova/202011291709Polya_data/3DPredictor/out/H1/Interval_chr10_51000_133764000validatingOrient.1000.contacts.gz.False.11.1500000.2001.1.270376.cont_with_CTCF135188.1000.txt",
]

# predictors_data=pd.read_csv(not_cor_predictors_file,sep="\t")
# predictors=str("|".join(list(predictors_data["predictors"])))
# print(predictors)
# i=1
# assert i>1
for training_file in training_files:
    validation_files = [
    # "/mnt/scratch/ws/psbelokopytova/202011291709Polya_data/3DPredictor/out/H1/Interval_chr7_86600000_87300000validatingOrient.1000.contacts.gz.False.11.1500000.2001.26953600.all_cont.1000.txt",
    # "/mnt/scratch/ws/psbelokopytova/202011291709Polya_data/3DPredictor/out/H1/Interval_chr7_86600000_86710000validatingOrient.1000.contacts.gz.False.11.1500000.2001.500000.all_cont.1000.txt",
    # "/mnt/scratch/ws/psbelokopytova/202011291709Polya_data/3DPredictor/out/H1/Interval_chr7_86600000_87400000validatingOrient.1000.contacts.gz.False.11.1500000.2001.500000.all_cont.1000.txt",
    # "/mnt/scratch/ws/psbelokopytova/202011291709Polya_data/3DPredictor/out/H1/Interval_chr7_86600000_89000000validatingOrient.1000.contacts.gz.False.11.1500000.2001.500000.all_cont.1000.txt",
    # "/mnt/scratch/ws/psbelokopytova/202011291709Polya_data/3DPredictor/out/mast_cells/Interval_chr5_75000000_76400000validatingOrient.1000.contacts.gz.False.11.1500000.2001.29840288.all_cont.1000.txt",
    # "/mnt/scratch/ws/psbelokopytova/202011291709Polya_data/3DPredictor/out/mast_cells/chr5validatingOrient.1000.contacts.gz.False.11.1500000.2001.1.all_cont.1000.txt",
    # "/mnt/scratch/ws/psbelokopytova/202011291709Polya_data/3DPredictor/out/mast_cells/Interval_chr5_74000000_76400000validatingOrient_from1kbNPC.1000.contacts.gz.False.11.1500000.2001.126915148.all_cont.1000.txt"
    # "/mnt/scratch/ws/psbelokopytova/202011291709Polya_data/3DPredictor/out/mast_cells/NPC_1KbInterval_chr5_74000000_76400000validatingOrient.1000.contacts.gz.False.11.1500000.2001.29840288.all_cont.1000.txt"
    # "/mnt/scratch/ws/psbelokopytova/202011291709Polya_data/3DPredictor/out/mast_cells/Interval_chr5_75000000_76400000fill_empty.validatingOrient.1000.contacts.gz.False.11.1500000.2001.221677785.all_cont.1000.txt"
    "/mnt/scratch/ws/psbelokopytova/202011291709Polya_data/3DPredictor/out/mast_cells/30KBdeletion_NPC_1KbInterval_chr5_75200000_76400000validatingOrient.1000.False.11.1500000.2001.2412461.all_cont.1000.txt"
    ]
    #Some examples of predictors filtering:
    #predictor.filter_predictors(".*CTCF.*|.*RNA.*", keep=False) #-- discard CTCF AND RNA
    #predictor.filter_predictors(".*CTCF.*", keep=False) #-- discard CTCF
    #predictor.filter_predictors(".*contact_dist.*|.*CTCF_W.*", keep=True) #-- keep only distance and CTCF in window

    for contact_type,apply_log in zip(["contacts"],[True]):
    #for contact_type,apply_log in zip(["contacts"],[False]):
        for (filter,keep),shortcut in zip(zip(["CTCF|RNA|contact_dist"], [True]),
                                          ["CTCF|RNA|contact_dist"]):#, "CTCF", "CTCF|contact_dist"]#"Loop","Loop|E1"]#,"Loop|E1|contact_dist","Loop|E1|contact_dist","Loop|E1|contact_dist|CTCF_L|CTCF_W|CTCF_R"] \
                # [True],#, True, True]),#,False,True,True]),
                #                                ["CTCF|contact_dist|H3K27acll"])):#,"CTCF_only", "CTCF and contact_dist" ]):#, "no loop,no E1,no dist", \
                #                                 #"loop,E1,dist", "loop,E1,dist,LRW_CTCF,Nbl"]):
            # filter2, keep2 = zip(["OrientBlock|plus_orientation|minus_orientation"], [False])
            if contact_type == "oe":
                weightFuncs = [ones_like]
                               #abs_log, decorate_mult_abs_log(mult_abs_log,100),decorate_overweight_loops(overweight_loops,10),
                    # decorate_overweight_loops(overweight_loops,1000), \
                                #decorateContactWeither(contactWeitherFunction, coeff=5),decorateContactWeither(contactWeitherFunction, power=3),
                                # decorateContactWeither(contactWeitherFunction, power=3, abs=True), abs_log, decorateContactWeither(contactWeitherFunction, coeff=5, piecing=True), \
                                #  decorateContactWeither(contactWeitherFunction, threshold=1.2, coeff=5, piecing=True, asymmetric=1)] \
                                # decorateContactWeither(contactWeitherFunction, power=3,asymmetric=1)]
                                # [ones_like, array, abs_log, decorate_mult_abs_log(mult_abs_log,100), decorate_overweight_loops(overweight_loops,100)] \
                                # [decorateContactWeither(contactWeitherFunction, coeff=5),decorateContactWeither(contactWeitherFunction, power=3), \
                                # decorateContactWeither(contactWeitherFunction, power=3, abs=True), decorateContactWeither(contactWeitherFunction, coeff=5, piecing=True), \
                                # decorateContactWeither(contactWeitherFunction, threshold=1.2, coeff=5, piecing=True, asymmetric=1), \
                                # decorateContactWeither(contactWeitherFunction, power=3,asymmetric=1)]
                                # #decorate_mult_abs_log(mult_abs_log,10000)]
            else:
                weightFuncs = [ones_like]
            for h in range(2, 3):
                for weightFunc in weightFuncs:
                    predictor = Predictor()
                    predictor.read_data_predictors(training_file)# + contact_type + suffix)
                    predictor.filter_predictors(filter, keep)
                    # for (filter2, keep2), shortcut in zip(
                    #         zip(["OrientBlock|ConvergentPair|minus_orientation|plus_orientation"], [False]),
                    #         ["woOrient"]):
                    #     predictor.filter_predictors(filter2, keep2)
                    print("load model")
                    trained_predictor = pickle.load(open("/mnt/scratch/ws/psbelokopytova/202011291709Polya_data/3DPredictor/out/models/912470444","rb"))
                    # trained_predictor = predictor.train(shortcut=shortcut, apply_log = apply_log,
                    #                                     weightsFunc = weightFunc, show_plot=False, out_dir=out_dir+"models/")
                    trained_predictor.out_dir = "/mnt/scratch/ws/psbelokopytova/202011291709Polya_data/3DPredictor/out/models/"
                    trained_predictor.draw_Feature_importances(show_plot=False)
                    print("load succesfully")
                    for validation_file in validation_files:
                        cell_type = "mast_cells"
                        if apply_log:
                                    trained_predictor.validate(validation_file, show_plot = False,cell_type=cell_type,
                                                               transformation=
                                    [decorate_return_coordinates_after_rearrangements(return_coordinates_after_deletion, intervals=[deletion])],
                                                               validators=[trained_predictor.plot_juicebox,
                                                                   trained_predictor.decorate_scc(
                                                                   trained_predictor.scc, h=h, scc_file=source_path+"scc.r",cell_type=cell_type, calculate_scc=True),
                                                                   ], out_dir=out_dir)
                                                           # validators=[trained_predictor.r2score, trained_predictor.scc,
                                                           #             trained_predictor.plot_matrix, trained_predictor.plot_juicebox])
                        else:
                            trained_predictor.validate(validation_file, show_plot=False,cell_type=cell_type,
                                                       transformation=[decorate_oe2obs(oe2obs,
                                                                                      expected_folder=expected_folder,
                                                                                      cell_type=cell_type,
                                                                                      coeff_fname="coefficient." + cell_type + ".5KB.txt")],
                                                       validators=[trained_predictor.plot_juicebox,
                                                                   trained_predictor.decorate_scc(
                                                           trained_predictor.scc, h=h, scc_file=source_path+"scc.r",cell_type=cell_type,
                                                                       out_dir=out_dir + "scc/" + cell_type + "/"
                                                                   ),
                                                           ])
                                                           # validators=[trained_predictor.r2score, trained_predictor.scc,
                                                           #             trained_predictor.plot_matrix, trained_predictor.plot_juicebox],
                                                           # )
                    #my_plot_matrix = partial(trained_predictor.plot_matrix,juicebox=True)
logging.info(datetime.datetime.now())