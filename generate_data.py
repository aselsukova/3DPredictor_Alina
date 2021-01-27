import sys
import os
import logging
from ChiPSeqReader import ChiPSeqReader
from Contacts_reader import ContactsReader
from hicFileReader import hicReader
from fastaFileReader import fastaReader
from RNASeqReader import RNAseqReader
from TssReader import TssReader
from E1_Reader import E1Reader,fileName2binsize
from shared import Interval, Parameters
from DataGenerator import generate_data
from PredictorGenerators import E1PredictorGenerator,ChipSeqPredictorGenerator, \
                SmallChipSeqPredictorGenerator,SmallE1PredictorGenerator, \
                SitesOrientPredictorGenerator, OrientBlocksPredictorGenerator, ConvergentPairPredictorGenerator, Distance_to_TSS_PG
from VectPredictorGenerators import loopsPredictorGenerator
from LoopReader import LoopReader
import pandas as pd
import pickle
import datetime

#generate_data.py chr_num input output cell_type bin_size

head_folder_path = os.path.dirname(os.path.abspath(sys.argv[0]))+"/3Dpredictor"
source_path = os.path.dirname(os.path.abspath(sys.argv[0]))+"/3Dpredictor/source"
source_path2 = os.path.dirname(os.path.abspath(sys.argv[0]))+"/3Dpredictor/nn/source"
sys.path.append(source_path)
sys.path.append(source_path2)
chr_num=sys.argv[1]
chr_nums=chr_num.split(",")

logging.basicConfig(format='%(asctime)s %(name)s: %(message)s', datefmt='%I:%M:%S', level=logging.DEBUG)

if __name__ == '__main__': #Requered for parallization, at least on Windows
    #,"chr10", "chr1"]:
    logging.basicConfig(format='%(asctime)s %(name)s: %(message)s', datefmt='%I:%M:%S', level=logging.DEBUG)

    input_folder = sys.argv[2]
    output_folder = sys.argv[3]
    cell_type= sys.argv[4]
    params = Parameters()
    params.binsize = sys.argv[5] #sequence resolution of contacts data. Use for finding of normalized coefficient file
    #params.window_size = params.binsize #region around contact to be binned for predictors. Usually equal to binsize
    params.mindist = params.binsize*2+1 #minimum distance between contacting regions
    params.maxdist = 1500000
    # params.sample_size = 500000 #how many contacts write to file
    params.sample_size = 2 #how many contacts write to file
    params.conttype = conttype
    params.max_cpus = 11
    params.keep_only_orient=False
    params.use_only_contacts_with_CTCF = "all_cont"#"all_cont"#"cont_with_CTCF"#"#"all_cont"#"cont_with_CTCF "
    rearrangement=False
    deletion = Interval("chr5", 75852814, 75881252)
    write_all_chrms_in_file=False #set True if you want write training file consisting several chromosomes
    fill_empty_contacts = False #set True if you want use all contacts in region, without empty contacts

    logging.getLogger(__name__).debug("Using input folder "+input_folder)

    # Read contacts data
    genome = fastaReader(input_folder + "sequence/hg38/hg38.fa", name="hg38",useOnlyChromosomes=["chr5"])
    genome = genome.read_data()
    # print(genome)
    # print(genome.data.keys())
    now = datetime.datetime.now()
    params.contacts_reader = hicReader(fname=input_folder + "H1/4DNFI2TK7L2F.hic", genome=genome, binsize=params.binsize)
    # params.contacts_reader = hicReader(fname=input_folder + "H1/control.chr4.50KBhic", genome=genome, binsize=1000)
    params.contacts_reader = params.contacts_reader.read_data(fill_empty_contacts=fill_empty_contacts, noDump=True)

    if params.use_only_contacts_with_CTCF == "cont_with_CTCF":
        params.proportion = 1
        params.contacts_reader.use_contacts_with_CTCF(CTCFfile=input_folder + "NPC/CTCF/GSE96107_NPC_CTCF.IDR0.05.filt.narrowPeak",
                                                        maxdist=params.maxdist, proportion=params.proportion, keep_only_orient=params.keep_only_orient,
                                                        CTCForientfile=input_folder + "NPC/CTCF/GSE96107_NPC_CTCF.IDR0.05.filt.narrowPeak-orient.bed")
        params.use_only_contacts_with_CTCF += str(params.contacts_reader.conts_with_ctcf)
        #make deletion
    if rearrangement:
        params.contacts_reader.delete_region(deletion)
        # Read CTCF data
    logging.info('create CTCF_PG')
    params.ctcf_reader = ChiPSeqReader(input_folder + "NPC/CTCF/GSE96107_NPC_CTCF.IDR0.05.filt.narrowPeak",
                                                            name="CTCF")
    params.ctcf_reader.read_file()
    params.ctcf_reader.set_sites_orientation(
            input_folder + "NPC/CTCF/GSE96107_NPC_CTCF.IDR0.05.filt.narrowPeak-orient.bed")
    if params.keep_only_orient:
        params.ctcf_reader.keep_only_with_orient_data()
    if rearrangement:
        params.ctcf_reader.delete_region(deletion)
    OrientCtcfpg = SitesOrientPredictorGenerator(params.ctcf_reader,
                                                     N_closest=4)
    NotOrientCTCFpg = SmallChipSeqPredictorGenerator(params.ctcf_reader,
                                                         params.binsize,
                                                         N_closest=4)

        # Read CTCF data and drop sites w/o known orientation
    params.ctcf_reader_orientOnly = ChiPSeqReader(input_folder + "NPC/CTCF/GSE96107_NPC_CTCF.IDR0.05.filt.narrowPeak",
                                                            name="CTCF")
    params.ctcf_reader_orientOnly.read_file()
    params.ctcf_reader_orientOnly.set_sites_orientation(
            input_folder + "NPC/CTCF/GSE96107_NPC_CTCF.IDR0.05.filt.narrowPeak-orient.bed")
    params.ctcf_reader_orientOnly.keep_only_with_orient_data()
    if rearrangement:
        params.ctcf_reader_orientOnly.delete_region(deletion)

    OrientBlocksCTCFpg = OrientBlocksPredictorGenerator(params.ctcf_reader_orientOnly,
                                                             params.binsize)
    ConvergentPairPG = ConvergentPairPredictorGenerator(params.ctcf_reader, binsize=params.binsize)

        # #Read other chip-seq data
        # logging.info('create chipPG')
        # chipPG = []
        # filenames_df = pd.read_csv(input_folder + "H1/Chip-seq/filenames.csv")
        # # assert len(os.listdir(input_folder + 'peaks/')) - 1 == len(filenames_df['name'])
        # # print(len(os.listdir(input_folder + 'peaks/')))
        # # print(len(filenames_df['name']))
        # # proteins=set(["RAD21", "SMC3", "POLR2A", "H3K27ac", "H3K27me3", "DNase-seq", "H3K9me3", "H3K4me1", "H3K4me2", "H3K4me3", "YY1"])
        # for index, row in filenames_df.iterrows():
        #     # if row["name"] in proteins:
        #         params.chip_reader = ChiPSeqReader(input_folder + 'H1/Chip-seq/' + row["filename"], name=row['name'])
        #         params.chip_reader.read_file()
        #         chipPG.append(SmallChipSeqPredictorGenerator(params.chip_reader,params.window_size,N_closest=4))

        #Read RNA-Seq data
    params.RNAseqReader = RNAseqReader(fname=input_folder + "mast_cells/RNA-seq/Our_rna_seq/StringTie on collection 64_ Gene abundance estimates/wt_1.fq.tabular",
                                           name="RNA")
    params.RNAseqReader.read_file(rename={ "Gene ID": "gene",
                              "Start": "start",
                              "End": "end",
                              "Reference": "chr",
                              "FPKM": "sigVal"},
                      sep="\t")
    if rearrangement:
        params.RNAseqReader.delete_region(deletion)
    RNAseqPG = SmallChipSeqPredictorGenerator(params.RNAseqReader,
                                                  window_size=params.binsize,
                                                  N_closest=3)

    params.pgs = [OrientCtcfpg, NotOrientCTCFpg, OrientBlocksCTCFpg,ConvergentPairPG,RNAseqPG]#+chipPG#+cagePG+metPG
        # # Generate train
        # train_chrs=[]
        # [train_chrs.append("chr"+chr) for chr in chr_nums]
        # if write_all_chrms_in_file:
        #     train_file_name="training.RandOn"+ str(params)
        #     params.out_file=output_folder+"_".join(train_chrs)+train_file_name
        # for trainChrName in train_chrs:
        #     print(trainChrName)
        #     # training_file_name = "training.RandOn" + trainChrName + str(params) + ".txt"
        #
        #     params.sample_size = len(params.contacts_reader.data[trainChrName])
        #     params.interval = Interval(trainChrName,
        #                           params.contacts_reader.get_min_contact_position(trainChrName),
        #                           params.contacts_reader.get_max_contact_position(trainChrName))
        #
        #     # params.out_file = output_folder + training_file_name
        #     if not write_all_chrms_in_file:
        #         train_file_name = "training.RandOn" + str(params) + ".txt"
        #         params.out_file = output_folder + params.interval.toFileName() + train_file_name
        #     generate_data(params,saveFileDescription=True)
        #     if not write_all_chrms_in_file:
        #         del(params.out_file)
        #     del (params.sample_size)


        # Generate test
    validate_chrs=[]
    [validate_chrs.append("chr"+chr) for chr in chr_nums]#,"chr16", "chr17"]#, "chr18"]#, "chr18", "chr19", "chr20"]#,"chr14", "chr15"]
    if write_all_chrms_in_file:
        validation_file_name = "validatingOrient." + str(params) + ".txt"
        params.out_file = output_folder + "_".join(validate_chrs) + validation_file_name
    for validateChrName in validate_chrs:
        print("chromosome", validateChrName)
        interval=Interval("chr5", 74000000, 76400000)
        params.sample_size = len(params.contacts_reader.data[validateChrName])

            # params.interval = Interval(validateChrName,
            #                            params.contacts_reader.get_min_contact_position(validateChrName),
            #                            params.contacts_reader.get_max_contact_position(validateChrName))
        params.interval = interval
        logging.getLogger(__name__).info("Generating validation dataset for interval "+str(params.interval))
        if not write_all_chrms_in_file:
            validation_file_name = "validatingOrient." + str(params) + ".txt"
            params.out_file = output_folder + "NPC_1Kb"+params.interval.toFileName() + validation_file_name
        generate_data(params)
        if not write_all_chrms_in_file:
            del(params.out_file)
        del (params.sample_size)

        # for interval in [Interval("chr2", 118000000, 129000000)]:
        # #                  Interval("chr10", 47900000, 53900000),
        # #                  Interval("chr10", 15000000, 20000000),
        # #                  Interval("chr10",36000000,41000000)]:
        # # Interval("chr1", 100000000, 110000000)]:
        #    logging.getLogger(__name__).info("Generating validation dataset for interval "+str(interval))
        #    validation_file_name = "validatingOrient." + str(params) + ".txt"
        #    params.interval = interval
        #    params.out_file = output_folder + params.interval.toFileName() + validation_file_name
        #    generate_data(params)

        # for object in [params.contacts_reader]+params.pgs:
        #     lostInterval = Interval("chr1",103842568,104979840)
        #     object.delete_region(lostInterval)
        #     params.interval = Interval("chr1",100000000,109000000)
        #     logging.getLogger(__name__).info("Saving data to file "+params.interval.toFileName() + "DEL." + lostInterval.toFileName()+validation_file_name)
        # params.out_file = params.interval.toFileName() + "DEL." + lostInterval.toFileName()+validation_file_name
        # generate_data(params)